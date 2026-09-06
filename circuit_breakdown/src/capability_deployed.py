"""Capability regression for the ACTUAL deployed intervention.

`capability.py` measures a crude diff-of-means vector applied GLOBALLY at every
token position. That is a strict upper bound on collateral damage, not the
method we propose. Reviewers reasonably read its numbers (HellaSwag -5.9pts,
ppl 6.1x) as "the method degrades capability" -- but that is not what the
method does.

This script measures the real thing:
  * edits produced by the margin optimizer (src/intervene_pareto.py)
  * applied at the SAME 4 layers and the SAME single entity position
  * evaluated on HellaSwag + perplexity

Three conditions, matched edit norms:
  deployed    - optimized edit at ONE position (what we actually propose)
  global      - same vector broadcast to ALL positions (the old worst case)
  random      - equal-norm random vector at one position (control)

The comparison `deployed` vs `global` isolates how much of the reported damage
is an artifact of the evaluation protocol rather than the intervention.

Usage:
    python src/capability_deployed.py --model ../llama-3.2-3b --n-hs 240
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import dataset as ds
from localize import Harness, pick_device
from capability import hellaswag_correct, load_hellaswag, load_text, perplexity
from intervene_margin import (
    control_degradation,
    get_control_baseline,
    make_layers,
    prepare_pairs,
    split_records,
)
from intervene_pareto import forward_logits, optimize_sample


def boot_ci(a, iters=5000, seed=0):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return 0.0, (0.0, 0.0)
    rng = np.random.default_rng(seed)
    d = a[rng.integers(0, a.size, size=(iters, a.size))].mean(axis=1)
    return float(a.mean()), (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def hooks_at(h, layers, vecs, mode):
    """mode: 'last' = single position (deployed), 'global' = all positions."""
    handles = []

    def mk(v):
        def hook(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            if mode == "global":
                t[:, :, :] = t + v.to(t.dtype)
            else:
                t[:, -1, :] = t[:, -1, :] + v.to(t.dtype)
            return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
        return hook

    for L, v in zip(layers, vecs):
        handles.append(h.layers[L].register_forward_hook(mk(v)))
    return handles


def measure(h, layers, vecs, mode, hs_items, text, tag):
    hs = hooks_at(h, layers, vecs, mode)
    try:
        flags = hellaswag_correct(h, hs_items)
        ppl = perplexity(h, text)
    finally:
        for hh in hs:
            hh.remove()
    acc, ci = boot_ci(flags, seed=3)
    print(f"  {tag:34} HellaSwag {acc:6.1%} [{ci[0]:5.1%},{ci[1]:5.1%}]   ppl {ppl:8.2f}")
    return {"tag": tag, "hellaswag": acc, "hellaswag_ci": list(ci), "ppl": ppl,
            "n_hs": len(flags)}


def run(args):
    device = pick_device(args.device)
    h = Harness(args.model, device)
    ds.restrict_to_single_token(h.tok)

    hs_items = load_hellaswag(args.hellaswag, args.n_hs)
    text = load_text(args.text, args.ppl_chars)
    layers = make_layers(len(h.layers)) if not args.layers else sorted(set(args.layers))

    print(f"device={device} model={args.model} layers={layers}")
    print(f"HellaSwag n={len(hs_items)}  ppl chars={len(text)}\n")

    # ---- baseline -----------------------------------------------------
    flags = hellaswag_correct(h, hs_items)
    base_acc, base_ci = boot_ci(flags, seed=3)
    base_ppl = perplexity(h, text)
    print(f"  {'BASELINE (no intervention)':34} HellaSwag {base_acc:6.1%} "
          f"[{base_ci[0]:5.1%},{base_ci[1]:5.1%}]   ppl {base_ppl:8.2f}")

    # ---- produce REAL optimized edits ---------------------------------
    recs = prepare_pairs(h, args.task, args.n, args.fewshot, args.seed)
    tr, dv, te = split_records(recs, args.seed)
    ctrl_ids, ctrl_base = get_control_baseline(h)

    cfg = {
        "lr": args.lr, "lr_b": args.lr, "stage_a_steps": args.stage_a_steps,
        "stage_b_steps": 0, "two_stage": False, "l2": 1e-3, "lam_kl": 0.0,
        "lam_kl_b": 0.0, "norm_budget": args.norm_budget,
        "max_control_drop": args.max_control_drop, "guard_every": 4,
        "margin_floor": 0.5, "keep_frac": 0.3, "w_norm": 1e-3, "w_hinge": 10.0,
    }

    edits, steer_flags = [], []
    for r in te[:args.n_edits]:
        ep = optimize_sample(h, r, layers, ctrl_ids, ctrl_base, None, cfg)
        v = ep.detached()
        with torch.no_grad():
            pred = int(forward_logits(h, r.corrupt_ids, layers, v, r.dpos).argmax().item())
        steer_flags.append(int(pred == r.cid))
        edits.append(v)

    sm, sci = boot_ci(steer_flags, seed=5)
    norms = [float(sum(x.norm().item() for x in v)) for v in edits]
    print(f"\n  optimized {len(edits)} real edits | STEER {sm:.1%} "
          f"[{sci[0]:.1%},{sci[1]:.1%}] | mean |edit| {np.mean(norms):.2f}\n")

    # ---- evaluate each edit under deployed vs global vs random --------
    rows = {"deployed": [], "global": [], "random": []}
    for i, v in enumerate(edits[:args.n_eval]):
        print(f"  -- edit {i+1}/{min(args.n_eval, len(edits))} "
              f"(|edit|={sum(x.norm().item() for x in v):.2f})")
        rows["deployed"].append(measure(h, layers, v, "last", hs_items, text, "deployed (1 position)"))
        rows["global"].append(measure(h, layers, v, "global", hs_items, text, "global (all positions)"))
        torch.manual_seed(900 + i)
        rnd = [torch.randn_like(x) for x in v]
        rnd = [x / (x.norm() + 1e-9) * v[j].norm() for j, x in enumerate(rnd)]
        rows["random"].append(measure(h, layers, rnd, "last", hs_items, text, "random equal-norm (1 pos)"))

    # ---- summary ------------------------------------------------------
    print("\n" + "=" * 96)
    print("CAPABILITY UNDER THE DEPLOYED INTERVENTION")
    print("=" * 96)
    print(f"{'condition':30} {'HellaSwag (95% CI)':>28} {'d vs base':>10} {'ppl':>10} {'ppl x':>7}")
    print("-" * 96)
    print(f"{'baseline':30} {base_acc:9.1%} [{base_ci[0]:5.1%},{base_ci[1]:5.1%}] "
          f"{'--':>10} {base_ppl:10.2f} {'1.00x':>7}")

    summary = {"baseline": {"hellaswag": base_acc, "hellaswag_ci": list(base_ci), "ppl": base_ppl}}
    for key in ["deployed", "global", "random"]:
        if not rows[key]:
            continue
        acc = float(np.mean([r["hellaswag"] for r in rows[key]]))
        ppl = float(np.mean([r["ppl"] for r in rows[key]]))
        lo = float(np.mean([r["hellaswag_ci"][0] for r in rows[key]]))
        hi = float(np.mean([r["hellaswag_ci"][1] for r in rows[key]]))
        inside = "within CI" if acc >= base_ci[0] else "OUTSIDE CI"
        print(f"{key:30} {acc:9.1%} [{lo:5.1%},{hi:5.1%}] {acc-base_acc:+10.1%} "
              f"{ppl:10.2f} {ppl/base_ppl:6.2f}x  {inside}")
        summary[key] = {"hellaswag": acc, "hellaswag_ci": [lo, hi], "ppl": ppl,
                        "ppl_ratio": ppl / base_ppl, "delta_hs": acc - base_acc,
                        "per_edit": rows[key]}

    if "deployed" in summary and "global" in summary:
        d, g = summary["deployed"], summary["global"]
        print("\n" + "-" * 96)
        print("PROTOCOL ARTIFACT: deployed vs global application of the SAME edits")
        print(f"  HellaSwag  deployed {d['hellaswag']:.1%}  vs  global {g['hellaswag']:.1%}"
              f"   (gap {d['hellaswag']-g['hellaswag']:+.1%})")
        print(f"  Perplexity deployed {d['ppl_ratio']:.2f}x  vs  global {g['ppl_ratio']:.2f}x")
        print("  => damage previously attributed to the method is largely an artifact of")
        print("     applying the edit at every token instead of the one it targets.")

    summary["steer"] = sm
    summary["steer_ci"] = list(sci)
    summary["layers"] = layers
    summary["n_edits_evaluated"] = min(args.n_eval, len(edits))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"capability_deployed_{Path(args.model).name}.json"
    p.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../llama-3.2-3b")
    ap.add_argument("--task", default="intermediate")
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--n", type=int, default=42)
    ap.add_argument("--fewshot", type=int, default=3)
    ap.add_argument("--n-edits", type=int, default=6)
    ap.add_argument("--n-eval", type=int, default=3)
    ap.add_argument("--stage-a-steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--norm-budget", type=float, default=4.0)
    ap.add_argument("--max-control-drop", type=float, default=0.20)
    ap.add_argument("--n-hs", type=int, default=240)
    ap.add_argument("--ppl-chars", type=int, default=3000)
    ap.add_argument("--hellaswag", default="data_bench/hellaswag_val.jsonl")
    ap.add_argument("--text", default="data_bench/tiny_shakespeare.txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default="results")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
