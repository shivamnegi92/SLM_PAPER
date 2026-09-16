"""Layer-onset sweep: patch ONLY the entity token position at a SINGLE layer L
(one site per pair) clean->corrupt, and measure faithfulness as a function of L.

This reveals the DEPTH at which the tracked state becomes causally sufficient at
the entity position -- the core mechanism figure, and it avoids the trivial
"patch the final residual" recovery by never touching the final position.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import dataset as ds
from localize import Harness, pick_device, build_fewshot


def prep_pairs(h, task, n, fewshot, min_len, max_len, seed):
    pairs = ds.generate(task, n + fewshot, seed=seed, min_len=min_len, max_len=max_len)
    ds.self_check(pairs)
    prefix = build_fewshot(pairs, fewshot, task)
    out = []
    for p in pairs[:n]:
        cid, kid = h.first_id(p.clean_target), h.first_id(p.corrupt_target)
        if cid == kid:
            continue
        clean_ids = h.encode(prefix + p.clean_prompt)
        corrupt_ids = h.encode(prefix + p.corrupt_prompt)
        if clean_ids.shape[1] != corrupt_ids.shape[1]:
            continue
        cw, kw = clean_ids[0].tolist(), corrupt_ids[0].tolist()
        diffs = [j for j, (a, b) in enumerate(zip(cw, kw)) if a != b]
        if len(diffs) != 1:
            continue
        dpos = diffs[0]
        clean_outs = h.cache_layer_outputs(clean_ids)
        ld_c = h.metric(h.logits_last(corrupt_ids), cid, kid).item()
        ld_cl = h.metric(h.logits_last(clean_ids), cid, kid).item()
        out.append((cid, kid, clean_ids, corrupt_ids, dpos, clean_outs, ld_c, ld_cl))
    return out


def sweep(h, prepped):
    faith = np.zeros(h.n_layers)
    for L in range(h.n_layers):
        vals = []
        for (cid, kid, clean_ids, corrupt_ids, dpos, clean_outs, ld_c, ld_cl) in prepped:
            ld_p = h.patch_sites(corrupt_ids, clean_outs, [(L, dpos)], cid, kid)
            denom = (ld_cl - ld_c) or 1e-6
            vals.append((ld_p - ld_c) / denom)
        faith[L] = float(np.mean(vals))
    return faith


def run(args):
    device = pick_device(args.device)
    h = Harness(args.model, device)
    ds.restrict_to_single_token(h.tok)
    curves = {}
    cfg = {"intermediate": (3, 4), "transfer": (1, 2)}
    for task in args.tasks:
        mn, mx = cfg[task]
        prepped = prep_pairs(h, task, args.n, args.fewshot, mn, mx, args.seed)
        print(f"[{task}] usable={len(prepped)}")
        faith = sweep(h, prepped)
        curves[task] = faith.tolist()
        onset = next((L for L in range(h.n_layers) if faith[L] >= 0.5), None)
        peak = int(np.argmax(faith))
        print(f"[{task}] entity-position single-layer faithfulness:")
        for L in range(h.n_layers):
            bar = "#" * int(max(0, faith[L]) * 30)
            print(f"   L{L:2d} {faith[L]:6.1%} {bar}")
        print(f"[{task}] onset (>=50%) at layer {onset}; peak at layer {peak} "
              f"({faith[peak]:.1%})\n")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "layer_sweep.json").write_text(json.dumps(curves, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        for task, c in curves.items():
            ax.plot(range(len(c)), c, marker="o", label=task)
        ax.axhline(0.5, ls="--", c="gray", lw=1)
        ax.set_xlabel("layer patched (entity position only)")
        ax.set_ylabel("faithfulness (logit-diff recovered)")
        ax.set_title("Llama-3.2-3B: causal onset of state-tracking at the entity token")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "layer_sweep.png", dpi=130)
        print(f"saved -> {outdir}/layer_sweep.png")
    except Exception as e:
        print("plot skipped:", e)
    print(f"saved -> {outdir}/layer_sweep.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../llama-3.2-3b")
    ap.add_argument("--tasks", nargs="*", default=["intermediate", "transfer"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--fewshot", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
