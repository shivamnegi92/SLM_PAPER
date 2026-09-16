"""Capability regression: does the BREAK steering direction damage general
language ability? We sweep the intervention strength (alpha) applied GLOBALLY
(worst case, all positions) at the control layer, and measure at each alpha:

  * BREAK rate on the tracking task (higher = stronger circuit control)
  * HellaSwag accuracy (acc_norm, length-normalized) -- a real benchmark
  * Tiny-Shakespeare perplexity -- generic LM quality
  * a RANDOM-direction control of equal norm (is our direction worse than noise?)

This legacy global sweep is a stress test, not a proven upper bound on
single-position damage. Active-prefix evaluation uses these scorers through
capability_deployed.py and records a separate, versioned protocol.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import torch

import dataset as ds
from localize import Harness, pick_device, build_fewshot


# ---------------- benchmark loaders ----------------
def load_hellaswag(path, n):
    items = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("label", "") == "":
                continue
            items.append((r["ctx"], r["endings"], int(r["label"])))
            if len(items) >= n:
                break
    return items


def load_text(path, max_chars):
    return Path(path).read_text()[:max_chars]


# ---------------- scoring ----------------
@torch.no_grad()
def ending_logprob(h, ctx, ending, intervention=None):
    """Length-normalized log-prob of `ending` tokens given `ctx`."""
    ctx_ids = h.tok(ctx, add_special_tokens=True)["input_ids"]
    full_ids = h.tok(ctx + " " + ending, add_special_tokens=True)["input_ids"]
    if not ctx_ids or full_ids[:len(ctx_ids)] != ctx_ids:
        raise ValueError("Context tokens must be an exact prefix of the scored sequence")
    cont = full_ids[len(ctx_ids):]
    if len(cont) == 0:
        return -1e9
    ids = torch.tensor([full_ids], device=h.device)
    scope = intervention(len(ctx_ids) - 1) if intervention is not None else nullcontext()
    with scope:
        logits = h.model(ids).logits[0]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    total = 0.0
    for i, tid in enumerate(cont):
        pos = len(ctx_ids) + i - 1  # predicts token at len(ctx_ids)+i
        total += logprobs[pos, tid].item()
    return total / len(cont)


@torch.no_grad()
def hellaswag_correct(h, items, intervention=None):
    flags = []
    for ctx, endings, label in items:
        scores = [ending_logprob(h, ctx, ending, intervention=intervention)
                  for ending in endings]
        flags.append(int(int(np.argmax(scores)) == label))
    return np.array(flags)


@torch.no_grad()
def continuation_nll(h, text, max_tokens=768, intervention=None, prefix_tokens=1):
    """Per-token negative log likelihood after a fixed prefix, including BOS."""
    if max_tokens < 2:
        raise ValueError("Perplexity needs at least two tokens")
    ids = h.tok(text, add_special_tokens=True)["input_ids"][:max_tokens]
    if not 1 <= prefix_tokens < len(ids):
        raise ValueError("Perplexity needs a nonempty prefix and a scored continuation")
    ids = torch.tensor([ids], device=h.device)
    scope = intervention(prefix_tokens - 1) if intervention is not None else nullcontext()
    with scope:
        logits = h.model(ids).logits[0].float()
    logprobs = torch.log_softmax(logits, dim=-1)
    nll = []
    for i in range(prefix_tokens, ids.shape[1]):
        nll.append(-logprobs[i - 1, ids[0, i]].item())
    return np.asarray(nll, dtype=np.float64)


def perplexity(h, text, max_tokens=768, intervention=None, prefix_tokens=1):
    nll = continuation_nll(h, text, max_tokens, intervention, prefix_tokens)
    return float(np.exp(nll.mean()))


# ---------------- break direction ----------------
def break_direction(h, task, n, fewshot, min_len, max_len, layer, seed):
    pairs = ds.generate(task, n + fewshot, seed=seed, min_len=min_len, max_len=max_len)
    prefix = build_fewshot(pairs, fewshot, task)
    dsum, cnt = None, 0
    tracking = []  # (clean_ids, cid) for break-rate
    ent_from_end = None
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
        cr = h.cache_layer_outputs(clean_ids)[layer][0, dpos].clone()
        kr = h.cache_layer_outputs(corrupt_ids)[layer][0, dpos].clone()
        d = cr - kr
        dsum = d if dsum is None else dsum + d
        cnt += 1
        tracking.append((clean_ids, cid, dpos))
    return dsum / cnt, tracking


def add_global_hook(h, layer, vec):
    def hook(m, i, o):
        t = o[0] if isinstance(o, tuple) else o
        t[:, :, :] = t[:, :, :] + vec.to(t.dtype)
        return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
    return h.layers[layer].register_forward_hook(hook)


@torch.no_grad()
def break_rate(h, layer, tracking, vec):
    ok = 0
    for (clean_ids, cid, dpos) in tracking:
        hd = add_global_hook(h, layer, vec)  # global == worst case
        pred = h.model(clean_ids).logits[0, -1].argmax().item()
        hd.remove()
        ok += (pred != cid)
    return ok / len(tracking)


def boot_ci(flags, iters=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(flags)
    means = [rng.choice(flags, n, replace=True).mean() for _ in range(iters)]
    return float(np.mean(flags)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run(args):
    device = pick_device(args.device)
    h = Harness(args.model, device)
    ds.restrict_to_single_token(h.tok)
    print(f"device={device} model={args.model} layer={args.layer}")

    d, tracking = break_direction(h, args.task, args.n_track, args.fewshot,
                                  args.min_len, args.max_len, args.layer, args.seed)
    dnorm = d.norm().item()
    print(f"break direction norm={dnorm:.3f}  tracking pairs={len(tracking)}")

    hs = load_hellaswag(args.hellaswag, args.n_hs)
    text = load_text(args.shakespeare, args.ppl_chars)
    print(f"hellaswag items={len(hs)}  ppl text chars={len(text)}")

    # baseline
    base_flags = hellaswag_correct(h, hs)
    base_acc, base_lo, base_hi = boot_ci(base_flags)
    base_ppl = perplexity(h, text)
    print(f"\nBASELINE  hellaswag acc={base_acc:.1%} [{base_lo:.1%},{base_hi:.1%}]  ppl={base_ppl:.2f}")

    rng = np.random.default_rng(args.seed)
    rand = torch.randn_like(d)
    rand = rand / rand.norm() * dnorm  # equal-norm random control

    rows = []
    print(f"\n{'alpha':>6} {'break%':>7} {'HS acc [95% CI]':>22} {'ppl':>8} {'rand HS':>8} {'rand ppl':>9}")
    for a in args.alphas:
        vec = a * d
        br = break_rate(h, args.layer, tracking, -vec)  # BREAK subtracts direction
        hd = add_global_hook(h, args.layer, vec)
        flags = hellaswag_correct(h, hs)
        ppl = perplexity(h, text)
        hd.remove()
        acc, lo, hi = boot_ci(flags)
        # random control at same alpha/norm
        hd = add_global_hook(h, args.layer, a * rand)
        rflags = hellaswag_correct(h, hs)
        rppl = perplexity(h, text)
        hd.remove()
        racc, _, _ = boot_ci(rflags)
        within = "within CI" if (lo <= base_acc <= hi or acc >= base_lo) else "DEGRADED"
        rows.append({"alpha": a, "break_rate": br, "hs_acc": acc, "hs_lo": lo,
                     "hs_hi": hi, "ppl": ppl, "rand_hs": racc, "rand_ppl": rppl,
                     "verdict": within})
        print(f"{a:>6.1f} {br:>6.0%} {acc:>8.1%} [{lo:.1%},{hi:.1%}] {ppl:>8.2f} "
              f"{racc:>7.1%} {rppl:>9.2f}  {within}")

    # find the sweet spot: max break with HS acc within base CI
    sweet = [r for r in rows if r["break_rate"] >= 0.8 and r["hs_acc"] >= base_lo]
    best = max(sweet, key=lambda r: r["break_rate"]) if sweet else None
    print("\nSWEET SPOT (break>=80%% and HS acc within base 95%% CI):",
          f"alpha={best['alpha']} break={best['break_rate']:.0%} HS={best['hs_acc']:.1%}"
          if best else "none -- break requires alpha that degrades HS")

    out = {"model": args.model, "task": args.task, "layer": args.layer,
           "baseline": {"hs_acc": base_acc, "hs_lo": base_lo, "hs_hi": base_hi,
                        "ppl": base_ppl},
           "direction_norm": dnorm, "sweep": rows,
           "sweet_spot": best}
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.model).name
    (outdir / f"capability_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {outdir}/capability_{tag}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../llama-3.2-3b")
    ap.add_argument("--task", default="intermediate")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--alphas", type=float, nargs="*",
                    default=[0.0, 2.0, 4.0, 6.0, 8.0, 12.0])
    ap.add_argument("--n-track", type=int, default=30)
    ap.add_argument("--n-hs", type=int, default=120)
    ap.add_argument("--ppl-chars", type=int, default=3000)
    ap.add_argument("--fewshot", type=int, default=3)
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--hellaswag", default="data_bench/hellaswag_val.jsonl")
    ap.add_argument("--shakespeare", default="data_bench/tinyshakespeare.txt")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
