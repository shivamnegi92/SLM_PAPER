"""Inference-time intervention: causally BREAK and STEER the tracking circuit.

Uses a diff-of-means steering direction at a single (layer, entity-position)
site -- derived from the same minimal pairs -- to show weight-free control:

  * STEER (corrupt -> clean): add +alpha * d at the entity position of a CORRUPT
    prompt; the model should now answer the CLEAN target ("as if" the tracked
    entity were the clean one).
  * BREAK (clean -> corrupt): subtract alpha * d at the entity position of a
    CLEAN prompt; tracking accuracy should collapse.

d = mean over pairs of (clean_resid[L,epos] - corrupt_resid[L,epos]).

Also a light capability check: apply the SAME direction at all positions of
generic sentences and measure next-token argmax agreement vs the untouched model
(should stay high -> we didn't lobotomize general behavior).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import dataset as ds
from localize import Harness, pick_device, build_fewshot

GENERIC = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The opposite of hot is",
    "Two plus two equals",
    "The sun rises in the",
    "A dog is a kind of",
    "The first month of the year is",
    "She opened the book and began to",
]


def cache_resid_at(h: Harness, input_ids, layer, pos):
    outs = h.cache_layer_outputs(input_ids)
    return outs[layer][0, pos].clone()  # [hidden]


def add_hook(h, layer, pos, vec):
    """Return a handle that adds vec at (layer, pos). pos<0 allowed (from end)."""
    def hook(m, i, o):
        t = o[0] if isinstance(o, tuple) else o
        p = pos if pos >= 0 else t.shape[1] + pos
        t[:, p, :] = t[:, p, :] + vec.to(t.dtype)
        return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
    return h.layers[layer].register_forward_hook(hook)


def run(args):
    device = pick_device(args.device)
    h = Harness(args.model, device)
    ds.restrict_to_single_token(h.tok)
    print(f"device={device} model={args.model} task={args.task}")

    pairs = ds.generate(args.task, args.n + args.fewshot, seed=args.seed,
                        min_len=args.min_len, max_len=args.max_len)
    ds.self_check(pairs)
    prefix = build_fewshot(pairs, args.fewshot, args.task)
    eval_pairs = pairs[:args.n]

    # assemble usable, position-aligned pairs and the entity position (from end)
    usable = []
    ent_from_end = None
    for p in eval_pairs:
        cid, kid = h.first_id(p.clean_target), h.first_id(p.corrupt_target)
        if cid == kid:
            continue
        clean_ids = h.encode(prefix + p.clean_prompt)
        corrupt_ids = h.encode(prefix + p.corrupt_prompt)
        if clean_ids.shape[1] != corrupt_ids.shape[1]:
            continue
        # entity position = the single differing token position
        cw = clean_ids[0].tolist()
        kw = corrupt_ids[0].tolist()
        diffs = [j for j, (a, b) in enumerate(zip(cw, kw)) if a != b]
        if len(diffs) != 1:
            continue
        efe = clean_ids.shape[1] - 1 - diffs[0]  # from end
        ent_from_end = efe if ent_from_end is None else ent_from_end
        if efe != ent_from_end:
            continue  # keep a consistent entity offset for a shared direction
        usable.append((p, cid, kid, clean_ids, corrupt_ids, diffs[0]))
    print(f"usable={len(usable)}  entity position from end = {ent_from_end}")

    layers = args.layers or [8, 12, 16, 20, 24]
    epos = -(ent_from_end + 1)

    print("\n=== BREAK / STEER by layer (alpha=%.1f) ===" % args.alpha)
    print(f"{'layer':>5} {'steer c->clean%':>15} {'break clean->corrupt%':>22}")
    best = None
    per_layer = {}
    for L in layers:
        # diff-of-means direction at (L, entity pos)
        dsum = None
        cnt = 0
        for (p, cid, kid, clean_ids, corrupt_ids, dpos) in usable:
            cr = cache_resid_at(h, clean_ids, L, dpos)
            kr = cache_resid_at(h, corrupt_ids, L, dpos)
            d = (cr - kr)
            dsum = d if dsum is None else dsum + d
            cnt += 1
        d = dsum / cnt  # mean clean-ward direction

        # STEER: corrupt + alpha*d -> should predict clean target
        steer_ok = 0
        for (p, cid, kid, clean_ids, corrupt_ids, dpos) in usable:
            hd = add_hook(h, L, dpos, args.alpha * d)
            with torch.no_grad():
                pred = h.model(corrupt_ids).logits[0, -1].argmax().item()
            hd.remove()
            steer_ok += (pred == cid)
        # BREAK: clean - alpha*d -> should stop predicting clean target
        break_ok = 0
        for (p, cid, kid, clean_ids, corrupt_ids, dpos) in usable:
            hd = add_hook(h, L, dpos, -args.alpha * d)
            with torch.no_grad():
                pred = h.model(clean_ids).logits[0, -1].argmax().item()
            hd.remove()
            break_ok += (pred != cid)
        sr, br = steer_ok / len(usable), break_ok / len(usable)
        per_layer[L] = {"steer": sr, "break": br}
        print(f"{L:>5} {sr:>14.0%} {br:>21.0%}")
        score = sr + br
        if best is None or score > best[1]:
            best = (L, score, d.detach().clone())

    Lbest, _, dbest = best
    print(f"\nbest control layer = {Lbest}")

    # ---- capability check at the best layer ----
    # apply +alpha*d at ALL positions of generic prompts; measure argmax agreement
    agree = 0
    total = 0
    for g in GENERIC:
        ids = h.encode(g)
        with torch.no_grad():
            base_pred = h.model(ids).logits[0, -1].argmax().item()

        def hook(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            t[:, :, :] = t[:, :, :] + (args.alpha * dbest).to(t.dtype)
            return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
        hh = h.layers[Lbest].register_forward_hook(hook)
        with torch.no_grad():
            new_pred = h.model(ids).logits[0, -1].argmax().item()
        hh.remove()
        agree += (base_pred == new_pred)
        total += 1
    print(f"\nCAPABILITY: generic next-token argmax agreement after applying the "
          f"direction at ALL positions (layer {Lbest}, alpha={args.alpha}): "
          f"{agree}/{total} = {agree/total:.0%}")

    out = {
        "model": args.model, "task": args.task, "alpha": args.alpha,
        "entity_pos_from_end": ent_from_end, "per_layer": per_layer,
        "best_layer": Lbest,
        "capability_argmax_agreement": agree / total,
    }
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"intervention_{args.task}.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {outdir}/intervention_{args.task}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../llama-3.2-3b")
    ap.add_argument("--task", default="intermediate", choices=["transfer", "intermediate"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--fewshot", type=int, default=3)
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=6.0)
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
