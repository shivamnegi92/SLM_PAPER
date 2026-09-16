"""Head-level localization: which ATTENTION HEADS implement the tracking move?

Upgrades "layer/position localization" to a real circuit claim (Review W2).
We attribution-patch the per-head contributions (the input to o_proj, sliced by
head) at the entity position and the final position, then verify the top-k heads
with real activation patching vs a random-k head baseline.

Head slice: the input to self_attn.o_proj is [batch, seq, n_heads*head_dim];
head h occupies columns [h*head_dim : (h+1)*head_dim].
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import dataset as ds
from localize import Harness, pick_device, build_fewshot


class HeadHarness(Harness):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        cfg = self.model.config
        self.n_heads = cfg.num_attention_heads
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // self.n_heads)
        self.oprojs = [lyr.self_attn.o_proj for lyr in self.layers]

    @torch.no_grad()
    def cache_head_inputs(self, input_ids):
        outs = []
        handles = []
        for op in self.oprojs:
            handles.append(op.register_forward_pre_hook(
                lambda m, args: outs.append(args[0].detach())))
        self.model(input_ids)
        for h in handles:
            h.remove()
        return outs  # len n_layers, each [1, seq, n_heads*head_dim]

    def head_attribution(self, clean_ids, corrupt_ids, cid, kid, pos):
        """effect[layer, head] via AtP at token position `pos` (may be negative
        index)."""
        clean_in = self.cache_head_inputs(clean_ids)
        captured = []
        handles = []

        def prehook(m, args):
            t = args[0]
            t.retain_grad()
            captured.append(t)
            return None
        for op in self.oprojs:
            handles.append(op.register_forward_pre_hook(prehook))
        self.model.zero_grad(set_to_none=True)
        logits = self.model(corrupt_ids).logits[0, -1]
        m = self.metric(logits, cid, kid)
        m.backward()
        for h in handles:
            h.remove()

        seq = clean_ids.shape[1]
        p = pos if pos >= 0 else seq + pos
        eff = np.zeros((self.n_layers, self.n_heads), dtype=np.float32)
        for L in range(self.n_layers):
            delta = (clean_in[L] - captured[L].detach())[0, p]  # [n_heads*hd]
            grad = captured[L].grad[0, p]                        # [n_heads*hd]
            contrib = (delta * grad).reshape(self.n_heads, self.head_dim).sum(-1)
            eff[L] = contrib.float().cpu().numpy()
        return eff

    def patch_heads(self, corrupt_ids, clean_in, head_sites, cid, kid, pos):
        """head_sites: list of (layer, head). Patch clean head slice at `pos`."""
        by_layer = {}
        for (L, hh) in head_sites:
            by_layer.setdefault(L, []).append(hh)
        handles = []
        seq = corrupt_ids.shape[1]
        p = pos if pos >= 0 else seq + pos

        def make(L):
            heads = by_layer[L]
            ci = clean_in[L]

            def prehook(m, args):
                t = args[0].clone()
                for hh in heads:
                    sl = slice(hh * self.head_dim, (hh + 1) * self.head_dim)
                    t[0, p, sl] = ci[0, p, sl].to(t.dtype)
                return (t,) + tuple(args[1:])
            return prehook
        for L in by_layer:
            handles.append(self.oprojs[L].register_forward_pre_hook(make(L)))
        with torch.no_grad():
            logits = self.model(corrupt_ids).logits[0, -1]
        for h in handles:
            h.remove()
        return self.metric(logits, cid, kid).item()


def run(args):
    device = pick_device(args.device)
    h = HeadHarness(args.model, device)
    ds.restrict_to_single_token(h.tok)
    cfg = {"intermediate": (3, 4), "transfer": (1, 2)}
    mn, mx = cfg[args.task]
    pairs = ds.generate(args.task, args.n + args.fewshot, seed=args.seed, min_len=mn, max_len=mx)
    prefix = build_fewshot(pairs, args.fewshot, args.task)

    usable = []
    for p in pairs[:args.n]:
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
        usable.append((cid, kid, clean_ids, corrupt_ids, diffs[0]))
    print(f"[{args.model}] task={args.task} usable={len(usable)} "
          f"heads/layer={h.n_heads} layers={h.n_layers}")

    # attribute at the FINAL position (where the answer is read out) -> mover heads
    acc = np.zeros((h.n_layers, h.n_heads))
    for (cid, kid, clean_ids, corrupt_ids, dpos) in usable:
        acc += h.head_attribution(clean_ids, corrupt_ids, cid, kid, pos=-1)
    mean_eff = acc / len(usable)

    flat = [(L, hd, mean_eff[L, hd]) for L in range(h.n_layers) for hd in range(h.n_heads)]
    flat.sort(key=lambda x: x[2], reverse=True)
    print("\nTOP MOVER HEADS (attribution at final position):")
    for (L, hd, s) in flat[:10]:
        print(f"  L{L:2d}.H{hd:2d}  score={s:+.4f}")

    # verify: patch top-k heads (final pos) clean->corrupt vs random-k
    def faith(head_sites):
        vals = []
        for (cid, kid, clean_ids, corrupt_ids, dpos) in usable:
            clean_in = h.cache_head_inputs(clean_ids)
            ld_p = h.patch_heads(corrupt_ids, clean_in, head_sites, cid, kid, pos=-1)
            ld_c = h.metric(h.logits_last(corrupt_ids), cid, kid).item()
            ld_cl = h.metric(h.logits_last(clean_ids), cid, kid).item()
            vals.append((ld_p - ld_c) / ((ld_cl - ld_c) or 1e-6))
        return np.array(vals)

    rng = np.random.default_rng(0)
    all_heads = [(L, hd) for L in range(h.n_layers) for hd in range(h.n_heads)]
    print("\nVERIFY (patch heads at final position):")
    curve = {}
    for k in [1, 2, 4, 8, 16]:
        sites = [(L, hd) for (L, hd, _) in flat[:k]]
        fk = faith(sites)
        curve[k] = [float(fk.mean()), float(fk.std())]
        print(f"  top-{k:2d} heads faithfulness = {fk.mean():6.1%} +/- {fk.std():4.1%}")
    rand = []
    for _ in range(args.random_repeats):
        idx = rng.choice(len(all_heads), size=8, replace=False)
        rand.append(faith([all_heads[i] for i in idx]).mean())
    print(f"  random-8 heads faithfulness = {np.mean(rand):.1%} +/- {np.std(rand):.1%}")

    out = {"model": args.model, "task": args.task, "n_heads": h.n_heads,
           "n_layers": h.n_layers,
           "top_heads": [[int(L), int(hd), float(s)] for (L, hd, s) in flat[:15]],
           "faithfulness_curve": curve,
           "random8_mean": float(np.mean(rand)), "random8_std": float(np.std(rand))}
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.model).name
    (outdir / f"heads_{tag}_{args.task}.json").write_text(json.dumps(out, indent=2))
    np.savetxt(outdir / f"heads_{tag}_{args.task}.csv", mean_eff, delimiter=",")
    print(f"\nsaved -> {outdir}/heads_{tag}_{args.task}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../llama-3.2-3b")
    ap.add_argument("--task", default="intermediate", choices=["intermediate", "transfer"])
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--fewshot", type=int, default=3)
    ap.add_argument("--random-repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
