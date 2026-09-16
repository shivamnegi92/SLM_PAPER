"""Causal localization of state-tracking in Llama-3.2-3B via attribution patching.

Pipeline (all local, fp32, no TransformerLens):
  1. Load base Llama-3.2-3B (fp32, MPS/CPU).
  2. Build clean/corrupt minimal pairs (src/dataset.py), few-shot prefixed.
  3. Baseline: does the model actually solve the task? (accuracy + logit diff)
  4. Attribution patching (AtP): approximate the effect of patching each
     (layer x position) residual-stream site, in ~2 passes/pair.
  5. Verify: real activation-patch the top-k sites clean->corrupt and measure
     faithfulness (normalized logit-diff recovered) vs a random-k baseline.
  6. Save heatmap PNG + CSV + results JSON; print a summary table.

Metric (see METRICS.md): logit_diff = logit(clean_target) - logit(corrupt_target)
at the final position. faithfulness = (LD_patched - LD_corrupt)/(LD_clean - LD_corrupt).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import dataset as ds


def pick_device(flag: str) -> str:
    if flag != "auto":
        return flag
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Harness:
    def __init__(self, model_path: str, device: str, dtype=torch.float32):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, low_cpu_mem_usage=True)
        self.model.to(device)
        self.model.eval()
        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)

    def first_id(self, word: str) -> int:
        # token the word contributes IN CONTEXT (after a dummy word), robust to
        # BPE vs SentencePiece leading-space handling.
        base = self.tok("the", add_special_tokens=False)["input_ids"]
        ext = self.tok("the " + word, add_special_tokens=False)["input_ids"]
        return ext[len(base)]

    def encode(self, text: str) -> torch.Tensor:
        return self.tok(text, return_tensors="pt").input_ids.to(self.device)

    @torch.no_grad()
    def logits_last(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids).logits[0, -1]

    def metric(self, logits_last: torch.Tensor, cid: int, kid: int) -> torch.Tensor:
        return logits_last[cid] - logits_last[kid]

    # ---- caching ----
    @torch.no_grad()
    def cache_layer_outputs(self, input_ids: torch.Tensor) -> list[torch.Tensor]:
        outs: list[torch.Tensor] = []
        handles = []
        for lyr in self.layers:
            handles.append(lyr.register_forward_hook(
                lambda m, i, o: outs.append((o[0] if isinstance(o, tuple) else o).detach())))
        self.model(input_ids)
        for h in handles:
            h.remove()
        return outs  # len n_layers, each [1, seq, hidden]

    # ---- attribution patching (AtP) ----
    def attribution(self, clean_ids, corrupt_ids, cid, kid):
        """Return effect[layer, pos] approx of patching clean->corrupt site."""
        clean_outs = self.cache_layer_outputs(clean_ids)  # detached
        captured: list[torch.Tensor] = []
        handles = []

        def hook(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            t.retain_grad()
            captured.append(t)
        for lyr in self.layers:
            handles.append(lyr.register_forward_hook(hook))
        self.model.zero_grad(set_to_none=True)
        logits = self.model(corrupt_ids).logits[0, -1]
        m = self.metric(logits, cid, kid)
        m.backward()
        for h in handles:
            h.remove()

        seq = clean_ids.shape[1]
        eff = np.zeros((self.n_layers, seq), dtype=np.float32)
        for L in range(self.n_layers):
            delta = (clean_outs[L] - captured[L].detach())  # [1,seq,hid]
            grad = captured[L].grad                          # [1,seq,hid]
            e = (delta * grad).sum(dim=-1)[0]                # [seq]
            eff[L] = e.float().cpu().numpy()
        return eff  # positive => patching this site moves metric toward clean

    # ---- real activation patching of a set of (layer,pos) sites ----
    def patch_sites(self, corrupt_ids, clean_outs, sites, cid, kid):
        """sites: list of (layer, pos). Patch clean_outs into corrupt run."""
        by_layer: dict[int, list[int]] = {}
        for (L, p) in sites:
            by_layer.setdefault(L, []).append(p)
        handles = []

        def make_hook(L):
            positions = by_layer[L]
            clean_t = clean_outs[L]

            def hook(m, i, o):
                if isinstance(o, tuple):
                    t = o[0]
                    t[:, positions, :] = clean_t[:, positions, :].to(t.dtype)
                    return (t,) + tuple(o[1:])
                else:
                    o[:, positions, :] = clean_t[:, positions, :].to(o.dtype)
                    return o
            return hook
        for L in by_layer:
            handles.append(self.layers[L].register_forward_hook(make_hook(L)))
        with torch.no_grad():
            logits = self.model(corrupt_ids).logits[0, -1]
        for h in handles:
            h.remove()
        return self.metric(logits, cid, kid).item()


def build_fewshot(pairs, k, task):
    """Few-shot prefix from the LAST k pairs (held out from eval)."""
    shots = pairs[-k:] if k > 0 else []
    prefix = ""
    for s in shots:
        prefix += f"{s.clean_prompt} {s.clean_target}.\n"
    return prefix


def run(args):
    device = pick_device(args.device)
    print(f"device={device} dtype=fp32 model={args.model}")
    t0 = time.time()
    h = Harness(args.model, device)
    print(f"loaded in {time.time()-t0:.1f}s | layers={h.n_layers}")

    nc, np_ = ds.restrict_to_single_token(h.tok)
    print(f"single-token vocab for this tokenizer: {nc} cities, {np_} people")

    pairs = ds.generate(args.task, args.n + args.fewshot, seed=args.seed,
                        min_len=args.min_len, max_len=args.max_len)
    ds.self_check(pairs)
    fewshot_pairs = pairs[args.n:] if args.fewshot else []
    eval_pairs = pairs[:args.n]
    prefix = build_fewshot(pairs, args.fewshot, args.task)

    # ---- baseline accuracy + logit diffs ----
    correct = 0
    ld_clean_all, ld_corrupt_all = [], []
    usable = []
    for p in eval_pairs:
        cid, kid = h.first_id(p.clean_target), h.first_id(p.corrupt_target)
        if cid == kid:
            continue
        clean_ids = h.encode(prefix + p.clean_prompt)
        corrupt_ids = h.encode(prefix + p.corrupt_prompt)
        if clean_ids.shape[1] != corrupt_ids.shape[1]:
            continue  # keep positions aligned
        lc = h.logits_last(clean_ids)
        pred = lc.argmax().item()
        if pred == cid:
            correct += 1
        ld_clean_all.append(h.metric(lc, cid, kid).item())
        lk = h.logits_last(corrupt_ids)
        ld_corrupt_all.append(h.metric(lk, cid, kid).item())
        usable.append((p, cid, kid, clean_ids, corrupt_ids))
    acc = correct / max(1, len(usable))
    print(f"\nBASELINE  n_usable={len(usable)}  next-token acc={acc:.1%}")
    print(f"  logit_diff clean  mean={np.mean(ld_clean_all):+.3f}")
    print(f"  logit_diff corrupt mean={np.mean(ld_corrupt_all):+.3f}")

    # ---- attribution patching, averaged over pairs ----
    seqs = [c.shape[1] for (_, _, _, c, _) in usable]
    maxseq = max(seqs)
    acc_eff = np.zeros((h.n_layers, maxseq), dtype=np.float64)
    counts = np.zeros(maxseq, dtype=np.int64)
    t1 = time.time()
    for j, (p, cid, kid, clean_ids, corrupt_ids) in enumerate(usable):
        eff = h.attribution(clean_ids, corrupt_ids, cid, kid)
        s = eff.shape[1]
        # right-align on the final token so the prediction position lines up
        acc_eff[:, maxseq - s:] += eff
        counts[maxseq - s:] += 1
        if (j + 1) % 5 == 0:
            print(f"  attribution {j+1}/{len(usable)}  ({(time.time()-t1)/(j+1):.2f}s/pair)")
    counts[counts == 0] = 1
    mean_eff = acc_eff / counts[None, :]

    # ---- precompute per-pair fixed quantities ONCE (big speedup) ----
    prep = []  # (off, clean_outs, ld_c, ld_cl, corrupt_ids, cid, kid, s)
    for (p, cid, kid, clean_ids, corrupt_ids) in usable:
        s = clean_ids.shape[1]
        off = maxseq - s
        clean_outs = h.cache_layer_outputs(clean_ids)
        ld_c = h.metric(h.logits_last(corrupt_ids), cid, kid).item()
        ld_cl = h.metric(h.logits_last(clean_ids), cid, kid).item()
        prep.append((off, clean_outs, ld_c, ld_cl, corrupt_ids, cid, kid, s))

    # ---- rank sites, verify top-k vs random-k ----
    flat = [(L, pos, mean_eff[L, pos]) for L in range(h.n_layers)
            for pos in range(maxseq)]
    flat.sort(key=lambda x: x[2], reverse=True)  # most positive first
    topk = [(L, pos) for (L, pos, _) in flat[:args.topk]]

    def faithfulness_for(sites_fn):
        sites = sites_fn()
        vals = []
        for (off, clean_outs, ld_c, ld_cl, corrupt_ids, cid, kid, s) in prep:
            local_sites = [(L, pos - off) for (L, pos) in sites
                           if 0 <= pos - off < s]
            if not local_sites:
                continue
            ld_p = h.patch_sites(corrupt_ids, clean_outs, local_sites, cid, kid)
            denom = (ld_cl - ld_c) or 1e-6
            vals.append((ld_p - ld_c) / denom)
        return np.array(vals)

    rng = np.random.default_rng(0)
    all_sites = [(L, pos) for L in range(h.n_layers) for pos in range(maxseq)]

    # (a) faithfulness-vs-circuit-size curve + minimal circuit
    ks = [1, 2, 4, 8, 16, 32, 64]
    ks = [k for k in ks if k <= len(flat)]
    print("\nVERIFY  faithfulness vs circuit size (top-k attribution sites):")
    curve = {}
    min_k_80 = None
    for k in ks:
        sites_k = [(L, pos) for (L, pos, _) in flat[:k]]
        fk = faithfulness_for(lambda s=sites_k: s)
        curve[k] = (float(fk.mean()), float(fk.std()))
        print(f"  k={k:3d}  faithfulness={fk.mean():6.1%} +/- {fk.std():4.1%}")
        if min_k_80 is None and fk.mean() >= 0.80:
            min_k_80 = k
    print(f"  --> minimal circuit for >=80%%: k={min_k_80}")

    # (b) random-k baseline at the reference k
    ref_k = args.topk
    rand_runs = []
    for _ in range(args.random_repeats):
        idx = rng.choice(len(all_sites), size=ref_k, replace=False)
        rsites = [all_sites[i] for i in idx]
        rand_runs.append(faithfulness_for(lambda s=rsites: s).mean())
    top_ref = faithfulness_for(lambda: [(L, pos) for (L, pos, _) in flat[:ref_k]])
    print(f"\n  top-{ref_k} faithfulness   = {top_ref.mean():.1%} +/- {top_ref.std():.1%}")
    print(f"  random-{ref_k} faithfulness= {np.mean(rand_runs):.1%} +/- {np.std(rand_runs):.1%}")

    # (c) position-resolved attribution (which token positions carry tracking)
    pos_score = mean_eff.sum(axis=0)
    top_pos = np.argsort(pos_score)[::-1][:5]
    print("\nTOP POSITIONS by summed attribution (right-aligned; last=prediction):")
    for pp in top_pos:
        tag = " <final>" if pp == maxseq - 1 else ""
        print(f"  pos {pp:3d} (from end {maxseq-1-pp:2d}): score={pos_score[pp]:+.4f}{tag}")

    # (d) REVIEWER-PROOF non-trivial circuit: exclude final position AND last 2
    #     layers, so the recovery cannot be the trivial downstream readout.
    nontrivial = [(L, pos) for (L, pos, _) in flat
                  if pos != maxseq - 1 and L < h.n_layers - 2][:args.topk]
    faith_nt = faithfulness_for(lambda: nontrivial)
    print(f"\n  NON-TRIVIAL circuit (no final pos, no last 2 layers), top-{args.topk}:")
    print(f"    faithfulness = {faith_nt.mean():.1%} +/- {faith_nt.std():.1%}")

    # layer-summed attribution (which layers carry tracking)
    layer_score = mean_eff.sum(axis=1)
    top_layers = np.argsort(layer_score)[::-1][:6]
    print("\nTOP LAYERS by summed attribution:")
    for L in top_layers:
        print(f"  layer {L:2d}: score={layer_score[L]:+.4f}")

    faith_top = top_ref

    # ---- save artifacts ----
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savetxt(outdir / f"attribution_{args.task}.csv", mean_eff, delimiter=",")
    results = {
        "model": args.model, "task": args.task, "device": device,
        "n_usable": len(usable), "baseline_acc": acc,
        "ld_clean_mean": float(np.mean(ld_clean_all)),
        "ld_corrupt_mean": float(np.mean(ld_corrupt_all)),
        "faithfulness_topk_mean": float(faith_top.mean()),
        "faithfulness_topk_std": float(faith_top.std()),
        "faithfulness_randomk_mean": float(np.mean(rand_runs)),
        "faithfulness_randomk_std": float(np.std(rand_runs)),
        "faithfulness_curve": curve,
        "min_k_for_80pct": min_k_80,
        "faithfulness_nontrivial_mean": float(faith_nt.mean()),
        "faithfulness_nontrivial_std": float(faith_nt.std()),
        "top_positions_from_end": [int(maxseq - 1 - pp) for pp in top_pos],
        "topk": args.topk,
        "top_layers": [int(x) for x in top_layers],
        "layer_scores": [float(x) for x in layer_score],
    }
    (outdir / f"results_{args.task}.json").write_text(json.dumps(results, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(mean_eff, aspect="auto", cmap="RdBu_r",
                       vmin=-np.abs(mean_eff).max(), vmax=np.abs(mean_eff).max())
        ax.set_xlabel("token position (right-aligned; last col = prediction)")
        ax.set_ylabel("layer")
        ax.set_title(f"Attribution patching effect - Llama-3.2-3B - {args.task}")
        fig.colorbar(im, label="metric recovered toward clean")
        fig.tight_layout()
        fig.savefig(outdir / f"heatmap_{args.task}.png", dpi=130)
        print(f"\nsaved heatmap -> {outdir}/heatmap_{args.task}.png")
    except Exception as e:
        print("heatmap skipped:", e)

    print(f"\nsaved results -> {outdir}/results_{args.task}.json")
    print(f"TOTAL wall-clock {time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../llama-3.2-3b")
    ap.add_argument("--task", default="transfer", choices=["transfer", "intermediate"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--fewshot", type=int, default=2)
    ap.add_argument("--min-len", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=2)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--random-repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
