"""DAS-style constructive steering (learned multi-layer projection).

Why: additive mean-diff vectors gave 0% top-1 STEER. This script learns
layer-wise mappings from corrupt residuals -> cleanward deltas on TRAIN, then
applies them at inference on DEV/TEST.

Model:
  For each layer L, learn low-rank ridge map
    delta_hat = ((x - mu) @ B^T) @ M
  where B are top-k PCA directions of corrupt residuals, and M maps PCA coords
  to the target delta (clean - corrupt).

Inference intervention at entity token position:
  x <- x + alpha * delta_hat

Search objective on DEV:
  score = steer - break_penalty * break + logit_weight * delta_logit_clean
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import dataset as ds
from localize import Harness, pick_device, build_fewshot


@dataclass
class PairRec:
    cid: int
    kid: int
    clean_ids: torch.Tensor
    corrupt_ids: torch.Tensor
    dpos: int


def _prepare_pairs(h: Harness, task: str, n: int, fewshot: int,
                   min_len: int, max_len: int, seed: int) -> list[PairRec]:
    pairs = ds.generate(task, n + fewshot, seed=seed, min_len=min_len, max_len=max_len)
    ds.self_check(pairs)
    prefix = build_fewshot(pairs, fewshot, task)
    usable: list[PairRec] = []
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
        usable.append(PairRec(cid, kid, clean_ids, corrupt_ids, diffs[0]))
    return usable


def _split(records: list[PairRec], seed: int, train_frac=0.6, dev_frac=0.2):
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    n = len(idx)
    ntr = int(n * train_frac)
    ndv = int(n * dev_frac)
    tr = [records[i] for i in idx[:ntr]]
    dv = [records[i] for i in idx[ntr:ntr + ndv]]
    te = [records[i] for i in idx[ntr + ndv:]]
    return tr, dv, te


def _cache_layer_pos(h: Harness, ids: torch.Tensor, dpos: int):
    outs = h.cache_layer_outputs(ids)
    return [o[0, dpos].detach().float().cpu() for o in outs]  # list [d]


def _candidate_layer_sets(n_layers: int) -> list[list[int]]:
    c = int(round(0.66 * n_layers))
    candidates = [
        [max(0, c - 4), max(0, c - 2), c, min(n_layers - 1, c + 2)],
        list(range(max(0, c - 4), min(n_layers, c + 5))),
        list(range(max(0, c - 6), min(n_layers, c + 3))),
        list(range(max(0, c - 2), min(n_layers, c + 7))),
    ]
    out, seen = [], set()
    for s in candidates:
        s2 = tuple(sorted(set([x for x in s if 0 <= x < n_layers])))
        if s2 and s2 not in seen:
            seen.add(s2)
            out.append(list(s2))
    return out


def _fit_maps(h: Harness, train: list[PairRec], k: int, ridge: float):
    """Return per-layer map params on CPU tensors.
    params[L] = dict(mu[d], Bt[d,k], M[k,d], clip)
    """
    n_layers = len(h.layers)
    per_clean = [[] for _ in range(n_layers)]
    per_corr = [[] for _ in range(n_layers)]

    for r in train:
        c = _cache_layer_pos(h, r.clean_ids, r.dpos)
        kvec = _cache_layer_pos(h, r.corrupt_ids, r.dpos)
        for L in range(n_layers):
            per_clean[L].append(c[L])
            per_corr[L].append(kvec[L])

    params = []
    for L in range(n_layers):
        Xc = torch.stack(per_corr[L], dim=0)  # [n,d]
        Xk = torch.stack(per_clean[L], dim=0)  # [n,d]
        D = Xk - Xc
        mu = Xc.mean(dim=0)
        X0 = Xc - mu

        # PCA basis from X0: X0 = U S Vh, rows of Vh are principal dirs
        U, S, Vh = torch.linalg.svd(X0, full_matrices=False)
        kk = int(min(k, Vh.shape[0]))
        B = Vh[:kk]                           # [kk,d]
        Z = X0 @ B.T                          # [n,kk]

        # ridge solve for M: (Z^T Z + lam I) M = Z^T D
        A = Z.T @ Z + ridge * torch.eye(kk)
        RHS = Z.T @ D
        M = torch.linalg.solve(A, RHS)        # [kk,d]

        clip = D.norm(dim=1).mean().item() * 1.5 + 1e-6
        params.append({
            "mu": mu,
            "Bt": B.T.contiguous(),         # [d,kk]
            "M": M.contiguous(),            # [kk,d]
            "clip": clip,
        })
    return params


def _add_hooks_das(h: Harness, layers: list[int], params, dpos: int, alpha: float):
    handles = []

    def mk_hook(p):
        mu = p["mu"].to(h.device)
        Bt = p["Bt"].to(h.device)
        M = p["M"].to(h.device)
        clip = p["clip"]

        def hook(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            x = t[:, dpos, :].float()                 # [1,d]
            z = (x - mu) @ Bt                         # [1,kk]
            d = z @ M                                 # [1,d]
            # norm clip for stability
            n = torch.norm(d, dim=-1, keepdim=True) + 1e-9
            fac = torch.clamp(torch.tensor(clip, device=t.device) / n, max=1.0)
            d = d * fac
            t[:, dpos, :] = t[:, dpos, :] + alpha * d.to(t.dtype)
            return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t

        return hook

    for L in layers:
        handles.append(h.layers[L].register_forward_hook(mk_hook(params[L])))
    return handles


def _eval(h: Harness, recs: list[PairRec], layers: list[int], params, alpha: float):
    steer_ok = 0
    break_ok = 0
    logit_gain = []

    for r in recs:
        with torch.no_grad():
            base_logits = h.model(r.corrupt_ids).logits[0, -1]
        base_clean = base_logits[r.cid].item()

        hs = _add_hooks_das(h, layers, params, r.dpos, alpha)
        with torch.no_grad():
            logits = h.model(r.corrupt_ids).logits[0, -1]
            pred = logits.argmax().item()
        for hh in hs:
            hh.remove()
        steer_ok += int(pred == r.cid)
        logit_gain.append(float(logits[r.cid].item() - base_clean))

        hb = _add_hooks_das(h, layers, params, r.dpos, -alpha)
        with torch.no_grad():
            pred2 = h.model(r.clean_ids).logits[0, -1].argmax().item()
        for hh in hb:
            hh.remove()
        break_ok += int(pred2 != r.cid)

    n = max(1, len(recs))
    return steer_ok / n, break_ok / n, float(np.mean(logit_gain))


def run(args):
    device = pick_device(args.device)
    h = Harness(args.model, device)
    ds.restrict_to_single_token(h.tok)
    print(f"device={device} model={args.model} task={args.task} layers={len(h.layers)}")

    if args.task == "intermediate":
        min_len, max_len = 3, 4
    else:
        min_len, max_len = 1, 2

    usable = _prepare_pairs(h, args.task, args.n, args.fewshot, min_len, max_len, args.seed)
    print(f"usable={len(usable)}")
    tr, dv, te = _split(usable, args.seed)
    print(f"split train/dev/test = {len(tr)}/{len(dv)}/{len(te)}")

    layer_sets = _candidate_layer_sets(len(h.layers))
    if args.layers:
        layer_sets = [sorted(set(args.layers))]

    best = None
    leaderboard = []

    print("\n=== DEV search (DAS maps) ===")
    print(f"{'layers':30} {'k':>3} {'alpha':>6} {'steer':>8} {'break':>8} {'dlogit':>9} {'score':>8}")

    for k in args.ranks:
        params = _fit_maps(h, tr, k=k, ridge=args.ridge)
        for ls in layer_sets:
            for a in args.alphas:
                steer, brk, dlog = _eval(h, dv, ls, params, a)
                score = steer - args.break_penalty * brk + args.logit_weight * dlog
                row = {
                    "layers": ls, "rank": k, "alpha": a,
                    "steer": steer, "break": brk,
                    "delta_logit_clean": dlog, "score": score,
                }
                leaderboard.append(row)
                print(f"{str(ls)[:30]:30} {k:3d} {a:6.2f} {steer:7.1%} {brk:7.1%} {dlog:9.3f} {score:8.3f}")
                if best is None or score > best["score"]:
                    best = row

    print("\nBEST on DEV:", best)

    # refit at best rank using train, evaluate on test
    params_best = _fit_maps(h, tr, k=int(best["rank"]), ridge=args.ridge)
    steer_te, break_te, dlog_te = _eval(h, te, best["layers"], params_best, float(best["alpha"]))

    print("\n=== TEST ===")
    print(f"steer corrupt->clean: {steer_te:.1%}")
    print(f"break clean->wrong : {break_te:.1%}")
    print(f"delta clean-target logit on corrupt: {dlog_te:+.3f}")

    out = {
        "model": args.model,
        "task": args.task,
        "n": len(usable),
        "split": {"train": len(tr), "dev": len(dv), "test": len(te)},
        "break_penalty": args.break_penalty,
        "logit_weight": args.logit_weight,
        "ridge": args.ridge,
        "search": leaderboard,
        "best_dev": best,
        "test": {
            "steer": steer_te,
            "break": break_te,
            "delta_logit_clean": dlog_te,
        },
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.model).name
    p = outdir / f"intervention_das_{tag}_{args.task}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../llama-3.2-3b")
    ap.add_argument("--task", default="intermediate", choices=["intermediate", "transfer"])
    ap.add_argument("--n", type=int, default=45)
    ap.add_argument("--fewshot", type=int, default=3)
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--ranks", type=int, nargs="*", default=[4, 8, 12])
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.5, 1.0, 2.0, 3.0])
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--break-penalty", type=float, default=0.25)
    ap.add_argument("--logit-weight", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
