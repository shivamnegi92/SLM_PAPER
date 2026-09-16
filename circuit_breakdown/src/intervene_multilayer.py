"""Multi-layer constructive steering for tracking circuits.

Goal: fix W4 (single-layer STEER=0%) by applying clean-ward directions across
multiple layers at the entity token position.

Method:
1) Build usable clean/corrupt pairs with aligned entity token position.
2) Compute per-layer clean-ward direction d_L = E[clean_resid - corrupt_resid]
   at the entity position on a TRAIN split.
3) Tune layer-set + alpha on DEV for corrupt->clean STEER, while monitoring
   clean->wrong BREAK side-effect.
4) Report final metrics on TEST split.

We keep it inference-time only (no weight updates).
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


def _cache_two_pos_vectors(h: Harness, ids: torch.Tensor, dpos: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return per-layer vectors at (entity position, final position)."""
    outs = h.cache_layer_outputs(ids)  # list [1,seq,hid]
    ent = [o[0, dpos].detach().clone() for o in outs]
    fin = [o[0, -1].detach().clone() for o in outs]
    return ent, fin


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


def _compute_directions(h: Harness, train: list[PairRec]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Compute per-layer clean-ward directions at (entity pos) and (final pos)."""
    n_layers = len(h.layers)
    sums_ent = [None] * n_layers
    sums_fin = [None] * n_layers
    for r in train:
        cent, cfin = _cache_two_pos_vectors(h, r.clean_ids, r.dpos)
        kent, kfin = _cache_two_pos_vectors(h, r.corrupt_ids, r.dpos)
        for L in range(n_layers):
            de = cent[L] - kent[L]
            df = cfin[L] - kfin[L]
            sums_ent[L] = de if sums_ent[L] is None else (sums_ent[L] + de)
            sums_fin[L] = df if sums_fin[L] is None else (sums_fin[L] + df)

    def normed(sums):
        out = []
        for s in sums:
            d = s / max(1, len(train))
            n = d.norm().item() + 1e-9
            out.append((d / n).detach())
        return out

    return normed(sums_ent), normed(sums_fin)


def _add_multi_hooks(
    h: Harness,
    layers: list[int],
    vecs_ent: list[torch.Tensor],
    vecs_fin: list[torch.Tensor],
    dpos: int,
    alpha_ent: float,
    alpha_fin: float,
):
    handles = []

    def mk_hook(vec_ent: torch.Tensor, vec_fin: torch.Tensor):
        def hook(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            if alpha_ent != 0:
                t[:, dpos, :] = t[:, dpos, :] + alpha_ent * vec_ent.to(t.dtype)
            if alpha_fin != 0:
                t[:, -1, :] = t[:, -1, :] + alpha_fin * vec_fin.to(t.dtype)
            return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
        return hook

    for L in layers:
        handles.append(h.layers[L].register_forward_hook(mk_hook(vecs_ent[L], vecs_fin[L])))
    return handles


def _eval(
    h: Harness,
    recs: list[PairRec],
    layers: list[int],
    vecs_ent: list[torch.Tensor],
    vecs_fin: list[torch.Tensor],
    alpha_ent: float,
    alpha_fin: float,
):
    steer_ok = 0  # corrupt -> clean target
    break_ok = 0  # clean -> not clean target
    logit_gain = []
    for r in recs:
        with torch.no_grad():
            base_logits = h.model(r.corrupt_ids).logits[0, -1]
        base_clean = base_logits[r.cid].item()

        hs = _add_multi_hooks(h, layers, vecs_ent, vecs_fin, r.dpos, alpha_ent, alpha_fin)
        with torch.no_grad():
            logits = h.model(r.corrupt_ids).logits[0, -1]
            pred = logits.argmax().item()
        for hh in hs:
            hh.remove()
        steer_ok += int(pred == r.cid)
        logit_gain.append(float(logits[r.cid].item() - base_clean))

        hb = _add_multi_hooks(h, layers, vecs_ent, vecs_fin, r.dpos, -alpha_ent, -alpha_fin)
        with torch.no_grad():
            pred2 = h.model(r.clean_ids).logits[0, -1].argmax().item()
        for hh in hb:
            hh.remove()
        break_ok += int(pred2 != r.cid)

    n = max(1, len(recs))
    return steer_ok / n, break_ok / n, float(np.mean(logit_gain))


def _candidate_layer_sets(n_layers: int) -> list[list[int]]:
    # centers around empirical handoff zone (~0.62-0.70 depth)
    c = int(round(0.66 * n_layers))
    candidates = [
        [max(0, c - 4), max(0, c - 2), c, min(n_layers - 1, c + 2)],
        [max(0, c - 6), max(0, c - 4), max(0, c - 2), c],
        [max(0, c - 2), c, min(n_layers - 1, c + 2), min(n_layers - 1, c + 4)],
        list(range(max(0, c - 4), min(n_layers, c + 5))),
        list(range(max(0, c - 6), min(n_layers, c + 3))),
        list(range(max(0, c - 2), min(n_layers, c + 7))),
    ]
    dedup = []
    seen = set()
    for s in candidates:
        s2 = sorted(set([x for x in s if 0 <= x < n_layers]))
        key = tuple(s2)
        if key and key not in seen:
            seen.add(key)
            dedup.append(s2)
    return dedup


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

    vecs_ent, vecs_fin = _compute_directions(h, tr)

    layer_sets = _candidate_layer_sets(len(h.layers))
    if args.layers:
        layer_sets = [sorted(set(args.layers))]

    alphas = args.alphas
    best = None
    leaderboard = []

    print("\n=== DEV search (maximize STEER, penalize BREAK side-effect) ===")
    print(f"{'layers':30} {'a_ent':>6} {'a_fin':>6} {'steer':>8} {'break':>8} {'dlogit':>9} {'score':>8}")
    for ls in layer_sets:
        for ae in alphas:
            for af in args.alpha_finals:
                steer, brk, dlog = _eval(h, dv, ls, vecs_ent, vecs_fin, ae, af)
                # prioritize steer; soft penalty for break-on-clean;
                # small bonus for lifting clean-target logit on corrupt prompts
                score = steer - args.break_penalty * brk + args.logit_weight * dlog
                leaderboard.append({
                    "layers": ls, "alpha_ent": ae, "alpha_fin": af,
                    "steer": steer, "break": brk, "delta_logit_clean": dlog,
                    "score": score,
                })
                print(f"{str(ls)[:30]:30} {ae:6.2f} {af:6.2f} {steer:7.1%} {brk:7.1%} {dlog:9.3f} {score:8.3f}")
                if best is None or score > best["score"]:
                    best = {
                        "layers": ls, "alpha_ent": ae, "alpha_fin": af,
                        "steer": steer, "break": brk, "delta_logit_clean": dlog,
                        "score": score,
                    }

    print("\nBEST on DEV:", best)

    steer_te, break_te, dlog_te = _eval(
        h, te, best["layers"], vecs_ent, vecs_fin,
        best["alpha_ent"], best["alpha_fin"],
    )
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
        "search": leaderboard,
        "best_dev": best,
        "test": {"steer": steer_te, "break": break_te, "delta_logit_clean": dlog_te},
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.model).name
    p = outdir / f"intervention_multilayer_{tag}_{args.task}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../llama-3.2-3b")
    ap.add_argument("--task", default="intermediate", choices=["intermediate", "transfer"])
    ap.add_argument("--n", type=int, default=45)
    ap.add_argument("--fewshot", type=int, default=3)
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.0, 0.5, 1.0, 2.0, 3.0, 4.0])
    ap.add_argument("--alpha-finals", type=float, nargs="*", default=[0.0, 0.5, 1.0, 2.0, 3.0, 4.0])
    ap.add_argument("--break-penalty", type=float, default=0.25)
    ap.add_argument("--logit-weight", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
