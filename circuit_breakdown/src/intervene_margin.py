"""Block B: Path-constrained / objective-aware intervention prototype.

Inference-time only (no weight updates).

For each corrupt prompt, optimize per-layer intervention vectors v_L to maximize:
  margin = logit(clean_target) - logit(corrupt_target)
with regularization + dynamic clamping:
- norm budget per layer
- capability guard on control prompts (argmax agreement with baseline)
  if degradation exceeds threshold, vectors are shrunk inline.

Outputs train/dev/test metrics and best config.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import dataset as ds
from localize import Harness, pick_device, build_fewshot


CONTROL_PROMPTS = [
    "The capital of France is",
    "Two plus two equals",
    "A short bedtime story begins",
]


@dataclass
class PairRec:
    cid: int
    kid: int
    clean_ids: torch.Tensor
    corrupt_ids: torch.Tensor
    dpos: int
    sample_id: str = ""
    input_hash: str = ""


def bootstrap_rate(flags, iters: int = 1000, seed: int = 0):
    arr = np.array(flags, dtype=np.float32)
    if len(arr) == 0:
        return 0.0, [0.0, 0.0]
    rng = np.random.default_rng(seed)
    n = len(arr)
    means = np.empty(iters, dtype=np.float32)
    for i in range(iters):
        s = rng.choice(arr, n, replace=True)
        means[i] = s.mean()
    return float(arr.mean()), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def encode_pair(h, pair, prefix=""):
    cid, kid = h.first_id(pair.clean_target), h.first_id(pair.corrupt_target)
    clean_ids = h.encode(prefix + pair.clean_prompt)
    corrupt_ids = h.encode(prefix + pair.corrupt_prompt)
    if cid == kid or clean_ids.shape != corrupt_ids.shape:
        raise ValueError("Pair targets/lengths are not aligned")
    differences = (clean_ids[0] != corrupt_ids[0]).nonzero().flatten().tolist()
    if len(differences) != 1:
        raise ValueError("Expected exactly one aligned token edit")
    semantic = [pair.task, pair.clean_prompt, pair.corrupt_prompt,
                pair.clean_target, pair.corrupt_target]
    sample_id = hashlib.sha256(json.dumps(semantic, ensure_ascii=True).encode()).hexdigest()
    tokenized = [clean_ids[0].tolist(), corrupt_ids[0].tolist()]
    input_hash = hashlib.sha256(json.dumps(tokenized).encode()).hexdigest()
    return PairRec(cid, kid, clean_ids, corrupt_ids, differences[0], sample_id, input_hash)


def prepare_pairs(h: Harness, task: str, n: int, fewshot: int, seed: int) -> list[PairRec]:
    if task == "intermediate":
        min_len, max_len = 3, 4
    else:
        min_len, max_len = 1, 2
    pairs = ds.generate(task, n + fewshot, seed=seed, min_len=min_len, max_len=max_len)
    ds.self_check(pairs)
    prefix = build_fewshot(pairs, fewshot, task)

    out = []
    for p in pairs[:n]:
        try:
            out.append(encode_pair(h, p, prefix))
        except ValueError:
            continue
    return out


def split_records(records: list[PairRec], seed: int, train_frac=0.6, dev_frac=0.2):
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    n = len(idx)
    ntr = int(n * train_frac)
    ndv = int(n * dev_frac)
    tr = [records[i] for i in idx[:ntr]]
    dv = [records[i] for i in idx[ntr:ntr + ndv]]
    te = [records[i] for i in idx[ntr + ndv:]]
    return tr, dv, te


def make_layers(n_layers: int):
    # downstream-ish path-constrained band
    return [x for x in [18, 20, 22, 24] if x < n_layers]


def get_control_baseline(h: Harness):
    ids_list = [h.encode(p) for p in CONTROL_PROMPTS]
    base = []
    with torch.no_grad():
        for ids in ids_list:
            base.append(int(h.model(ids).logits[0, -1].argmax().item()))
    return ids_list, base


def apply_vectors_hooks(h: Harness, layers, vecs, dpos_or_neg1: int):
    hs = []

    def mk(v):
        def hook(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            pos = dpos_or_neg1 if dpos_or_neg1 >= 0 else t.shape[1] + dpos_or_neg1
            t[:, pos, :] = t[:, pos, :] + v.to(t.dtype)
            return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
        return hook

    for j, L in enumerate(layers):
        hs.append(h.layers[L].register_forward_hook(mk(vecs[j])))
    return hs


def control_degradation(h: Harness, layers, vecs, ctrl_ids, ctrl_base):
    changed = 0
    with torch.no_grad():
        for ids, b in zip(ctrl_ids, ctrl_base):
            hs = apply_vectors_hooks(h, layers, vecs, -1)  # last token for generic controls
            try:
                pred = int(h.model(ids).logits[0, -1].argmax().item())
            finally:
                for hh in hs:
                    hh.remove()
            changed += int(pred != b)
    return changed / max(1, len(ctrl_ids))


def optimize_sample(
    h: Harness,
    r: PairRec,
    layers,
    ctrl_ids,
    ctrl_base,
    steps=20,
    lr=0.08,
    l2=1e-3,
    norm_budget=8.0,
    max_control_drop=0.34,
):
    d = h.model.config.hidden_size
    vecs = [torch.zeros((1, d), device=h.device, requires_grad=True) for _ in layers]
    opt = torch.optim.Adam(vecs, lr=lr)

    for _ in range(steps):
        hs = apply_vectors_hooks(h, layers, vecs, r.dpos)
        logits = h.model(r.corrupt_ids).logits[0, -1]
        for hh in hs:
            hh.remove()

        margin = logits[r.cid] - logits[r.kid]
        reg = sum((v.float() ** 2).mean() for v in vecs)
        loss = -margin + l2 * reg

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # dynamic clamping: projected norm budget
        with torch.no_grad():
            for i in range(len(vecs)):
                n = vecs[i].norm().item() + 1e-9
                if n > norm_budget:
                    vecs[i].mul_(norm_budget / n)

        # capability guard: shrink if control degradation too high
        with torch.no_grad():
            drop = control_degradation(h, layers, [v.detach() for v in vecs], ctrl_ids, ctrl_base)
            if drop > max_control_drop:
                for i in range(len(vecs)):
                    vecs[i].mul_(0.7)

    return [v.detach() for v in vecs]


def eval_set(h: Harness, recs: list[PairRec], layers, cfg, ctrl_ids, ctrl_base):
    steer_flags, break_flags, dlog = [], [], []
    ctrl_drops = []

    for r in recs:
        vecs = optimize_sample(
            h, r, layers, ctrl_ids, ctrl_base,
            steps=cfg["steps"],
            lr=cfg["lr"],
            l2=cfg["l2"],
            norm_budget=cfg["norm_budget"],
            max_control_drop=cfg["max_control_drop"],
        )

        with torch.no_grad():
            base = h.model(r.corrupt_ids).logits[0, -1]
        base_clean = base[r.cid].item()

        hs = apply_vectors_hooks(h, layers, vecs, r.dpos)
        with torch.no_grad():
            logits = h.model(r.corrupt_ids).logits[0, -1]
            pred = int(logits.argmax().item())
        for hh in hs:
            hh.remove()

        steer_flags.append(int(pred == r.cid))
        dlog.append(float(logits[r.cid].item() - base_clean))

        hb = apply_vectors_hooks(h, layers, [-v for v in vecs], r.dpos)
        with torch.no_grad():
            p2 = int(h.model(r.clean_ids).logits[0, -1].argmax().item())
        for hh in hb:
            hh.remove()
        break_flags.append(int(p2 != r.cid))

        ctrl_drops.append(control_degradation(h, layers, vecs, ctrl_ids, ctrl_base))

    return {
        "steer": float(np.mean(steer_flags)) if steer_flags else 0.0,
        "break": float(np.mean(break_flags)) if break_flags else 0.0,
        "delta_logit_clean": float(np.mean(dlog)) if dlog else 0.0,
        "control_drop": float(np.mean(ctrl_drops)) if ctrl_drops else 0.0,
        "control_drop_std": float(np.std(ctrl_drops)) if ctrl_drops else 0.0,
        "steer_flags": steer_flags,
        "break_flags": break_flags,
        "control_drops": ctrl_drops,
    }


def run(args):
    device = pick_device(args.device)
    h = Harness(args.model, device)
    ds.restrict_to_single_token(h.tok)

    recs = prepare_pairs(h, args.task, args.n, args.fewshot, args.seed)
    tr, dv, te = split_records(recs, args.seed)
    layers = make_layers(len(h.layers)) if not args.layers else sorted(set(args.layers))

    ctrl_ids, ctrl_base = get_control_baseline(h)

    print(f"device={device} model={args.model} task={args.task}")
    print(f"usable={len(recs)} split train/dev/test={len(tr)}/{len(dv)}/{len(te)}")
    print(f"layers={layers}")

    # dev search
    candidates = []
    for steps in args.steps:
        for lr in args.lrs:
            for norm_budget in args.norm_budgets:
                cfg = {
                    "steps": int(steps),
                    "lr": float(lr),
                    "l2": float(args.l2),
                    "norm_budget": float(norm_budget),
                    "max_control_drop": float(args.max_control_drop),
                }
                m = eval_set(h, dv, layers, cfg, ctrl_ids, ctrl_base)
                score = m["steer"] - args.break_penalty * m["break"] + args.logit_weight * m["delta_logit_clean"] - args.control_penalty * m["control_drop"]
                candidates.append({"cfg": cfg, "dev": m, "score": float(score)})

    best = sorted(candidates, key=lambda x: x["score"], reverse=True)[0]

    # test with best cfg
    mt = eval_set(h, te, layers, best["cfg"], ctrl_ids, ctrl_base)
    steer_mean, steer_ci = bootstrap_rate(mt["steer_flags"], iters=args.bootstrap_iters, seed=args.seed)
    break_mean, break_ci = bootstrap_rate(mt["break_flags"], iters=args.bootstrap_iters, seed=args.seed + 1)

    print("\n=== BLOCK B RESULT (TEST) ===")
    print(
        f"steer={steer_mean:.1%} [{steer_ci[0]:.1%},{steer_ci[1]:.1%}]  "
        f"break={break_mean:.1%} [{break_ci[0]:.1%},{break_ci[1]:.1%}]  "
        f"dlogit={mt['delta_logit_clean']:+.3f}  control_drop={mt['control_drop']:.1%}"
    )

    out = {
        "model": args.model,
        "task": args.task,
        "n": len(recs),
        "split": {"train": len(tr), "dev": len(dv), "test": len(te)},
        "layers": layers,
        "best": best,
        "test": {
            **mt,
            "steer": steer_mean,
            "steer_ci": steer_ci,
            "break": break_mean,
            "break_ci": break_ci,
        },
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.model).name
    p = outdir / f"intervene_margin_{tag}_{args.task}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"saved -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../llama-3.2-3b")
    ap.add_argument("--task", default="intermediate", choices=["intermediate", "transfer"])
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--fewshot", type=int, default=3)
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--steps", type=int, nargs="*", default=[12, 20])
    ap.add_argument("--lrs", type=float, nargs="*", default=[0.05, 0.08])
    ap.add_argument("--norm-budgets", type=float, nargs="*", default=[4.0, 8.0])
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--max-control-drop", type=float, default=0.34)
    ap.add_argument("--break-penalty", type=float, default=0.25)
    ap.add_argument("--control-penalty", type=float, default=0.20)
    ap.add_argument("--logit-weight", type=float, default=0.02)
    ap.add_argument("--bootstrap-iters", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
