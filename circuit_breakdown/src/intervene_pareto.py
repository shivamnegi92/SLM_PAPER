"""Tier-1 frontier methods: two-stage Pareto + KL trust region + subspace projection.

Builds directly on `intervene_margin.py`. Three composable upgrades:

1. TWO-STAGE (--two-stage)
   Stage A: maximize margin = logit(clean) - logit(corrupt) until top-1 flips.
   Stage B: freeze the flip (hinge constraint), then minimize edit norm +
            distributional drift. Keeps STEER, shrinks collateral.

2. KL TRUST REGION (--lam-kl > 0)
   Inline penalty KL(p_intervened || p_base) computed over the NON-target
   vocabulary (clean/corrupt tokens masked out). Lets the edit swing the two
   competing tokens freely while protecting the rest of the distribution.
   This is the principled replacement for a raw norm clamp.

3. SUBSPACE PROJECTION (--subspace-rank > 0)
   Edits are parameterized as v_L = z_L @ B_L, where B_L is a rank-r basis of
   the tracking subspace (SVD of clean-corrupt activation differences on the
   train split). Edits are structurally unable to leave the tracking subspace,
   so they cannot wander into capability-destroying directions.

   NOTE ON LEARNING RATE: Adam takes steps of size ~lr per coordinate, so the
   norm of one step grows like lr*sqrt(#params). Optimizing r coefficients
   instead of `hidden` raw dims therefore shrinks the effective step by
   sqrt(hidden/r) (~20x at r=8, hidden=3072). Without correction, subspace arms
   are not step-matched to full-space arms and appear to fail for purely
   numerical reasons. We rescale lr by sqrt(hidden/r) by default; disable with
   --no-lr-autoscale.

   NOTE ON ACHIEVABLE RANK: the basis comes from an SVD over n_train difference
   vectors, so rank is capped at n_train+1 regardless of the requested value.
   The realized rank is recorded as `subspace_rank_actual`.

All methods are inference-time only. No weight updates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import dataset as ds
from localize import Harness, pick_device
from intervene_margin import (
    CONTROL_PROMPTS,
    PairRec,
    bootstrap_rate,
    control_degradation,
    get_control_baseline,
    make_layers,
    prepare_pairs,
    split_records,
)


# ---------------------------------------------------------------- edit params


class EditParams:
    """Learnable per-layer edit vectors, optionally confined to a subspace."""

    def __init__(self, layers, hidden, device, bases=None):
        self.layers = layers
        self.bases = bases  # list of [rank, hidden] or None
        if bases is None:
            self._p = [
                torch.zeros((1, hidden), device=device, requires_grad=True)
                for _ in layers
            ]
        else:
            self._p = [
                torch.zeros((1, b.shape[0]), device=device, requires_grad=True)
                for b in bases
            ]

    def vectors(self):
        """Differentiable list of [1, hidden] edit vectors."""
        if self.bases is None:
            return self._p
        return [z @ b for z, b in zip(self._p, self.bases)]

    def parameters(self):
        return self._p

    @torch.no_grad()
    def scale(self, factor: float):
        for p in self._p:
            p.mul_(factor)

    @torch.no_grad()
    def clamp_norm(self, budget: float):
        for v, p in zip(self.vectors(), self._p):
            n = v.norm().item() + 1e-9
            if n > budget:
                p.mul_(budget / n)

    @torch.no_grad()
    def detached(self):
        return [v.detach() for v in self.vectors()]

    @torch.no_grad()
    def total_norm(self):
        return float(sum(v.norm().item() for v in self.vectors()))


def fit_subspace_bases(h: Harness, train: list[PairRec], layers, rank: int, complement: bool = False, comp_dim: int = 0):
    """Rank-r basis per layer from clean-corrupt entity-position differences.

    If complement=True, returns instead an orthonormal basis for a random
    `comp_dim`-dimensional subspace of the ORTHOGONAL COMPLEMENT of the
    difference subspace. This is the control condition: same edit budget, same
    dimensionality, but explicitly outside the causally-identified directions.
    """
    diffs = {L: [] for L in layers}
    for r in train:
        clean_out = h.cache_layer_outputs(r.clean_ids)
        corr_out = h.cache_layer_outputs(r.corrupt_ids)
        for L in layers:
            d = clean_out[L][0, r.dpos] - corr_out[L][0, r.dpos]
            diffs[L].append(d.detach().float().cpu())

    bases = []
    for L in layers:
        X = torch.stack(diffs[L], 0)  # [n, hidden]
        # center so the basis captures variation, not just the mean direction
        mu = X.mean(0, keepdim=True)
        Xc = torch.cat([mu, X - mu], 0)
        _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
        rk = int(min(rank, Vh.shape[0]))
        B = Vh[:rk]

        if complement:
            hidden = X.shape[1]
            k = comp_dim if comp_dim > 0 else rk
            g = torch.Generator().manual_seed(1234 + L)
            R = torch.randn(k, hidden, generator=g)
            # remove any component lying in the difference subspace
            R = R - (R @ B.T) @ B
            # orthonormalize the remainder
            Q, _ = torch.linalg.qr(R.T)
            B = Q.T[:k]

        B = B / (B.norm(dim=1, keepdim=True) + 1e-9)
        bases.append(B.to(h.device))
    return bases


# ------------------------------------------------------------------- hooking


def hooks_for(h: Harness, layers, vecs, pos):
    handles = []

    def mk(v):
        def hook(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            p = pos if pos >= 0 else t.shape[1] + pos
            t[:, p, :] = t[:, p, :] + v.to(t.dtype)
            return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t

        return hook

    for L, v in zip(layers, vecs):
        handles.append(h.layers[L].register_forward_hook(mk(v)))
    return handles


def forward_logits(h: Harness, ids, layers, vecs, pos):
    hs = hooks_for(h, layers, vecs, pos)
    try:
        return h.model(ids).logits[0, -1]
    finally:
        for hh in hs:
            hh.remove()


# ------------------------------------------------------------------- KL term


def kl_nontarget(logits, base_logits, cid, kid):
    """KL(p_intervened || p_base) restricted to vocab minus {clean, corrupt}."""
    neg = torch.finfo(logits.dtype).min
    li = logits.clone()
    li[cid] = neg
    li[kid] = neg
    lb = base_logits.clone()
    lb[cid] = neg
    lb[kid] = neg
    logp_i = F.log_softmax(li, dim=-1)
    logp_b = F.log_softmax(lb, dim=-1)
    return (logp_i.exp() * (logp_i - logp_b)).sum()


# -------------------------------------------------------------- optimization


def optimize_sample(h, r, layers, ctrl_ids, ctrl_base, bases, cfg):
    hidden = h.model.config.hidden_size
    ep = EditParams(layers, hidden, h.device, bases=bases)
    opt = torch.optim.Adam(ep.parameters(), lr=cfg["lr"])

    with torch.no_grad():
        base_logits = h.model(r.corrupt_ids).logits[0, -1].detach()

    lam_kl = cfg["lam_kl"]
    guard_every = max(1, int(cfg["guard_every"]))

    # ---- Stage A: push the decision over the line -------------------------
    margin_a = None
    for step in range(cfg["stage_a_steps"]):
        logits = forward_logits(h, r.corrupt_ids, layers, ep.vectors(), r.dpos)
        margin = logits[r.cid] - logits[r.kid]
        loss = -margin
        if lam_kl > 0:
            loss = loss + lam_kl * kl_nontarget(logits, base_logits, r.cid, r.kid)
        loss = loss + cfg["l2"] * sum((v.float() ** 2).mean() for v in ep.vectors())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ep.clamp_norm(cfg["norm_budget"])
        margin_a = float(margin.detach().item())

        if (step + 1) % guard_every == 0:
            drop = control_degradation(h, layers, ep.detached(), ctrl_ids, ctrl_base)
            if drop > cfg["max_control_drop"]:
                ep.scale(0.7)

    # ---- Stage B: keep the flip, shrink the damage ------------------------
    if cfg["two_stage"] and cfg["stage_b_steps"] > 0:
        floor = max(cfg["margin_floor"], cfg["keep_frac"] * max(margin_a or 0.0, 0.0))
        opt_b = torch.optim.Adam(ep.parameters(), lr=cfg["lr_b"])
        for step in range(cfg["stage_b_steps"]):
            logits = forward_logits(h, r.corrupt_ids, layers, ep.vectors(), r.dpos)
            margin = logits[r.cid] - logits[r.kid]

            vecs = ep.vectors()
            norm_pen = sum((v.float() ** 2).sum() for v in vecs)
            hinge = F.relu(floor - margin)
            loss = cfg["w_norm"] * norm_pen + cfg["w_hinge"] * hinge
            if lam_kl > 0:
                loss = loss + cfg["lam_kl_b"] * kl_nontarget(
                    logits, base_logits, r.cid, r.kid
                )

            opt_b.zero_grad(set_to_none=True)
            loss.backward()
            opt_b.step()
            ep.clamp_norm(cfg["norm_budget"])

            if (step + 1) % guard_every == 0:
                drop = control_degradation(
                    h, layers, ep.detached(), ctrl_ids, ctrl_base
                )
                if drop > cfg["max_control_drop"]:
                    ep.scale(0.7)

    return ep


# -------------------------------------------------------------- evaluation


def eval_set(h, recs, layers, bases, cfg, ctrl_ids, ctrl_base):
    steer, brk, dlog, drops, norms = [], [], [], [], []

    for r in recs:
        ep = optimize_sample(h, r, layers, ctrl_ids, ctrl_base, bases, cfg)
        vecs = ep.detached()

        with torch.no_grad():
            base_clean = h.model(r.corrupt_ids).logits[0, -1][r.cid].item()
            logits = forward_logits(h, r.corrupt_ids, layers, vecs, r.dpos)
            pred = int(logits.argmax().item())
            steer.append(int(pred == r.cid))
            dlog.append(float(logits[r.cid].item() - base_clean))

            neg = [-v for v in vecs]
            p2 = int(
                forward_logits(h, r.clean_ids, layers, neg, r.dpos).argmax().item()
            )
            brk.append(int(p2 != r.cid))

        drops.append(control_degradation(h, layers, vecs, ctrl_ids, ctrl_base))
        norms.append(ep.total_norm())

    return {
        "steer": float(np.mean(steer)) if steer else 0.0,
        "break": float(np.mean(brk)) if brk else 0.0,
        "delta_logit_clean": float(np.mean(dlog)) if dlog else 0.0,
        "control_drop": float(np.mean(drops)) if drops else 0.0,
        "edit_norm_mean": float(np.mean(norms)) if norms else 0.0,
        "steer_flags": steer,
        "break_flags": brk,
    }


def build_cfg(args):
    return {
        "lr": args.lr,
        "lr_b": args.lr_b,
        "stage_a_steps": args.stage_a_steps,
        "stage_b_steps": args.stage_b_steps,
        "two_stage": bool(args.two_stage),
        "l2": args.l2,
        "lam_kl": args.lam_kl,
        "lam_kl_b": args.lam_kl_b if args.lam_kl_b is not None else args.lam_kl,
        "norm_budget": args.norm_budget,
        "max_control_drop": args.max_control_drop,
        "guard_every": args.guard_every,
        "margin_floor": args.margin_floor,
        "keep_frac": args.keep_frac,
        "w_norm": args.w_norm,
        "w_hinge": args.w_hinge,
    }


def run(args):
    device = pick_device(args.device)
    h = Harness(args.model, device)
    ds.restrict_to_single_token(h.tok)

    recs = prepare_pairs(h, args.task, args.n, args.fewshot, args.seed)
    tr, dv, te = split_records(recs, args.seed)
    layers = make_layers(len(h.layers)) if not args.layers else sorted(set(args.layers))
    ctrl_ids, ctrl_base = get_control_baseline(h)

    method_bits = []
    if args.two_stage:
        method_bits.append("two_stage")
    if args.lam_kl > 0:
        method_bits.append(f"kl{args.lam_kl:g}")
    if args.subspace_rank > 0:
        method_bits.append(f"{'comp' if args.subspace_complement else 'sub'}{args.subspace_rank}")
    method = "+".join(method_bits) if method_bits else "margin_baseline"

    print(f"device={device} model={args.model} task={args.task}")
    print(f"usable={len(recs)} split train/dev/test={len(tr)}/{len(dv)}/{len(te)}")
    print(f"layers={layers} method={method}")

    bases = None
    rank_actual = 0
    cfg = build_cfg(args)
    if args.subspace_rank > 0:
        bases = fit_subspace_bases(
            h, tr, layers, args.subspace_rank,
            complement=args.subspace_complement,
            comp_dim=args.subspace_rank,
        )
        rank_actual = int(bases[0].shape[0])
        cap = len(tr) + 1
        note = "" if rank_actual >= args.subspace_rank else f"  (CAPPED by n_train+1={cap})"
        kind = "ORTHOGONAL-COMPLEMENT control" if args.subspace_complement else "tracking subspace"
        print(f"{kind}: rank requested={args.subspace_rank} actual={rank_actual}{note}")
        if args.lr_autoscale:
            hidden = h.model.config.hidden_size
            scale = (hidden / max(1, rank_actual)) ** 0.5
            cfg["lr"] *= scale
            cfg["lr_b"] *= scale
            print(f"lr autoscaled x{scale:.1f} -> stage_a lr={cfg['lr']:.4f} (step-matched to full space)")
    mt = eval_set(h, te, layers, bases, cfg, ctrl_ids, ctrl_base)

    s_mean, s_ci = bootstrap_rate(mt["steer_flags"], iters=args.bootstrap_iters, seed=args.seed)
    b_mean, b_ci = bootstrap_rate(mt["break_flags"], iters=args.bootstrap_iters, seed=args.seed + 1)

    print("\n=== PARETO RESULT (TEST) ===")
    print(
        f"steer={s_mean:.1%} [{s_ci[0]:.1%},{s_ci[1]:.1%}]  "
        f"break={b_mean:.1%} [{b_ci[0]:.1%},{b_ci[1]:.1%}]  "
        f"dlogit={mt['delta_logit_clean']:+.3f}  "
        f"ctrl={mt['control_drop']:.1%}  "
        f"|edit|={mt['edit_norm_mean']:.2f}"
    )

    out = {
        "model": args.model,
        "task": args.task,
        "method": method,
        "n": len(recs),
        "split": {"train": len(tr), "dev": len(dv), "test": len(te)},
        "layers": layers,
        "cfg": cfg,
        "subspace_rank": args.subspace_rank,
        "subspace_rank_actual": rank_actual,
        "subspace_complement": bool(args.subspace_complement),
        "lr_autoscale": bool(args.lr_autoscale),
        "seed": args.seed,
        "test": {**mt, "steer": s_mean, "steer_ci": s_ci, "break": b_mean, "break_ci": b_ci},
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.model).name
    name = args.out_name or f"pareto_{tag}_{args.task}_{method}_s{args.seed}"
    p = outdir / f"{name}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"saved -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../llama-3.2-3b")
    ap.add_argument("--task", default="intermediate", choices=["intermediate", "transfer"])
    ap.add_argument("--n", type=int, default=42)
    ap.add_argument("--fewshot", type=int, default=3)
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    # method switches
    ap.add_argument("--two-stage", action="store_true")
    ap.add_argument("--lam-kl", type=float, default=0.0)
    ap.add_argument("--lam-kl-b", type=float, default=None)
    ap.add_argument("--subspace-rank", type=int, default=0)
    ap.add_argument(
        "--no-lr-autoscale",
        dest="lr_autoscale",
        action="store_false",
        help="disable sqrt(hidden/rank) lr correction for subspace runs",
    )
    ap.set_defaults(lr_autoscale=True)
    ap.add_argument(
        "--subspace-complement",
        action="store_true",
        help="control: edit inside the orthogonal complement of the difference subspace",
    )
    # stage A
    ap.add_argument("--stage-a-steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-3)
    # stage B
    ap.add_argument("--stage-b-steps", type=int, default=6)
    ap.add_argument("--lr-b", type=float, default=0.05)
    ap.add_argument("--w-norm", type=float, default=1e-3)
    ap.add_argument("--w-hinge", type=float, default=10.0)
    ap.add_argument("--margin-floor", type=float, default=0.5)
    ap.add_argument("--keep-frac", type=float, default=0.3)
    # guards
    ap.add_argument("--norm-budget", type=float, default=4.0)
    ap.add_argument("--max-control-drop", type=float, default=0.20)
    ap.add_argument("--guard-every", type=int, default=4)
    # misc
    ap.add_argument("--bootstrap-iters", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--out-name", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
