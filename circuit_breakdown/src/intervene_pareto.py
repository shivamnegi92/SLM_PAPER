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

NOTE ON CROSS-ARCHITECTURE BUDGETS: residual-stream norms differ enormously
between model families (measured at the entity position, mid-depth window:
Llama-3.2-3B ~23.6, Nemotron-Mini-4B ~126.4, Phi-3.5-mini ~177.4 -- a 7.5x
spread). An absolute norm budget is therefore NOT comparable across models: a
budget of 4.0 is a 17% perturbation on Llama but only 2.3% on Phi. Use
--rel-budget to specify the budget as a fraction of the measured mean residual
norm at the target position; this is the scale-invariant setting and is what
should be used for any cross-architecture claim.

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
from experiment_metrics import prediction_outcome, summarize_predictions, rate_ci
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
        budgets = [budget] * len(self._p) if np.isscalar(budget) else list(budget)
        if len(budgets) != len(self._p) or any(value <= 0 or not np.isfinite(value) for value in budgets):
            raise ValueError("Positive finite per-layer budgets are required")
        for v, p, limit in zip(self.vectors(), self._p, budgets):
            n = v.norm().item() + 1e-9
            if n > limit:
                p.mul_(limit / n)

    @torch.no_grad()
    def detached(self):
        return [v.detach() for v in self.vectors()]

    @torch.no_grad()
    def total_norm(self):
        return float(sum(v.norm().item() for v in self.vectors()))


def basis_from_differences(matrix, rank, complement=False, comp_dim=0, seed=0):
    if matrix.ndim != 2 or not torch.isfinite(matrix).all().item() or rank < 1:
        raise ValueError("Finite difference matrix and positive rank required")
    matrix = matrix.detach().cpu().double()
    mean = matrix.mean(0, keepdim=True)
    centered = torch.cat([mean, matrix - mean], 0)
    _, singular, right = torch.linalg.svd(centered, full_matrices=False)
    tolerance = max(centered.shape) * torch.finfo(torch.float32).eps * singular[0].item()
    numerical_rank = int((singular > tolerance).sum().item())
    actual = min(rank, numerical_rank)
    if actual < 1:
        raise ValueError("Difference matrix has no nonzero tracking directions")
    tracking = right[:actual]
    basis = tracking
    if complement:
        dimension = comp_dim or actual
        if dimension > matrix.shape[1] - actual:
            raise ValueError("Requested control exceeds orthogonal complement dimension")
        generator = torch.Generator().manual_seed(seed)
        random = torch.randn(dimension, matrix.shape[1], generator=generator, dtype=torch.float64)
        random -= (random @ tracking.T) @ tracking
        orthonormal, _ = torch.linalg.qr(random.T)
        basis = orthonormal.T[:dimension]
    information = {
        "requested_rank": rank, "tracking_rank": actual, "basis_rank": len(basis),
        "numerical_rank": numerical_rank, "tolerance": tolerance,
        "explained_energy": float((singular[:actual] ** 2).sum() / (singular ** 2).sum()),
        "orthonormal_error": float((basis @ basis.T - torch.eye(len(basis))).abs().max()),
        "tracking_overlap": float((basis @ tracking.T).abs().max()),
        "control_seed": seed if complement else None,
    }
    return basis.float(), information


def fit_subspace_bases(h: Harness, train: list[PairRec], layers, rank: int, complement: bool = False,
                       comp_dim: int = 0, basis_seed=0, return_info=False):
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

    bases, information = [], []
    for L in layers:
        X = torch.stack(diffs[L], 0)  # [n, hidden]
        B, info = basis_from_differences(X, rank, complement, comp_dim, 1234 + L + basis_seed)
        bases.append(B.to(h.device))
        information.append({"layer": L, **info})
    return (bases, information) if return_info else bases


# ------------------------------------------------------------------- hooking


def hooks_for(h: Harness, layers, vecs, pos):
    handles = []

    def mk(v):
        def hook(m, i, o):
            t = (o[0] if isinstance(o, tuple) else o).clone()
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


def target_margin(logits, target):
    competitors = logits.clone()
    competitors[target] = -torch.inf
    return logits[target] - competitors.max()


def optimize_sample(h, r, layers, ctrl_ids, ctrl_base, bases, cfg):
    hidden = h.model.config.hidden_size
    ep = EditParams(layers, hidden, h.device, bases=bases)
    ep.trace = []
    opt = torch.optim.Adam(ep.parameters(), lr=cfg["lr"])
    objective = cfg.get("objective", "two_way_margin")
    if objective not in ("two_way_margin", "cross_entropy"):
        raise ValueError("Unknown intervention objective")

    with torch.no_grad():
        base_logits = h.model(r.corrupt_ids).logits[0, -1].detach()

    lam_kl = cfg["lam_kl"]
    guard_every = max(1, int(cfg["guard_every"]))

    # ---- Stage A: push the decision over the line -------------------------
    margin_a = None
    for step in range(cfg["stage_a_steps"]):
        logits = forward_logits(h, r.corrupt_ids, layers, ep.vectors(), r.dpos)
        margin = target_margin(logits, r.cid) if objective == "cross_entropy" else logits[r.cid] - logits[r.kid]
        loss = (-F.log_softmax(logits.float(), dim=0)[r.cid]
            if objective == "cross_entropy" else -margin)
        if lam_kl > 0:
            loss = loss + lam_kl * kl_nontarget(logits, base_logits, r.cid, r.kid)
        loss = loss + cfg["l2"] * sum((v.float() ** 2).mean() for v in ep.vectors())
        if not torch.isfinite(loss).item():
            raise ValueError("Non-finite optimization loss")
        ep.trace.append({"stage": "A", "step": step, "before_update": True,
                         "target_margin": float(target_margin(logits, r.cid).detach()),
                         "target_success": int(logits.argmax().item() == r.cid),
                         "edit_norm": ep.total_norm()})

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
        with torch.no_grad():
            final_a = forward_logits(h, r.corrupt_ids, layers, ep.vectors(), r.dpos)
            margin_a = float(target_margin(final_a, r.cid) if objective == "cross_entropy"
                             else final_a[r.cid] - final_a[r.kid])
        floor = max(cfg["margin_floor"], cfg["keep_frac"] * max(margin_a or 0.0, 0.0))
        opt_b = torch.optim.Adam(ep.parameters(), lr=cfg["lr_b"])
        for step in range(cfg["stage_b_steps"]):
            logits = forward_logits(h, r.corrupt_ids, layers, ep.vectors(), r.dpos)
            margin = target_margin(logits, r.cid) if objective == "cross_entropy" else logits[r.cid] - logits[r.kid]

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

    with torch.no_grad():
        final_logits = forward_logits(h, r.corrupt_ids, layers, ep.vectors(), r.dpos)
        ep.trace.append({"stage": "final", "step": cfg["stage_a_steps"] + (
            cfg["stage_b_steps"] if cfg["two_stage"] else 0), "before_update": False,
            "target_margin": float(target_margin(final_logits, r.cid)),
            "target_success": int(final_logits.argmax().item() == r.cid),
            "edit_norm": ep.total_norm()})
    return ep


def optimize_sample_converged(h, r, layers, ctrl_ids, ctrl_base, bases, cfg):
    """Single-stage optimizer with convergence-based stopping instead of a
    fixed step count.

    Motivated by reviewer Gap G1: comparing bases at a FIXED step budget
    confounds representational capacity with ease-of-optimization. A
    rank-constrained edit may need more steps (or a different effective
    learning rate) to reach the same loss as an unconstrained edit; stopping
    everyone at the same step count can make a basis look causally
    insufficient when it is merely slower to optimize.

    Stops when the loss changes by less than convergence_eps for
    convergence_patience consecutive steps, or after max_steps, whichever
    comes first. Returns (EditParams, diagnostics) where diagnostics records
    whether convergence was reached and how many steps were used, so a
    reviewer can audit whether ranks were compared under equal-effort
    optimization rather than equal step count.
    """
    hidden = h.model.config.hidden_size
    ep = EditParams(layers, hidden, h.device, bases=bases)
    ep.trace = []
    opt = torch.optim.Adam(ep.parameters(), lr=cfg["lr"])
    objective = cfg.get("objective", "cross_entropy")
    if objective not in ("two_way_margin", "cross_entropy"):
        raise ValueError("Unknown intervention objective")

    with torch.no_grad():
        base_logits = h.model(r.corrupt_ids).logits[0, -1].detach()

    lam_kl = cfg.get("lam_kl", 0.0)
    guard_every = max(1, int(cfg.get("guard_every", 4)))
    eps = cfg["convergence_eps"]
    patience = int(cfg["convergence_patience"])
    max_steps = int(cfg["max_steps"])
    if eps <= 0 or patience < 1 or max_steps < 1:
        raise ValueError("convergence_eps, convergence_patience, max_steps must be positive")

    prev_loss = None
    stable_steps = 0
    converged = False
    steps_taken = 0

    for step in range(max_steps):
        logits = forward_logits(h, r.corrupt_ids, layers, ep.vectors(), r.dpos)
        margin = target_margin(logits, r.cid) if objective == "cross_entropy" else logits[r.cid] - logits[r.kid]
        loss = (-F.log_softmax(logits.float(), dim=0)[r.cid]
                if objective == "cross_entropy" else -margin)
        if lam_kl > 0:
            loss = loss + lam_kl * kl_nontarget(logits, base_logits, r.cid, r.kid)
        loss = loss + cfg["l2"] * sum((v.float() ** 2).mean() for v in ep.vectors())
        if not torch.isfinite(loss).item():
            raise ValueError("Non-finite optimization loss")

        loss_value = float(loss.detach().item())
        ep.trace.append({"step": step, "loss": loss_value,
                          "target_margin": float(target_margin(logits, r.cid).detach()),
                          "target_success": int(logits.argmax().item() == r.cid),
                          "edit_norm": ep.total_norm()})

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ep.clamp_norm(cfg["norm_budget"])
        steps_taken = step + 1

        if (step + 1) % guard_every == 0:
            drop = control_degradation(h, layers, ep.detached(), ctrl_ids, ctrl_base)
            if drop > cfg["max_control_drop"]:
                ep.scale(0.7)

        if prev_loss is not None and abs(prev_loss - loss_value) < eps:
            stable_steps += 1
            if stable_steps >= patience:
                converged = True
                break
        else:
            stable_steps = 0
        prev_loss = loss_value

    with torch.no_grad():
        final_logits = forward_logits(h, r.corrupt_ids, layers, ep.vectors(), r.dpos)
    diagnostics = {"converged": converged, "steps_taken": steps_taken, "max_steps": max_steps,
                   "final_loss": prev_loss,
                   "final_target_margin": float(target_margin(final_logits, r.cid)),
                   "final_target_success": int(final_logits.argmax().item() == r.cid)}
    return ep, diagnostics


# -------------------------------------------------------------- evaluation


def eval_set(h, recs, layers, bases, cfg, ctrl_ids, ctrl_base):
    """Evaluate with a LADDER of decision metrics, most->least discontinuous.

    Schaeffer et al. (2304.15004) show discontinuous metrics manufacture
    apparent all-or-nothing behaviour while the underlying quantity moves
    smoothly. Our Finding 3 reproduced exactly that inside this method, so we
    now report every rung:

      steer        full-vocab argmax == clean          (most discontinuous)
      steer_2way   p(clean) > p(corrupt)               (intermediate)
      delta_p2way  change in p(clean)/(p(clean)+p(corrupt))  (continuous, bounded)
      delta_logit  change in logit(clean)              (continuous, unbounded)

    Also records `pred_is_corrupt`: when steering fails, is the model still
    emitting the corrupt token, or has it been pushed to some third token?
    That distinguishes "edit did nothing" from "edit broke the decision".
    """
    steer, brk, dlog, drops, norms = [], [], [], [], []
    steer2, dp2, pred_corrupt = [], [], []
    records = []

    for r in recs:
        ep = optimize_sample(h, r, layers, ctrl_ids, ctrl_base, bases, cfg)
        vecs = ep.detached()

        with torch.no_grad():
            base_logits = h.model(r.corrupt_ids).logits[0, -1]
            base_clean_pred = int(h.model(r.clean_ids).logits[0, -1].argmax().item())
            base_corrupt_pred = int(base_logits.argmax().item())
            base_clean = base_logits[r.cid].item()
            base_p2 = float(
                torch.softmax(
                    torch.stack([base_logits[r.cid], base_logits[r.kid]]).float(), 0
                )[0]
            )

            logits = forward_logits(h, r.corrupt_ids, layers, vecs, r.dpos)
            pred = int(logits.argmax().item())
            steer.append(int(pred == r.cid))
            pred_corrupt.append(int(pred == r.kid))
            dlog.append(float(logits[r.cid].item() - base_clean))

            p2 = float(
                torch.softmax(
                    torch.stack([logits[r.cid], logits[r.kid]]).float(), 0
                )[0]
            )
            steer2.append(int(p2 > 0.5))
            dp2.append(p2 - base_p2)

            neg = [-v for v in vecs]
            pb = int(
                forward_logits(h, r.clean_ids, layers, neg, r.dpos).argmax().item()
            )
            brk.append(int(pb != r.cid))
            positive_clean = int(forward_logits(h, r.clean_ids, layers, vecs, r.dpos).argmax().item())
            outcome = prediction_outcome(r.cid, r.kid, base_clean_pred, base_corrupt_pred,
                                         pred, positive_clean, pb)
            outcome.update({
                "sample_id": r.sample_id, "input_hash": r.input_hash,
                "delta_p2way": p2 - base_p2, "steer_2way": int(p2 > 0.5),
                "delta_logit": dlog[-1], "edit_norms": [float(vector.norm().item()) for vector in vecs],
                "target_probability": float(torch.softmax(logits.float(), 0)[r.cid].item()),
                "target_margin": float(target_margin(logits, r.cid)),
                "optimization_trace": getattr(ep, "trace", []),
            })
            records.append(outcome)

        drops.append(control_degradation(h, layers, vecs, ctrl_ids, ctrl_base))
        norms.append(ep.total_norm())

    m = lambda x: float(np.mean(x)) if x else 0.0
    return {
        **summarize_predictions(records),
        "steer": m(steer),
        "steer_2way": m(steer2),
        "break": m(brk),
        "delta_logit_clean": m(dlog),
        "delta_p2way": m(dp2),
        "pred_is_corrupt": m(pred_corrupt),
        "control_drop": m(drops),
        "edit_norm_mean": m(norms),
        "steer_flags": steer,
        "steer2_flags": steer2,
        "break_flags": brk,
        "delta_p2way_all": dp2,
        "records": records,
        "break_semantics": "legacy_unconditional_negative_edit_error; use negative_edit_disruption",
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


def measure_resid_norm(h: Harness, recs: list[PairRec], layers, max_samples: int = 8):
    """Mean residual-stream norm at the entity position over the target layers."""
    vals = []
    for r in recs[:max_samples]:
        outs = h.cache_layer_outputs(r.corrupt_ids)
        for L in layers:
            vals.append(outs[L][0, r.dpos].norm().item())
    return float(np.mean(vals)) if vals else 1.0


def run(args):
    device = pick_device(args.device)
    h = Harness(args.model, device)
    ds.restrict_to_single_token(h.tok)

    recs = prepare_pairs(h, args.task, args.n, args.fewshot, args.seed)
    tr, dv, te = split_records(recs, args.seed)
    layers = make_layers(len(h.layers)) if not args.layers else sorted(set(args.layers))
    ctrl_ids, ctrl_base = get_control_baseline(h)

    resid_norm = measure_resid_norm(h, tr, layers)
    if args.rel_budget is not None:
        args.norm_budget = args.rel_budget * resid_norm
        print(f"resid norm at entity pos = {resid_norm:.2f}  |  "
              f"rel_budget {args.rel_budget:.3f} -> norm_budget {args.norm_budget:.2f}")
    else:
        print(f"resid norm at entity pos = {resid_norm:.2f}  |  "
              f"norm_budget {args.norm_budget:.2f} "
              f"(= {args.norm_budget / resid_norm:.1%} relative)")

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

    s_mean, s_ci = rate_ci(mt["steer_flags"])
    b_mean, b_ci = rate_ci(mt["break_flags"])
    s2_mean, s2_ci = rate_ci(mt["steer2_flags"])

    print("\n=== PARETO RESULT (TEST) ===")
    print(
        f"steer={s_mean:.1%} [{s_ci[0]:.1%},{s_ci[1]:.1%}]  "
        f"steer2way={s2_mean:.1%} [{s2_ci[0]:.1%},{s2_ci[1]:.1%}]  "
        f"break={b_mean:.1%} [{b_ci[0]:.1%},{b_ci[1]:.1%}]"
    )
    print(
        f"dlogit={mt['delta_logit_clean']:+.3f}  "
        f"dp2way={mt['delta_p2way']:+.4f}  "
        f"pred_is_corrupt={mt['pred_is_corrupt']:.1%}  "
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
        "resid_norm": resid_norm,
        "rel_budget": (args.norm_budget / resid_norm) if resid_norm else None,
        "subspace_rank": args.subspace_rank,
        "subspace_rank_actual": rank_actual,
        "subspace_complement": bool(args.subspace_complement),
        "lr_autoscale": bool(args.lr_autoscale),
        "seed": args.seed,
        "test": {**mt, "steer": s_mean, "steer_ci": s_ci, "break": b_mean, "break_ci": b_ci,
                 "steer_2way": s2_mean, "steer_2way_ci": s2_ci},
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
    ap.add_argument(
        "--rel-budget",
        type=float,
        default=None,
        help="norm budget as a FRACTION of mean residual norm at the target "
             "position; overrides --norm-budget. Use for cross-architecture runs.",
    )
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
