"""Instrumented development trajectories with the frozen study's Adam updates."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as functional

from intervene_margin import control_degradation
from intervene_pareto import EditParams, forward_logits, target_margin


def validate_checkpoints(checkpoints, steps):
    if (not checkpoints or list(checkpoints) != sorted(set(checkpoints))
            or any(not isinstance(step, int) or isinstance(step, bool) or step < 1 for step in checkpoints)
            or checkpoints[-1] != steps):
        raise ValueError("Checkpoints must be distinct positive steps ending at the trajectory length")


def point_metrics(logits, target, params, step):
    if not torch.isfinite(logits).all().item():
        raise ValueError("Non-finite trajectory logits")
    return {"step": step, "prediction": int(logits.argmax().item()),
            "target_success": int(logits.argmax().item() == target),
            "target_margin": float(target_margin(logits, target).detach()),
            "target_probability": float(logits.float().softmax(0)[target].detach()),
            "target_cross_entropy": float(-functional.log_softmax(logits.float(), 0)[target].detach()),
            "edit_norms": [float(vector.detach().norm()) for vector in params.vectors()]}


def optimize_trajectory(harness, record, layers, controls, control_predictions, bases, cfg, checkpoints):
    steps = cfg["stage_a_steps"]
    validate_checkpoints(checkpoints, steps)
    if cfg.get("objective") != "cross_entropy" or cfg["two_stage"] or cfg["lam_kl"] != 0:
        raise ValueError("Pilot supports only the declared single-stage cross-entropy Adam objective")
    params = EditParams(layers, harness.model.config.hidden_size, harness.device, bases=bases)
    optimizer = torch.optim.Adam(params.parameters(), lr=cfg["lr"])
    traces, snapshots, guard_events = [], {}, []
    for step in range(steps + 1):
        logits = forward_logits(harness, record.corrupt_ids, layers, params.vectors(), record.dpos)
        measured = point_metrics(logits, record.cid, params, step)
        traces.append(measured)
        if step == 0 or step in checkpoints:
            snapshots[str(step)] = {
                "metrics": dict(measured), "logits": logits.detach().float().cpu().clone(),
                "vectors": [vector.detach().float().cpu().clone() for vector in params.vectors()],
            }
        if step == steps:
            break
        loss = -functional.log_softmax(logits.float(), 0)[record.cid]
        loss = loss + cfg["l2"] * sum((vector.float() ** 2).mean() for vector in params.vectors())
        if not torch.isfinite(loss).item():
            raise ValueError("Non-finite trajectory loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = math.sqrt(sum(float(parameter.grad.detach().float().square().sum())
                                      for parameter in params.parameters()))
        if not math.isfinite(gradient_norm):
            raise ValueError("Non-finite trajectory gradient")
        measured["parameter_gradient_norm"] = gradient_norm
        optimizer.step()
        params.clamp_norm(cfg["norm_budget"])
        if (step + 1) % max(1, int(cfg["guard_every"])) == 0:
            drop = control_degradation(harness, layers, params.detached(), controls, control_predictions)
            shrink = drop > cfg["max_control_drop"]
            if shrink:
                params.scale(.7)
            guard_events.append({"step": step + 1, "pre_shrink_control_drop": drop, "shrunk": shrink})
    return {"trace": traces, "snapshots": snapshots, "guard_events": guard_events,
            "ever_target_success": any(point["target_success"] for point in traces),
            "best_seen_target_margin": max(point["target_margin"] for point in traces),
            "selection_note": "Best-seen values are development diagnostics, not a new success/stopping rule."}