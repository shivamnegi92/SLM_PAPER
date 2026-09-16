"""Guard/rate controls around the unchanged development trajectory kernel."""
from __future__ import annotations

from itertools import product

import numpy as np
import torch

from audit_study import assert_matches
from convergence_pilot import optimize_trajectory
from intervene_pareto import forward_logits
from run_validated_study import configuration


def cells(protocol):
    return [{"method": method, "guard": guard, "rate_factor": factor,
             "key": f"{method}_{guard}_rate{factor:g}"}
            for method, guard, factor in product(protocol["conditions"], protocol["guard_modes"], protocol["rate_factors"])]


def cell_config(budgets, base_effective_rate, cell):
    if cell["guard"] not in ("shrink", "observe") or cell["rate_factor"] not in (1., .25):
        raise ValueError("Undeclared guard/rate cell")
    cfg = configuration(budgets, 128, base_effective_rate * cell["rate_factor"])
    cfg["max_control_drop"] = .2 if cell["guard"] == "shrink" else 1.
    return cfg


@torch.no_grad()
def checkpoint_controls(harness, controls, baseline, layers, snapshots):
    if not controls or len(controls) != len(baseline):
        raise ValueError("Guard controls must pair with baseline predictions")
    results = {}
    for step, snapshot in snapshots.items():
        vectors = [vector.to(harness.device) for vector in snapshot["vectors"]]
        scores = [forward_logits(harness, ids, layers, vectors, -1).detach().float().cpu().clone() for ids in controls]
        if any(not torch.isfinite(score).all() for score in scores):
            raise ValueError("Non-finite checkpoint guard scores")
        predictions = [int(score.argmax()) for score in scores]
        results[step] = {"baseline_predictions": list(baseline), "predictions": predictions,
                         "logits": scores,
                         "disagreements": [int(prediction != expected) for prediction, expected in zip(predictions, baseline)]}
    return results


def run_cell(harness, record, layers, controls, baseline, bases, cfg, checkpoints):
    trajectory = optimize_trajectory(harness, record, layers, controls, baseline, bases, cfg, checkpoints)
    guarded = checkpoint_controls(harness, controls, baseline, layers, trajectory["snapshots"])
    return {"trajectory": trajectory, "checkpoint_controls": guarded}


def validate_guard_records(saved):
    cfg, trajectory = saved["cfg"], saved["trajectory"]
    events = trajectory["guard_events"]
    assert_matches([event["step"] for event in events], list(range(4, 129, 4)), "guard.steps")
    for event in events:
        drop = event["pre_shrink_control_drop"]
        if not np.isfinite(drop) or not 0 <= drop <= 1:
            raise ValueError("Invalid guard disagreement rate")
        assert_matches(event["shrunk"], drop > cfg["max_control_drop"], "guard.policy")
    assert_matches(sorted(saved["checkpoint_controls"]), sorted(trajectory["snapshots"]), "guard.checkpoints")
    for scores in saved["checkpoint_controls"].values():
        if len(scores["logits"]) != len(scores["baseline_predictions"]) or not scores["logits"]:
            raise ValueError("Truncated checkpoint guard scores")
        if any(score.ndim != 1 or not torch.isfinite(score).all() for score in scores["logits"]):
            raise ValueError("Invalid checkpoint guard logits")
        predictions = [int(score.argmax()) for score in scores["logits"]]
        assert_matches(scores["predictions"], predictions, "guard.predictions")
        assert_matches(scores["disagreements"], [int(prediction != baseline) for prediction, baseline in
                                                zip(predictions, scores["baseline_predictions"])], "guard.disagreements")