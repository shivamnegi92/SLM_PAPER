"""Execute a predeclared, paired guard-by-learning-rate development diagnostic."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
import time

import numpy as np
import torch

from audit_study import assert_matches, file_sha256, read_json
from experiment_metrics import paired_delta_ci
from guard_rate_pilot import cell_config, cells, run_cell, validate_guard_records
from intervene_margin import CONTROL_PROMPTS
from localize import Harness, pick_device
from run_convergence_pilot import (build_comparison, compute_lock, input_identity, pilot_records,
                                   validate_case as validate_trajectory_case, write_tensor_case)
from run_validated_study import basis_sets, control_set, effective_rate, layer_budgets
from study_data import fingerprint, write_new


def validate_protocol(protocol):
    required = {"version": "guard_rate_pilot_v1", "scope": "development_only_guard_rate_factorial_not_confirmatory",
                "model": "phi-3.5-mini", "task": "transfer", "training_seed": 2, "development_seed": 2,
                "development_indices": [0, 1, 2, 3], "calibration_seed": 0,
                "conditions": ["full", "track8"], "guard_modes": ["shrink", "observe"],
                "rate_factors": [1., .25], "optimizer": "Adam", "initialization": "zero",
                "checkpoints": [32, 64, 128], "relative_budget": .3, "precision": "float32",
                "objective": "cross_entropy", "output": "results/guard_rate_pilot_v1",
                "stopping_rule": "complete_all_32_trajectories_no_data_dependent_extension"}
    for name, value in required.items():
        assert_matches(protocol[name], value, f"protocol.{name}")


def validate_saved(saved, comparison, cell, expected, setup):
    validate_trajectory_case(saved, comparison, cell["key"], expected)
    assert_matches(saved["cell"], cell, "case.cell")
    assert_matches(saved["cfg"], setup["configs"][cell["key"]], "case.cfg")
    assert_matches(saved["setup_sha256"], fingerprint(setup), "case.setup")
    validate_guard_records(saved)
    for scores in saved["checkpoint_controls"].values():
        assert_matches(scores["baseline_predictions"], setup["control_predictions"], "case.guard_baseline")
    for step, snapshot in saved["trajectory"]["snapshots"].items():
        logits = snapshot["logits"]
        target = expected["clean_target_id"]
        competitors = logits.clone()
        competitors[target] = -torch.inf
        measured = snapshot["metrics"]
        margin = float(logits[target] - competitors.max())
        if not np.isclose(measured["target_margin"], margin, atol=1e-5, rtol=1e-5):
            raise ValueError("Checkpoint margin does not match saved logits")
        trace_point = saved["trajectory"]["trace"][int(step)]
        for name, value in measured.items():
            assert_matches(trace_point[name], value, f"case.trace.{name}")


def load_all(output, comparison, setup, require_complete):
    result = {}
    for cell in cells(comparison["protocol"]):
        rows = []
        for expected in comparison["development_inputs"]:
            path = output / "cases" / cell["key"] / f"{expected['sample_id']}.pt"
            if path.exists():
                saved = torch.load(path, map_location="cpu", weights_only=True)
                validate_saved(saved, comparison, cell, expected, setup)
                rows.append(saved)
            elif require_complete:
                raise ValueError(f"Missing declared guard/rate case: {cell['key']}")
        result[cell["key"]] = rows
    return result


def summarize(all_cases, comparison):
    design = cells(comparison["protocol"])
    identifiers = [record["sample_id"] for record in comparison["development_inputs"]]
    if len(identifiers) != 4 or len(set(identifiers)) != 4 or set(all_cases) != {cell["key"] for cell in design}:
        raise ValueError("Guard/rate summary requires the declared four-pair factorial")
    output = {"completed": True, "protocol": comparison["protocol"]["version"],
              "comparison_sha256": fingerprint(comparison), "n_development_pairs": 4,
              "n_trajectories": 32, "cells": {}, "contrasts": [],
              "limitations": comparison["protocol"]["limitations"],
              "reviewer_gap_status": "G1 remains open: no optimizer convergence, basis superiority or safety proof"}
    for cell in design:
        rows = all_cases[cell["key"]]
        if [row["input"]["sample_id"] for row in rows] != identifiers:
            raise ValueError("Missing, duplicated or misordered paired cases")
        checkpoints = {}
        for step in comparison["protocol"]["checkpoints"]:
            points = [row["trajectory"]["snapshots"][str(step)]["metrics"] for row in rows]
            agreements = [row["checkpoint_controls"][str(step)]["disagreements"] for row in rows]
            checkpoints[str(step)] = {"successes": sum(point["target_success"] for point in points), "n": 4,
                                      "mean_target_margin": float(np.mean([point["target_margin"] for point in points])),
                                      "mean_guard_disagreement": float(np.mean(agreements)),
                                      "cases_with_any_guard_disagreement": sum(any(flags) for flags in agreements)}
        output["cells"][cell["key"]] = {"cell": cell, "checkpoints": checkpoints,
                                         "shrink_events": sum(event["shrunk"] for row in rows for event in row["trajectory"]["guard_events"]),
                                         "observed_guard_breaches": sum(event["pre_shrink_control_drop"] > .2 for row in rows for event in row["trajectory"]["guard_events"]),
                                         "elapsed_seconds": sum(row["elapsed_seconds"] for row in rows)}
    for cell in design:
        pairs = []
        if cell["guard"] == "shrink":
            pairs.append((f"{cell['method']}_observe_rate{cell['rate_factor']:g}", "observe_minus_shrink"))
        if cell["rate_factor"] == 1.:
            pairs.append((f"{cell['method']}_{cell['guard']}_rate0.25", "quarter_minus_original_rate"))
        for other, label in pairs:
            first = [row["trajectory"]["snapshots"]["128"]["metrics"]["target_success"] for row in all_cases[cell["key"]]]
            second = [row["trajectory"]["snapshots"]["128"]["metrics"]["target_success"] for row in all_cases[other]]
            change, interval = paired_delta_ci(first, second)
            baseline = output["cells"][cell["key"]]["checkpoints"]["128"]
            altered = output["cells"][other]["checkpoints"]["128"]
            output["contrasts"].append({"comparison": label, "first_cell": other, "second_cell": cell["key"],
                                        "override_difference": change, "paired_ci": interval,
                                        "mean_margin_difference": altered["mean_target_margin"] - baseline["mean_target_margin"],
                                        "mean_guard_disagreement_difference": altered["mean_guard_disagreement"] - baseline["mean_guard_disagreement"]})
    return output


def markdown(summary):
    lines = ["# Guard and Learning-Rate Pilot v1", "",
             "Paired development-only diagnostic: four Phi transfer pairs, eight conditions. "
             "No original test result is changed. Observe-only means no shrink enforcement, not safety.", "",
             "| Method | Guard | Rate factor | Overrides 32/64/128 (each /4) | Margin at 128 | Guard disagreement at 128 | Shrinks |",
             "|---|---|---:|---|---:|---:|---:|"]
    for values in summary["cells"].values():
        cell, points = values["cell"], values["checkpoints"]
        counts = "/".join(str(points[str(step)]["successes"]) for step in (32, 64, 128))
        lines.append(f"| {cell['method']} | {cell['guard']} | {cell['rate_factor']:g} | {counts} | "
                     f"{points['128']['mean_target_margin']:.4f} | {points['128']['mean_guard_disagreement']:.1%} | {values['shrink_events']} |")
    lines.extend(["", "Guard disagreement is the mean across four edits and three optimizer-visible prompts, "
                  "not held-out accuracy or a population estimate. Full checkpoint logits, edit vectors, "
                  "and guard predictions remain in the per-case tensor artifacts.", "",
                  "| Contrast at step 128 | First cell | Second cell | Override difference [paired interval], pp | Margin difference |",
                  "|---|---|---|---|---:|"])
    for contrast in summary["contrasts"]:
        lower, upper = contrast["paired_ci"]
        lines.append(f"| {contrast['comparison']} | {contrast['first_cell']} | {contrast['second_cell']} | "
                     f"{contrast['override_difference'] * 100:.1f} [{lower * 100:.1f}, {upper * 100:.1f}] | {contrast['mean_margin_difference']:.4f} |")
    lines.extend(["", "Four-pair intervals are wide; the crossed factors allow a local diagnosis, not a "
                  "population or general mechanism conclusion. All contrasts were declared before this run.", "",
                  "## Remaining Gap", "", summary["reviewer_gap_status"], ""])
    lines.extend(f"- {value}" for value in summary["limitations"])
    return "\n".join(lines) + "\n"


def run(project, protocol_path, device):
    protocol = read_json(protocol_path)
    validate_protocol(protocol)
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable")
    with compute_lock(project / "results/revision_compute"):
        comparison, manifest, calibration = build_comparison(project, protocol, device)
        comparison["source_sha256"].update({name: file_sha256(project / "src" / name)
                                            for name in ("guard_rate_pilot.py", "run_guard_rate_pilot.py")})
        output = project / protocol["output"]
        run_path = output / "run.json"
        if run_path.exists():
            assert_matches(read_json(run_path)["comparison"], comparison, "resume.comparison")
        else:
            write_new(run_path, {"comparison": comparison, "created_at_utc": datetime.now(timezone.utc).isoformat()})
        setup_path = output / "setup.json"
        setup = read_json(setup_path) if setup_path.exists() else None
        existing = load_all(output, comparison, setup, False) if setup else {}
        if not existing or any(len(rows) != 4 for rows in existing.values()):
            harness = Harness(str(project.parent / protocol["model"]), device)
            harness.model.requires_grad_(False)
            train, development = pilot_records(harness, manifest, protocol)
            assert_matches([input_identity(record) for record in development], comparison["development_inputs"], "live.inputs")
            layers = comparison["layers"]
            controls, baseline = control_set(harness, CONTROL_PROMPTS)
            budgets, residual_norms = layer_budgets(harness, train, layers, .3)
            bases, geometry = basis_sets(harness, train, layers, protocol["training_seed"])
            design = cells(protocol)
            configs = {cell["key"]: cell_config(budgets, effective_rate(harness, bases[cell["method"]],
                                                                       calibration["selected"][cell["method"]]), cell)
                       for cell in design}
            current_setup = {"configs": configs, "residual_norms": residual_norms, "basis_diagnostics": geometry,
                             "control_token_ids": [ids[0].tolist() for ids in controls], "control_predictions": baseline,
                             "comparison_sha256": fingerprint(comparison)}
            if setup is None:
                write_new(setup_path, current_setup)
                setup = current_setup
            else:
                assert_matches(setup, current_setup, "resume.setup")
            for cell in design:
                for index, record in enumerate(development):
                    expected = input_identity(record)
                    destination = output / "cases" / cell["key"] / f"{record.sample_id}.pt"
                    if destination.exists():
                        continue
                    started = time.monotonic()
                    measured = run_cell(harness, record, layers, controls, baseline, bases[cell["method"]],
                                        configs[cell["key"]], protocol["checkpoints"])
                    saved = {"completed": True, "comparison_sha256": fingerprint(comparison),
                             "setup_sha256": fingerprint(setup), "condition": cell["key"], "cell": cell,
                             "input": expected, "cfg": configs[cell["key"]], **measured,
                             "elapsed_seconds": time.monotonic() - started}
                    validate_saved(saved, comparison, cell, expected, setup)
                    write_tensor_case(destination, saved)
                    checkpoints = [measured["trajectory"]["snapshots"][str(step)]["metrics"]["target_success"] for step in (32, 64, 128)]
                    print(f"SAVED {cell['key']} dev{index + 1}/4: checkpoints={checkpoints}; {saved['elapsed_seconds']:.1f}s", flush=True)
                    del measured, saved
                    gc.collect()
                    if device == "mps":
                        torch.mps.synchronize()
                        torch.mps.empty_cache()
            del harness, bases
        for name, digest in comparison["source_sha256"].items():
            assert_matches(file_sha256(project / "src" / name), digest, "postrun.source")
        result = summarize(load_all(output, comparison, setup, True), comparison)
        if (output / "summary.json").exists():
            assert_matches(read_json(output / "summary.json"), result, "resume.summary")
        else:
            write_new(output / "summary.json", result)
        text = markdown(result)
        if (output / "RESULTS.md").exists():
            assert_matches((output / "RESULTS.md").read_text(), text, "resume.report")
        else:
            with (output / "RESULTS.md").open("x") as stream:
                stream.write(text)
        print("Guard/rate pilot complete: 32 trajectories; no general safety or convergence claim", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", type=Path, default=Path(__file__).resolve().parents[1] / "protocols/guard_rate_pilot_v1.json")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    args = parser.parse_args()
    run(args.project.resolve(), args.protocol.resolve(), pick_device(args.device))


if __name__ == "__main__":
    main()