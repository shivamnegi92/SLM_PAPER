"""Read-only consistency checks for saved study artifacts."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from analyze_dissociation import analyze
from capability import load_hellaswag
from collect_study import MODELS, TASKS, collect
from experiment_metrics import paired_comparison, prediction_outcome, rate_ci, summarize_predictions
from intervene_margin import CONTROL_PROMPTS
from localize import Harness
from public_benchmarks import load_arc, text_windows
from run_capability_study import paired_accuracy, window_summary
from run_validated_study import configuration, matched_layers, model_fingerprint, source_fingerprint
from study_data import fingerprint, load_manifest, manifest_records, write_new


CONDITIONS = ("active_prefix", "random_prefix", "zero_prefix", "global")


class TokenOnlyHarness(Harness):
    def __init__(self, model_path):
        self.device = "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON value in {path}: {value}")
    with Path(path).open() as stream:
        return json.load(stream, parse_constant=reject_constant)


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def assert_matches(actual, expected, label):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"{label}: field mismatch")
        for name, value in expected.items():
            assert_matches(actual[name], value, f"{label}.{name}")
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError(f"{label}: length mismatch")
        for index, (stored, rebuilt) in enumerate(zip(actual, expected)):
            assert_matches(stored, rebuilt, f"{label}[{index}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if (isinstance(actual, bool) or not isinstance(actual, (int, float))
                or not math.isfinite(actual) or not math.isfinite(expected)
                or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)):
            raise ValueError(f"{label}: numerical mismatch ({actual!r} != {expected!r})")
    elif actual != expected:
        raise ValueError(f"{label}: value mismatch ({actual!r} != {expected!r})")


def comparable_summary(summary):
    normalized = deepcopy(summary)
    for condition in normalized["summaries"].values():
        condition["source_files"] = [Path(path).name for path in condition["source_files"]]
    return normalized


def audit_paired_summary(root, seeds=(0, 1, 2)):
    root = Path(root)
    conditions = ["full", "track8", "comp8"]
    stored = read_json(root / "paired_summary.json")
    rebuilt = analyze("diss", conditions, list(seeds), root)
    assert_matches(comparable_summary(stored), comparable_summary(rebuilt), str(root))
    return {"n_records": sum(row["n"] for row in rebuilt["summaries"].values()),
            "n_seeds": len(seeds), "conditions": conditions}


def audit_choices(records, items, label):
    expected_ids = [fingerprint([context, endings, target]) for context, endings, target in items]
    assert_matches([row["id"] for row in records], expected_ids, f"{label}.identities")
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError(f"{label}: duplicate benchmark items")
    for record, (_, endings, target) in zip(records, items):
        scores = np.asarray(record["scores"], dtype=float)
        if scores.shape != (len(endings),) or not np.isfinite(scores).all():
            raise ValueError(f"{label}: invalid choice scores")
        prediction = int(scores.argmax())
        assert_matches(record["target"], target, f"{label}.target")
        assert_matches(record["prediction"], prediction, f"{label}.prediction")
        assert_matches(record["correct"], int(prediction == target), f"{label}.correct")


def audit_windows(windows, label):
    if len(windows) != 5:
        raise ValueError(f"{label}: expected five text windows")
    for window in windows:
        values = np.asarray(window, dtype=float)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"{label}: invalid token losses")


def audit_capability(run, benchmarks):
    if run.get("completed") is not True:
        raise ValueError("Incomplete capability result")
    if set(run["conditions"]) != set(CONDITIONS):
        raise ValueError("Capability conditions differ from the declared protocol")
    if set(run["baseline_benchmarks"]) != set(benchmarks):
        raise ValueError("Capability benchmark set differs")
    for name, items in benchmarks.items():
        audit_choices(run["baseline_benchmarks"][name], items, f"baseline.{name}")
    audit_windows(run["baseline_token_nll"], "baseline.text")
    active_norms = np.asarray(run["conditions"]["active_prefix"]["edit_norms"], dtype=float)
    budgets = np.asarray(run["cfg"]["norm_budget"], dtype=float)
    if (active_norms.shape != budgets.shape or active_norms.shape != (len(run["layers"]),)
            or not np.isfinite(active_norms).all() or not np.isfinite(budgets).all()
            or np.any(active_norms < 0) or np.any(budgets <= 0)
            or np.any(active_norms > budgets * (1 + 1e-5) + 1e-5)):
        raise ValueError("Capability edit norms violate the declared layer budgets")
    for condition, result in run["conditions"].items():
        if set(result["benchmarks"]) != set(benchmarks):
            raise ValueError(f"{condition}: missing benchmark")
        norms = np.asarray(result["edit_norms"], dtype=float)
        expected_norms = np.zeros_like(active_norms) if condition == "zero_prefix" else active_norms
        if (norms.shape != expected_norms.shape or not np.isfinite(norms).all()
                or not np.allclose(norms, expected_norms, atol=1e-5, rtol=1e-5)):
            raise ValueError(f"{condition}: edit norms do not match the control")
        for name, items in benchmarks.items():
            saved = result["benchmarks"][name]
            baseline = run["baseline_benchmarks"][name]
            audit_choices(saved["items"], items, f"{condition}.{name}")
            rebuilt = paired_accuracy(baseline, saved["items"])
            assert_matches(saved, rebuilt, f"{condition}.{name}")
            if condition == "zero_prefix":
                for clean, edited in zip(baseline, saved["items"]):
                    if not np.allclose(clean["scores"], edited["scores"], atol=1e-5, rtol=1e-5):
                        raise ValueError("Zero edit changed choice scores")
        audit_windows(result["token_nll"], f"{condition}.text")
        rebuilt_text = window_summary(run["baseline_token_nll"], result["token_nll"])
        assert_matches(result["text"], rebuilt_text, f"{condition}.text")
        if condition == "zero_prefix":
            for clean, edited in zip(run["baseline_token_nll"], result["token_nll"]):
                if not np.allclose(clean, edited, atol=1e-5, rtol=1e-5):
                    raise ValueError("Zero edit changed token losses")
    active = run["conditions"]["active_prefix"]
    return {"benchmark_items_per_edit": sum(len(items) for items in benchmarks.values()),
            "conditions": len(CONDITIONS), "text_windows": 5,
            "accuracy_bounds_met": {name: value["two_point_loss_bound_met"]
                                     for name, value in active["benchmarks"].items()},
            "text_bound_met": active["text"]["ten_percent_ratio_bound_met"]}


def audit_baseline(result, count, label):
    assert_matches(result["n"], count, f"{label}.n")
    for prefix, accuracy, interval in (("clean", "clean_accuracy", "clean_ci"),
                                       ("counterfactual", "counterfactual_accuracy", "counterfactual_ci")):
        flags = result[f"{prefix}_flags"]
        if len(flags) != count:
            raise ValueError(f"{label}: truncated baseline flags")
        mean, bounds = rate_ci(flags)
        assert_matches(result[accuracy], mean, f"{label}.{accuracy}")
        assert_matches(result[interval], bounds, f"{label}.{interval}")


def identity(record, seed):
    return {"sample_id": record.sample_id, "input_hash": record.input_hash, "seed": seed,
            "clean_target_id": record.cid, "corrupt_target_id": record.kid}


def audit_identity(row, expected, label):
    for name, value in expected.items():
        assert_matches(row[name], value, f"{label}.{name}")


def audit_prediction(row):
    names = ("clean_target_id", "corrupt_target_id", "baseline_clean_prediction",
             "baseline_corrupt_prediction", "steered_prediction", "positive_clean_prediction",
             "negative_clean_prediction")
    rebuilt = prediction_outcome(*(row[name] for name in names))
    audit_identity(row, rebuilt, row["sample_id"])


def audit_trace(trace, steps, label):
    if not trace or trace[-1]["stage"] != "final" or trace[-1]["step"] != steps:
        raise ValueError(f"{label}: missing final optimization trace")
    for point in trace:
        if not math.isfinite(point["target_margin"]) or not math.isfinite(point["edit_norm"]):
            raise ValueError(f"{label}: non-finite optimization trace")


def audit_intervention(run, expected_records, calibration, hidden_size, root):
    condition, seed = run["condition"], run["seed"]
    comparison = calibration["comparison"]
    label = f"{comparison['model']}/{comparison['task']}/{condition}/s{seed}"
    if run.get("completed") is not True or run["protocol_version"] != comparison["version"]:
        raise ValueError(f"{label}: incomplete intervention")
    assert_matches(run["comparison"], comparison, f"{label}.comparison")
    assert_matches(run["calibration_sha256"], fingerprint(calibration), f"{label}.calibration")
    executor = Path(__file__).with_name("run_study_cases.py")
    assert_matches(run["executor_sha256"], fingerprint(executor.read_text()), f"{label}.executor")
    budgets = (np.asarray(run["residual_norms"]) * comparison["relative_budget"]).tolist()
    rate = calibration["selected"][condition]
    if condition != "full":
        rate *= (hidden_size / 8) ** .5
    assert_matches(run["cfg"], configuration(budgets, comparison["steps"], rate), f"{label}.cfg")
    records = run["test"]["records"]
    if len(records) != len(expected_records):
        raise ValueError(f"{label}: wrong test-record count")
    drops = []
    for row, expected in zip(records, expected_records):
        audit_identity(row, identity(expected, seed), label)
        audit_prediction(row)
        audit_trace(row["optimization_trace"], comparison["steps"], label)
        terminal = row["optimization_trace"][-1]
        assert_matches(terminal["target_success"], row["steer"], f"{label}.terminal_success")
        if not math.isclose(terminal["target_margin"], row["target_margin"], abs_tol=1e-5, rel_tol=1e-5):
            raise ValueError(f"{label}: terminal margin mismatch")
        norms = np.asarray(row["edit_norms"])
        if (norms.shape != (len(budgets),) or not np.isfinite(norms).all()
                or np.any(norms < 0) or np.any(norms > np.asarray(budgets) * (1 + 1e-5) + 1e-5)):
            raise ValueError(f"{label}: recorded edit exceeded its layer budget")
        if not 0 <= row["target_probability"] <= 1:
            raise ValueError(f"{label}: invalid target probability")
        checkpoint = read_json(root / "test_cases" / f"{condition}_s{seed}" / f"{row['sample_id']}.json")
        if checkpoint.get("completed") is not True:
            raise ValueError(f"{label}: incomplete case checkpoint")
        for name in ("comparison", "cfg", "seed", "condition"):
            assert_matches(checkpoint[name], run[name], f"{label}.checkpoint.{name}")
        assert_matches(checkpoint["record"], row, f"{label}.checkpoint.record")
        drops.append(checkpoint["control_drop"])
    rebuilt = {**summarize_predictions(records), "records": records,
               "delta_p2way": float(np.mean([row["delta_p2way"] for row in records])),
               "control_drop": float(np.mean(drops))}
    assert_matches(run["test"], rebuilt, f"{label}.test")
    for section in ("tracking", "control"):
        values = run["basis_diagnostics"][section]
        assert_matches([row["layer"] for row in values], comparison["layers"], f"{label}.basis_layers")
        for value in values:
            rank = value["tracking_rank"] if section == "tracking" else value["rank"]
            error = value["orthonormal_error"] if section == "tracking" else value["orthogonal_error"]
            if rank != 8 or not math.isfinite(error) or error > 1e-4:
                raise ValueError(f"{label}: invalid recorded basis geometry")


def audit_heads(run, expected):
    if run.get("completed") is not True:
        raise ValueError("Incomplete head result")
    rows = run["records"]
    if run["n_test"] != len(rows) or len(rows) != len(expected):
        raise ValueError("Head sample count mismatch")
    for row, expected_row in zip(rows, expected):
        audit_identity(row, expected_row, "heads")
        if abs(row["baseline_receiver_clamp_delta"]) > 1e-4:
            raise ValueError("Baseline receiver clamp was not a no-op")
        valid = row["clean_corrupt_logit_gap"] > 1e-5
        for name in ("top8_faithfulness", "random8_faithfulness", "full_residual_faithfulness",
                     "tracking_projection_faithfulness", "control_projection_faithfulness"):
            if (row[name] is not None) != valid:
                raise ValueError("Invalid head normalization denominator")
    valid_rows = [row for row in rows if row["top8_faithfulness"] is not None]
    assert_matches(run["n_valid_logit_gaps"], len(valid_rows), "heads.valid_gaps")
    for name, first, second, population in (
            ("faithfulness_difference", "top8_faithfulness", "random8_faithfulness", valid_rows),
            ("compensation_vs_random", "receiver_clamp_compensation", "random_clamp_compensation", rows)):
        rebuilt = paired_comparison([{**row, "value": row[first]} for row in population],
                                    [{**row, "value": row[second]} for row in population],
                                    "value", binary=False) if population else None
        assert_matches(run[name], rebuilt, f"heads.{name}")
    return {"n_records": len(rows), "n_valid_logit_gaps": len(valid_rows)}


def without_location_metadata(value):
    if isinstance(value, list):
        return [without_location_metadata(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for name, item in value.items():
        if name == "source" and isinstance(item, str):
            result[name] = Path(item).name
        elif name == "source_files":
            result[name] = [Path(path).name for path in item]
        else:
            result[name] = without_location_metadata(item)
    return result


def audit_snapshot(saved, rebuilt):
    if saved.get("matrix_complete") is not True or saved.get("missing") != []:
        raise ValueError("Snapshot is incomplete")
    assert_matches(without_location_metadata(saved), without_location_metadata(rebuilt), "snapshot")


def audit_study(project, snapshot):
    project, snapshot = Path(project).resolve(), Path(snapshot).resolve()
    manifest = load_manifest(project / "data/validated_manifest_v1.json")
    assert_matches(manifest["seeds"], [0, 1, 2], "manifest.seeds")
    assert_matches(manifest["counts"], {"train": 64, "dev": 24, "test": 50, "fewshot": 3}, "manifest.counts")
    study, capability, heads = (project / "results" / name for name in
                                ("validated_v1", "capability_benchmarks_v1", "heads_validated_v1"))
    core_hash = source_fingerprint()
    scorer_hash = fingerprint({name: (project / "src" / name).read_text() for name in
                               ("run_capability_study.py", "capability.py", "capability_deployed.py")})
    benchmarks = {"hellaswag": load_hellaswag(project / "data_bench/hellaswag_val.jsonl", 240),
                  "arc_easy": load_arc(project / "data_bench/arc_easy_test_200.json")}
    windows = text_windows((project / "data_bench/tinyshakespeare.txt").read_text(), 5, 2000)
    report = {"audit_version": "postrun_consistency_v1", "passed": True,
              "manifest_sha256": manifest["sha256"], "core_source_sha256": core_hash,
              "scorer_sha256": scorer_hash, "paired_studies": [], "capability_runs": [],
              "head_runs": [], "baseline_only_runs": 0,
              "limitations": [
                  "Recomputes saved evidence; does not rerun model forward passes or prove historical logits.",
                  "Original model fingerprints cover configuration and root weight sizes, not weight content.",
                  "Head runs lack run-time model/source content hashes; post-run hashes cannot recover them.",
                  "Baseline-only test files contain flags but no item identities; counts and rates are auditable, not original ordering.",
                  "Scalar continuous intervention changes lack saved full logits for independent reconstruction.",
                  "Recorded basis diagnostics and optimizer traces do not prove convergence or causal completeness.",
                  "Path spelling is normalized; statistical/content consistency is checked independently of checkout location."]}
    for model in MODELS:
        model_path = project.parent / model
        model_hash = model_fingerprint(model_path)
        model_config = read_json(model_path / "config.json")
        layers = matched_layers(model_config["num_hidden_layers"])
        harness = TokenOnlyHarness(model_path)
        intermediate_records, full_runs = {}, {}
        for task in TASKS:
            root = study / model / task
            calibration = read_json(root / "calibration.json")
            if calibration.get("completed") is not True:
                raise ValueError(f"Incomplete calibration: {model}/{task}")
            comparison = calibration["comparison"]
            required = {"version": "tracking_control_v1", "manifest_sha256": manifest["sha256"],
                        "model": model, "model_fingerprint": model_hash, "task": task,
                        "source_sha256": core_hash, "relative_budget": .3, "steps": 32,
                        "objective": "cross_entropy", "lr_grid": [.025, .05, .1],
                        "seed_policy": [0, 1, 2], "dtype": "float32", "layers": layers,
                        "guard_prompts": CONTROL_PROMPTS}
            audit_identity(comparison, required, f"{model}/{task}.comparison")
            audit_baseline(calibration["baseline_dev"], 24, "development")
            eligible = min(calibration["baseline_dev"][name] for name in
                           ("clean_accuracy", "counterfactual_accuracy")) >= .8
            assert_matches(calibration["eligible_for_tracking_claim"], eligible, "competence_gate")
            if eligible:
                report["paired_studies"].append({"model": model, "task": task,
                                                  **audit_paired_summary(root)})
                for condition, candidates in calibration["candidates"].items():
                    assert_matches([choice["base_lr"] for choice in candidates], [.025, .05, .1], "calibration.grid")
                    for choice in candidates:
                        assert_matches(len(choice["traces"]), 4, "calibration.n_traces")
                        for trace in choice["traces"]:
                            audit_trace(trace, 32, "calibration")
                        assert_matches(choice["mean_target_margin"],
                                       float(np.mean([trace[-1]["target_margin"] for trace in choice["traces"]])),
                                       "calibration.margin")
                    best = max(candidates, key=lambda choice: choice["mean_target_margin"])
                    assert_matches(calibration["selected"][condition], best["base_lr"], "calibration.selected")
            for seed in manifest["seeds"]:
                splits = manifest_records(harness, manifest, task, seed)
                if task == "intermediate":
                    intermediate_records[seed] = splits["test"]
                if not eligible:
                    baseline = read_json(root / f"baseline_s{seed}.json")
                    if baseline.get("completed") is not True or baseline["seed"] != seed:
                        raise ValueError("Incomplete or misidentified baseline-only run")
                    assert_matches(baseline["comparison"], comparison, "baseline.comparison")
                    audit_baseline(baseline["baseline_test"], 50, "baseline_test")
                    report["baseline_only_runs"] += 1
                    continue
                anchor = None
                for condition in ("full", "track8", "comp8"):
                    run = read_json(root / f"diss_{condition}_s{seed}.json")
                    audit_identity(run, {"condition": condition, "seed": seed}, "run.filename")
                    assert_matches(run["split"], {"train": 64, "dev": 24, "test": 50}, "run.split")
                    audit_intervention(run, splits["test"], calibration, model_config["hidden_size"], root)
                    if anchor is not None:
                        for key in ("residual_norms", "basis_diagnostics"):
                            assert_matches(run[key], anchor[key], f"paired.{key}")
                    else:
                        anchor = run
                    if task == "intermediate" and condition == "full":
                        full_runs[seed] = run
            print(f"Audited {model}/{task}", flush=True)
        head_run = read_json(heads / f"heads_{model}_intermediate.json")
        audit_identity(head_run, {"model": model, "task": "intermediate", "manifest_sha256": manifest["sha256"],
                                  "protocol": "heldout_heads_compensation_v1", "n_discovery": 12,
                                  "n_test": 72, "projection_layers": layers}, "heads.metadata")
        expected_heads = []
        for seed in manifest["seeds"]:
            for record, raw in zip(intermediate_records[seed][:24], full_runs[seed]["test"]["records"][:24]):
                expected_heads.append({**identity(record, seed),
                                       "baseline_clean_correct": raw["baseline_clean_correct"],
                                       "baseline_corrupt_correct": raw["baseline_corrupt_correct"]})
        report["head_runs"].append({"model": model, **audit_heads(head_run, expected_heads)})
        previous = None
        for seed in manifest["seeds"]:
            root = capability / model
            run = read_json(root / f"capability_s{seed}.json")
            required = {"protocol": "capability_benchmarks_v1", "model": model,
                        "model_fingerprint": model_hash, "manifest": manifest["sha256"],
                        "core_source_sha256": core_hash, "scorer_sha256": scorer_hash,
                        "items": {name: fingerprint(items) for name, items in benchmarks.items()},
                        "windows": [fingerprint(window) for window in windows],
                        "prefix_tokens": 32, "max_tokens": 512, "steps": 32, "relative_budget": .3}
            assert_matches(run["comparison"], required, "capability.comparison")
            record = intermediate_records[seed][0]
            audit_identity(run, {"seed": seed, "edit_sample_id": record.sample_id,
                                 "input_hash": record.input_hash, "layers": layers,
                                 "cfg": full_runs[seed]["cfg"]}, "capability.identity")
            report["capability_runs"].append({"model": model, "seed": seed, **audit_capability(run, benchmarks)})
            if previous is not None:
                for name in ("baseline_benchmarks", "baseline_token_nll"):
                    assert_matches(run[name], previous[name], f"capability.reused_{name}")
            previous = run
            edit = torch.load(root / "edits" / f"edit_s{seed}.pt", map_location="cpu", weights_only=True)
            audit_identity(edit, {"comparison": required, "sample_id": record.sample_id,
                                  "cfg": run["cfg"], "trace": run["optimization_trace"]}, "capability.edit")
            norms = [float(vector.norm()) for vector in edit["vectors"]]
            if not np.allclose(norms, run["conditions"]["active_prefix"]["edit_norms"], atol=1e-5, rtol=1e-5):
                raise ValueError("Capability checkpoint vector norms differ")
            for condition in CONDITIONS:
                saved_condition = read_json(root / "condition_cases" / f"{condition}_s{seed}.json")
                assert_matches(saved_condition, {"comparison": required, "result": run["conditions"][condition]},
                               "capability.condition_checkpoint")
                for benchmark in benchmarks:
                    checkpoint = read_json(root / "benchmark_cases" / f"{condition}_{benchmark}_s{seed}.json")
                    assert_matches(checkpoint, {"comparison": required,
                                               "items": run["conditions"][condition]["benchmarks"][benchmark]["items"]},
                                   "capability.benchmark_checkpoint")
            print(f"Audited {model}/capability/seed{seed}", flush=True)
    audit_snapshot(read_json(snapshot / "ledger.json"), collect(study, capability, heads))
    report["snapshot_matches_sources"] = True
    report["snapshot_sha256"] = file_sha256(snapshot / "ledger.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    report = audit_study(args.project, args.snapshot)
    write_new(args.output, report)
    print(f"Audit passed with {len(report['limitations'])} scope limitations; saved {args.output}")


if __name__ == "__main__":
    main()