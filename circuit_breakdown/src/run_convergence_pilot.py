"""Run the separately declared development-only optimizer convergence pilot."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gc
from importlib import metadata
import os
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
import torch

from audit_study import TokenOnlyHarness, assert_matches, file_sha256, read_json
from capture_reproducibility import verify_files
from convergence_pilot import optimize_trajectory, validate_checkpoints
import dataset as dataset_module
from experiment_metrics import paired_delta_ci, rate_ci
from intervene_margin import CONTROL_PROMPTS, encode_pair
from localize import Harness
from run_validated_study import basis_sets, configuration, control_set, effective_rate, layer_budgets, source_fingerprint
from study_data import fingerprint, load_manifest, write_new


def validate_protocol(protocol):
    if (protocol["version"] != "convergence_pilot_v1"
            or protocol["scope"] != "development_only_diagnostic_not_confirmatory"
            or protocol["conditions"] != ["full", "track8"]
            or protocol["optimizer"] != "Adam" or protocol["initialization"] != "zero"
            or protocol["precision"] != "float32" or protocol["objective"] != "cross_entropy"
            or protocol["relative_budget"] != .3 or protocol["task"] != "transfer"
            or protocol["model"] != "phi-3.5-mini"
            or protocol["training_seed"] != protocol["development_seed"]
            or protocol["calibration_seed"] != 0):
        raise ValueError("Protocol differs from the supported, declared convergence pilot")
    validate_checkpoints(protocol["checkpoints"], 128)
    if protocol["checkpoints"] != [32, 64, 128]:
        raise ValueError("The pilot requires checkpoints 32, 64 and 128")
    indexes = protocol["development_indices"]
    if indexes != list(range(8)) or protocol["development_seed"] != 1:
        raise ValueError("The pilot uses the first eight seed-1 development examples only")


def pilot_records(harness, manifest, protocol):
    splits = manifest["tasks"][protocol["task"]][str(protocol["development_seed"])]
    prefix = "".join(f"{pair['clean_prompt']} {pair['clean_target']}.\n" for pair in splits["fewshot"])
    train = [encode_pair(harness, dataset_module.Pair(**pair), prefix) for pair in splits["train"]]
    development = [encode_pair(harness, dataset_module.Pair(**splits["dev"][index]), prefix)
                   for index in protocol["development_indices"]]
    if set(record.sample_id for record in train) & set(record.sample_id for record in development):
        raise ValueError("Training and development samples overlap")
    return train, development


def input_identity(record):
    return {"sample_id": record.sample_id, "input_hash": record.input_hash,
            "clean_target_id": record.cid, "corrupt_target_id": record.kid, "edited_position": record.dpos,
            "clean_token_ids": record.clean_ids[0].tolist(), "corrupt_token_ids": record.corrupt_ids[0].tolist()}


def build_comparison(project, protocol, device):
    manifest = load_manifest(project / protocol["manifest"])
    assert_matches(manifest["sha256"], protocol["manifest_sha256"], "manifest")
    capture = read_json(project / protocol["provenance_reference"])
    assert_matches(capture["sha256"], fingerprint({key: value for key, value in capture.items() if key != "sha256"}),
                   "provenance_reference")
    assets = capture["models"][protocol["model"]]
    model = project.parent / protocol["model"]
    print("Verifying model-file content before loading weights", flush=True)
    verify_files(assets, model)
    calibration = read_json(project / "results/validated_v1" / protocol["model"] / protocol["task"] / "calibration.json")
    assert_matches(calibration["comparison"]["source_sha256"], source_fingerprint(), "frozen_source")
    assert_matches(calibration["comparison"]["manifest_sha256"], manifest["sha256"], "calibration_manifest")
    if not calibration.get("completed") or not calibration["eligible_for_tracking_claim"]:
        raise ValueError("Original calibration is incomplete or task-ineligible")
    names = ("run_convergence_pilot.py", "convergence_pilot.py", "localize.py", "intervene_margin.py",
             "intervene_pareto.py", "run_validated_study.py", "study_data.py", "dataset.py",
             "experiment_metrics.py", "audit_study.py", "capture_reproducibility.py")
    tokenizer = TokenOnlyHarness(model)
    train, development = pilot_records(tokenizer, manifest, protocol)
    comparison = {"protocol": protocol, "protocol_sha256": fingerprint(protocol),
                  "source_sha256": {name: file_sha256(project / "src" / name) for name in names},
                  "frozen_core_sha256": source_fingerprint(), "calibration_sha256": fingerprint(calibration),
                  "model_assets": assets, "manifest_sha256": manifest["sha256"],
                  "layers": calibration["comparison"]["layers"],
                  "base_learning_rates": calibration["selected"],
                  "training_ids": [record.sample_id for record in train],
                  "development_inputs": [input_identity(record) for record in development],
                  "environment": {"device": device, "python": platform.python_version(),
                                  "system": platform.system(), "machine": platform.machine(),
                                  "packages": {name: metadata.version(name) for name in
                                               ("torch", "numpy", "transformers", "tokenizers")}}}
    return comparison, manifest, calibration


@contextmanager
def compute_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "compute.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A revision model pilot is already running in this project") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def write_tensor_case(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_case(saved, comparison, condition, expected):
    if saved.get("completed") is not True:
        raise ValueError("Incomplete convergence case")
    assert_matches(saved["comparison_sha256"], fingerprint(comparison), "case.comparison")
    assert_matches(saved["condition"], condition, "case.condition")
    assert_matches(saved["input"], expected, "case.input")
    checkpoints = comparison["protocol"]["checkpoints"]
    trajectory = saved["trajectory"]
    assert_matches(sorted(int(step) for step in trajectory["snapshots"]), [0] + checkpoints, "case.checkpoints")
    assert_matches([point["step"] for point in trajectory["trace"]], list(range(129)), "case.trace_steps")
    for step, snapshot in trajectory["snapshots"].items():
        logits = snapshot["logits"]
        if logits.ndim != 1 or not torch.isfinite(logits).all():
            raise ValueError("Invalid saved checkpoint logits")
        measured = snapshot["metrics"]
        assert_matches(measured["step"], int(step), "case.snapshot_step")
        assert_matches(measured["prediction"], int(logits.argmax()), "case.prediction")
        assert_matches(measured["target_success"], int(logits.argmax() == expected["clean_target_id"]), "case.success")
        vectors = snapshot["vectors"]
        if len(vectors) != len(saved["cfg"]["norm_budget"]):
            raise ValueError("Wrong number of saved edit vectors")
        for vector, limit, reported in zip(vectors, saved["cfg"]["norm_budget"], measured["edit_norms"]):
            if (not torch.isfinite(vector).all() or float(vector.norm()) > limit * (1 + 1e-5) + 1e-5
                    or not np.isclose(float(vector.norm()), reported, atol=1e-5, rtol=1e-5)):
                raise ValueError("Invalid saved edit-vector norm")


def case_path(output, condition, expected):
    return output / "cases" / condition / f"{expected['sample_id']}.pt"


def load_cases(output, comparison, require_complete=False):
    cases = {}
    for condition in comparison["protocol"]["conditions"]:
        cases[condition] = []
        for expected in comparison["development_inputs"]:
            path = case_path(output, condition, expected)
            if not path.exists():
                if require_complete:
                    raise ValueError(f"Missing convergence case: {condition}/{expected['sample_id']}")
                continue
            saved = torch.load(path, map_location="cpu", weights_only=True)
            validate_case(saved, comparison, condition, expected)
            cases[condition].append(saved)
    return cases


def summarize(cases, comparison):
    count = len(comparison["development_inputs"])
    if set(cases) != set(comparison["protocol"]["conditions"]) or any(len(rows) != count for rows in cases.values()):
        raise ValueError("Pilot summary requires all declared cases")
    result = {"completed": True, "protocol": comparison["protocol"]["version"],
              "scope": "development_diagnostic_only", "comparison_sha256": fingerprint(comparison),
              "n_development_pairs": count, "n_trajectories": sum(len(rows) for rows in cases.values()),
              "conditions": {}, "limitations": comparison["protocol"]["limitations"],
              "reviewer_gap_status": "G1 remains open pending optimizer/basis controls and untouched-test confirmation"}
    for condition, rows in cases.items():
        checkpoint_rows = {}
        for step in comparison["protocol"]["checkpoints"]:
            points = [row["trajectory"]["snapshots"][str(step)]["metrics"] for row in rows]
            flags = [point["target_success"] for point in points]
            mean, interval = rate_ci(flags)
            checkpoint_rows[str(step)] = {"n": count, "successes": sum(flags), "override_rate": mean,
                                          "override_ci": interval,
                                          "mean_target_margin": float(np.mean([point["target_margin"] for point in points])),
                                          "mean_target_cross_entropy": float(np.mean([point["target_cross_entropy"] for point in points]))}
        first_flags = [row["trajectory"]["snapshots"]["32"]["metrics"]["target_success"] for row in rows]
        last_flags = [row["trajectory"]["snapshots"]["128"]["metrics"]["target_success"] for row in rows]
        difference, bounds = paired_delta_ci(first_flags, last_flags)
        result["conditions"][condition] = {
            "checkpoints": checkpoint_rows, "override_128_minus_32": difference, "paired_ci": bounds,
            "mean_margin_128_minus_32": checkpoint_rows["128"]["mean_target_margin"] - checkpoint_rows["32"]["mean_target_margin"],
            "ever_successes": sum(row["trajectory"]["ever_target_success"] for row in rows),
            "ever_success_but_final_failed": sum(row["trajectory"]["ever_target_success"] and not flag for row, flag in zip(rows, last_flags)),
            "guard_shrinks": sum(event["shrunk"] for row in rows for event in row["trajectory"]["guard_events"]),
            "sample_ids": [row["input"]["sample_id"] for row in rows],
            "elapsed_seconds": sum(row["elapsed_seconds"] for row in rows)}
    return result


def markdown(summary):
    lines = ["# Convergence Pilot v1", "",
             "Completed development-only diagnostic on eight Phi transfer pairs. "
             "This is not a test-set result, a reusable editor, or a convergence proof.", "",
             "| Method | Steps | Target overrides | Rate [95% interval] | Mean target margin | Mean target cross-entropy |",
             "|---|---:|---:|---|---:|---:|"]
    notes = []
    for condition, values in summary["conditions"].items():
        for step, row in values["checkpoints"].items():
            lower, upper = row["override_ci"]
            lines.append(f"| {condition} | {step} | {row['successes']}/{row['n']} | "
                         f"{row['override_rate']:.1%} [{lower:.1%}, {upper:.1%}] | "
                         f"{row['mean_target_margin']:.4f} | {row['mean_target_cross_entropy']:.4f} |")
        notes.extend(["", f"**{condition}:** 128-minus-32 override change "
                      f"{values['override_128_minus_32'] * 100:.1f} percentage points; "
                      f"paired interval [{values['paired_ci'][0] * 100:.1f}, {values['paired_ci'][1] * 100:.1f}]. "
                      f"Mean margin change {values['mean_margin_128_minus_32']:.4f}. "
                      f"Ever-successful cases {values['ever_successes']}; earlier success but final failure "
                      f"{values['ever_success_but_final_failed']}; guard shrinks {values['guard_shrinks']}.", ""])
    lines.extend(notes)
    lines.extend(["## Interpretation Limits", "", summary["reviewer_gap_status"], ""])
    lines.extend(f"- {limitation}" for limitation in summary["limitations"])
    lines.extend(["", "Best-seen success is descriptive and may occur before a later guard shrink. "
                  "It is not an automatically chosen safe checkpoint. Full and track8 use the "
                  "original method-specific effective rates. Keep the original 32-step test study unchanged.", ""])
    return "\n".join(lines)


def run(project, protocol_path, device):
    protocol = read_json(protocol_path)
    validate_protocol(protocol)
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    output = project / protocol["output"]
    with compute_lock(project / "results/revision_compute"):
        comparison, manifest, calibration = build_comparison(project, protocol, device)
        run_path = output / "run.json"
        if run_path.exists():
            assert_matches(read_json(run_path)["comparison"], comparison, "resume.comparison")
        else:
            write_new(run_path, {"comparison": comparison, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                                 "scope": "new development experiment; original study is immutable"})
        existing = load_cases(output, comparison)
        expected_count = len(comparison["development_inputs"])
        if not all(len(rows) == expected_count for rows in existing.values()):
            harness = Harness(str(project.parent / protocol["model"]), device)
            harness.model.requires_grad_(False)
            train, development = pilot_records(harness, manifest, protocol)
            layers = comparison["layers"]
            controls, predictions = control_set(harness, CONTROL_PROMPTS)
            budgets, residual_norms = layer_budgets(harness, train, layers, protocol["relative_budget"])
            bases, geometry = basis_sets(harness, train, layers, protocol["training_seed"])
            for condition in protocol["conditions"]:
                cfg = configuration(budgets, 128, effective_rate(harness, bases[condition], calibration["selected"][condition]))
                for index, record in enumerate(development):
                    expected = input_identity(record)
                    path = case_path(output, condition, expected)
                    if path.exists():
                        continue
                    started = time.monotonic()
                    trajectory = optimize_trajectory(harness, record, layers, controls, predictions, bases[condition],
                                                     cfg, protocol["checkpoints"])
                    result = {"completed": True, "comparison_sha256": fingerprint(comparison),
                              "condition": condition, "input": expected, "cfg": cfg,
                              "residual_norms": residual_norms, "basis_diagnostics": geometry,
                              "trajectory": trajectory, "elapsed_seconds": time.monotonic() - started}
                    validate_case(result, comparison, condition, expected)
                    write_tensor_case(path, result)
                    rates = ", ".join(f"s{step}={trajectory['snapshots'][str(step)]['metrics']['target_success']}"
                                      for step in protocol["checkpoints"])
                    print(f"SAVED {condition} development {index + 1}/{expected_count}: {rates}; "
                          f"{result['elapsed_seconds']:.1f}s", flush=True)
                    del trajectory, result
                    gc.collect()
                    if device == "mps":
                        torch.mps.synchronize()
                        torch.mps.empty_cache()
            del harness, bases
        for name, digest in comparison["source_sha256"].items():
            assert_matches(file_sha256(project / "src" / name), digest, "postrun.source")
        result = summarize(load_cases(output, comparison, require_complete=True), comparison)
        if (output / "summary.json").exists():
            assert_matches(read_json(output / "summary.json"), result, "resume.summary")
        else:
            write_new(output / "summary.json", result)
        report = markdown(result)
        if (output / "RESULTS.md").exists():
            assert_matches((output / "RESULTS.md").read_text(), report, "resume.report")
        else:
            with (output / "RESULTS.md").open("x") as stream:
                stream.write(report)
        print(f"Pilot complete: {result['n_trajectories']} trajectories; G1 scientific gap remains open", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", type=Path, default=Path(__file__).resolve().parents[1] / "protocols/convergence_pilot_v1.json")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    args = parser.parse_args()
    run(args.project.resolve(), args.protocol.resolve(), pick_device(args.device))


if __name__ == "__main__":
    main()