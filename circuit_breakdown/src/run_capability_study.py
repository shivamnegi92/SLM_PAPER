"""Corrected multi-benchmark capability exposure with preselected edits/windows."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time

import numpy as np
import torch

from capability import continuation_nll, ending_logprob, load_hellaswag
from capability_deployed import hooks_at
from experiment_metrics import paired_delta_ci, rate_ci
from intervene_margin import CONTROL_PROMPTS
from intervene_pareto import optimize_sample
from localize import Harness, pick_device
from public_benchmarks import load_arc, text_windows
from run_validated_study import configuration, control_set, layer_budgets, matched_layers, model_fingerprint, source_fingerprint
from study_data import fingerprint, load_manifest, manifest_records, write_new


def exposure(harness, layers, vectors, mode):
    @contextmanager
    def scope(position):
        handles = hooks_at(harness, layers, vectors, mode, position=position)
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
    return scope


def accuracy_outcomes(harness, items, intervention=None):
    outcomes = []
    for index, (context, endings, label) in enumerate(items):
        scores = [ending_logprob(harness, context, ending, intervention) for ending in endings]
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite multiple-choice scores")
        outcomes.append({"id": fingerprint([context, endings, label]), "correct": int(np.argmax(scores) == label),
                         "prediction": int(np.argmax(scores)), "target": label, "scores": scores})
        if (index + 1) % 40 == 0:
            print(f"  scored {index + 1}/{len(items)} questions", flush=True)
            if harness.device == "mps":
                torch.mps.empty_cache()
    return outcomes


def paired_accuracy(baseline, edited):
    if [item["id"] for item in baseline] != [item["id"] for item in edited]:
        raise ValueError("Benchmark item pairing mismatch")
    if len({item["id"] for item in baseline}) != len(baseline):
        raise ValueError("Duplicated benchmark item")
    base_flags = [item["correct"] for item in baseline]
    flags = [item["correct"] for item in edited]
    accuracy, interval = rate_ci(flags)
    delta, delta_ci = paired_delta_ci(base_flags, flags)
    return {"accuracy": accuracy, "accuracy_ci": interval, "delta": delta,
            "delta_ci": delta_ci, "n_unique_items": len(flags), "items": edited,
            "two_point_loss_bound_met": delta_ci[0] > -0.02,
            "interval_method": "approx95 paired Bonferroni gain/loss Wilson"}


def window_summary(baseline, edited):
    if len(baseline) != len(edited) or not baseline:
        raise ValueError("Window pairing mismatch")
    differences = []
    for base, changed in zip(baseline, edited):
        if len(base) != len(changed):
            raise ValueError("Token spans differ between paired text windows")
        differences.append(float(np.mean(changed) - np.mean(base)))
    rng = np.random.default_rng(619)
    samples = np.asarray(differences)
    bootstrap = samples[rng.integers(0, len(samples), size=(9999, len(samples)))].mean(axis=1)
    interval = np.exp(np.quantile(bootstrap, [.025, .975])).tolist()
    return {"mean_log_perplexity_change": float(samples.mean()),
            "perplexity_ratio": float(np.exp(samples.mean())), "ratio_ci": interval,
            "per_window_log_changes": differences,
            "ten_percent_ratio_bound_met": interval[1] < 1.10,
            "scope": "paired window bootstrap; conditional on five fixed contiguous-source windows"}


def run(args):
    manifest = load_manifest(args.manifest)
    model_name = Path(args.model).name
    output = Path(args.outdir) / model_name
    benchmarks = {"hellaswag": load_hellaswag(args.hellaswag, args.n_hs), "arc_easy": load_arc(args.arc)}
    windows = text_windows(Path(args.text).read_text(), 5, 2000)
    comparison = {"protocol": "capability_benchmarks_v1", "model": model_name,
                  "model_fingerprint": model_fingerprint(args.model), "manifest": manifest["sha256"],
                  "core_source_sha256": source_fingerprint(),
                  "scorer_sha256": fingerprint({name: (Path(__file__).parent / name).read_text()
                                                for name in ("run_capability_study.py", "capability.py", "capability_deployed.py")}),
                  "items": {name: fingerprint(items) for name, items in benchmarks.items()},
                  "windows": [fingerprint(window) for window in windows],
                  "prefix_tokens": 32, "max_tokens": 512, "steps": 32, "relative_budget": .3}
    pending = []
    for seed in args.seeds:
        path = output / f"capability_s{seed}.json"
        if path.exists():
            with path.open() as stream:
                saved = json.load(stream)
            if not saved.get("completed") or saved["comparison"] != comparison:
                raise ValueError(f"Incompatible existing capability result: {path}")
        else:
            pending.append(seed)
    if not pending:
        return
    harness = Harness(args.model, pick_device(args.device))
    harness.model.requires_grad_(False)
    layers = matched_layers(len(harness.layers))
    guard = control_set(harness, CONTROL_PROMPTS)
    baseline = {name: accuracy_outcomes(harness, items) for name, items in benchmarks.items()}
    baseline_nll = [continuation_nll(harness, window, max_tokens=512, prefix_tokens=32).tolist()
                    for window in windows]
    calibration_path = Path(args.study) / model_name / "intermediate" / "calibration.json"
    with calibration_path.open() as stream:
        calibration = json.load(stream)
    if calibration["comparison"]["manifest_sha256"] != manifest["sha256"]:
        raise ValueError("Calibration and capability manifests differ")
    rate = calibration.get("selected", {}).get("full", .05)
    for seed in pending:
        started = time.monotonic()
        splits = manifest_records(harness, manifest, "intermediate", seed)
        record = splits["test"][0]
        budgets, _ = layer_budgets(harness, splits["train"], layers, .3)
        cfg = configuration(budgets, 32, rate)
        edit_path = output / "edits" / f"edit_s{seed}.pt"
        if edit_path.exists():
            saved_edit = torch.load(edit_path, map_location="cpu", weights_only=True)
            if (saved_edit["comparison"] != comparison or saved_edit["sample_id"] != record.sample_id
                    or saved_edit["cfg"] != cfg):
                raise ValueError("Saved capability edit differs from the frozen study")
            vectors = [vector.to(harness.device) for vector in saved_edit["vectors"]]
            trace = saved_edit["trace"]
        else:
            edit = optimize_sample(harness, record, layers, *guard, None, cfg)
            vectors, trace = edit.detached(), edit.trace
            edit_path.parent.mkdir(parents=True, exist_ok=True)
            with edit_path.open("xb") as stream:
                torch.save({"comparison": comparison, "sample_id": record.sample_id, "cfg": cfg,
                            "vectors": [vector.cpu() for vector in vectors], "trace": trace}, stream)
        generator = torch.Generator().manual_seed(49000 + seed)
        random = [torch.randn(vector.shape, generator=generator).to(vector.device) for vector in vectors]
        random = [control / (control.norm() + 1e-9) * vector.norm() for control, vector in zip(random, vectors)]
        conditions = {"active_prefix": (vectors, "prefix"), "random_prefix": (random, "prefix"),
                      "zero_prefix": ([torch.zeros_like(vector) for vector in vectors], "prefix"),
                      "global": (vectors, "global")}
        results = {}
        for condition, (edits, mode) in conditions.items():
            condition_path = output / "condition_cases" / f"{condition}_s{seed}.json"
            if condition_path.exists():
                with condition_path.open() as stream:
                    saved_condition = json.load(stream)
                if saved_condition["comparison"] != comparison:
                    raise ValueError("Capability checkpoint protocol mismatch")
                results[condition] = saved_condition["result"]
                continue
            intervention = exposure(harness, layers, edits, mode)
            scores = {}
            for name, items in benchmarks.items():
                print(f"{model_name} seed{seed} {condition} {name}", flush=True)
                checkpoint = output / "benchmark_cases" / f"{condition}_{name}_s{seed}.json"
                if checkpoint.exists():
                    with checkpoint.open() as stream:
                        saved_benchmark = json.load(stream)
                    if saved_benchmark["comparison"] != comparison:
                        raise ValueError("Benchmark checkpoint protocol mismatch")
                    scored = saved_benchmark["items"]
                else:
                    scored = accuracy_outcomes(harness, items, intervention)
                    write_new(checkpoint, {"comparison": comparison, "items": scored})
                scores[name] = paired_accuracy(baseline[name], scored)
                if condition == "zero_prefix":
                    for base, changed in zip(baseline[name], scored):
                        if not np.allclose(base["scores"], changed["scores"], atol=1e-5, rtol=1e-5):
                            raise ValueError("Zero edit changed multiple-choice log probabilities")
            losses = [continuation_nll(harness, window, max_tokens=512, prefix_tokens=32,
                                        intervention=intervention).tolist() for window in windows]
            text_result = window_summary(baseline_nll, losses)
            if condition == "zero_prefix" and not np.allclose(text_result["per_window_log_changes"], 0, atol=1e-5):
                raise ValueError("Zero edit changed text losses")
            results[condition] = {"benchmarks": scores, "text": text_result, "token_nll": losses,
                                  "edit_norms": [float(vector.norm()) for vector in edits]}
            write_new(condition_path, {"comparison": comparison, "result": results[condition]})
        result = {"completed": True, "comparison": comparison, "seed": seed, "layers": layers,
                  "edit_sample_id": record.sample_id, "input_hash": record.input_hash,
                  "baseline_benchmarks": baseline, "baseline_token_nll": baseline_nll,
                  "cfg": cfg, "conditions": results, "optimization_trace": trace,
                  "elapsed_seconds": time.monotonic() - started,
                  "scope": "active exposure on unrelated inputs; not an entity-trigger deployment; "
                           "three fixed edits evaluated separately on reused benchmark items"}
        write_new(output / f"capability_s{seed}.json", result)
        print(f"SAVED capability {model_name} seed{seed}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "model", "study", "outdir", "hellaswag", "arc", "text"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--n-hs", type=int, default=240)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    run(parser.parse_args())


if __name__ == "__main__":
    main()