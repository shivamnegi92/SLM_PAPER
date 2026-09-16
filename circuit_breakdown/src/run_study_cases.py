"""Durable per-case execution of an already calibrated, frozen study protocol."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from analyze_dissociation import analyze
from experiment_metrics import summarize_predictions
from intervene_margin import CONTROL_PROMPTS
from intervene_pareto import eval_set
from localize import Harness, pick_device
from run_validated_study import (basis_sets, configuration, control_set, effective_rate,
                                 layer_budgets, matched_layers, read_completed, source_fingerprint)
from study_data import fingerprint, load_manifest, manifest_records, write_new


def run(args):
    manifest = load_manifest(args.manifest)
    model_name = Path(args.model).name
    root = Path(args.study) / model_name / args.task
    with (root / "calibration.json").open() as stream:
        calibration = json.load(stream)
    comparison = calibration["comparison"]
    if (comparison["manifest_sha256"] != manifest["sha256"]
            or comparison["source_sha256"] != source_fingerprint()):
        raise ValueError("Frozen source or manifest changed after calibration")
    if not calibration["eligible_for_tracking_claim"]:
        raise ValueError("Task did not pass the development competence gate")
    device = pick_device(args.device)
    harness = Harness(args.model, device)
    harness.model.requires_grad_(False)
    layers = matched_layers(len(harness.layers))
    if layers != comparison["layers"]:
        raise ValueError("Layer configuration changed")
    controls = control_set(harness, CONTROL_PROMPTS)
    batch_count = 0
    for seed in args.seeds:
        splits = manifest_records(harness, manifest, args.task, seed)
        pending = [condition for condition in args.conditions
                   if read_completed(root / f"diss_{condition}_s{seed}.json", comparison) is None]
        if not pending:
            continue
        budgets, residual_norms = layer_budgets(harness, splits["train"], layers, comparison["relative_budget"])
        bases, information = basis_sets(harness, splits["train"], layers, seed)
        for condition in pending:
            cfg = configuration(budgets, comparison["steps"], effective_rate(
                harness, bases[condition], calibration["selected"][condition]))
            case_root = root / "test_cases" / f"{condition}_s{seed}"
            records, seconds, guard_drops = [], [], []
            for index, record in enumerate(splits["test"]):
                path = case_root / f"{record.sample_id}.json"
                saved = read_completed(path, comparison)
                if saved is None:
                    if args.max_new_cases is not None and batch_count >= args.max_new_cases:
                        print("Bounded test batch complete; resume unchanged", flush=True)
                        return
                    started = time.monotonic()
                    metrics = eval_set(harness, [record], layers, bases[condition], cfg, *controls)
                    metrics["records"][0]["seed"] = seed
                    saved = {"completed": True, "comparison": comparison, "condition": condition,
                             "seed": seed, "record": metrics["records"][0], "cfg": cfg,
                             "control_drop": metrics["control_drop"],
                             "elapsed_seconds": time.monotonic() - started}
                    write_new(path, saved)
                    batch_count += 1
                    print(f"{model_name}/{args.task}/{condition}/s{seed}: saved {index + 1}/{len(splits['test'])}", flush=True)
                    del metrics
                    gc.collect()
                    if device == "mps":
                        torch.mps.synchronize()
                        torch.mps.empty_cache()
                if (saved["condition"] != condition or saved["seed"] != seed
                        or saved["record"]["input_hash"] != record.input_hash or saved["cfg"] != cfg):
                    raise ValueError("Saved case differs from frozen condition/input")
                records.append(saved["record"])
                seconds.append(saved["elapsed_seconds"])
                guard_drops.append(saved["control_drop"])
            result = {"completed": True, "protocol_version": comparison["version"],
                      "comparison": comparison, "condition": condition, "seed": seed,
                      "split": {name: len(values) for name, values in splits.items()},
                      "cfg": cfg, "residual_norms": residual_norms, "basis_diagnostics": information,
                      "test": {**summarize_predictions(records), "records": records,
                               "delta_p2way": float(np.mean([row["delta_p2way"] for row in records])),
                               "control_drop": float(np.mean(guard_drops))},
                      "elapsed_seconds": sum(seconds),
                      "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "calibration_sha256": fingerprint(calibration),
                      "executor_sha256": fingerprint(Path(__file__).read_text())}
            write_new(root / f"diss_{condition}_s{seed}.json", result)
            print(f"COMPLETED {condition} s{seed}: steer={result['test']['steer']:.1%}, "
                  f"same-sign damage={result['test']['same_sign_damage']}", flush=True)
        del bases
    if all((root / f"diss_{condition}_s{seed}.json").exists()
           for condition in ("full", "track8", "comp8") for seed in manifest["seeds"]):
        target = root / "paired_summary.json"
        if not target.exists():
            write_new(target, analyze("diss", ["full", "track8", "comp8"], manifest["seeds"], root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "model", "study", "task"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--conditions", nargs="+", choices=["full", "track8", "comp8"],
                        default=["full", "track8", "comp8"])
    parser.add_argument("--max-new-cases", type=int, default=None)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.conditions)) != len(args.conditions):
        parser.error("Seeds and conditions must be distinct")
    run(args)


if __name__ == "__main__":
    main()