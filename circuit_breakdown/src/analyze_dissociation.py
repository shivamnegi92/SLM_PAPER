"""Strict paired analysis of completed, identity-bearing intervention runs.

Legacy arrays are accepted only with --legacy-descriptive, without significance
claims. Output files are distinct from historical summaries and never overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiment_metrics import align_records, paired_comparison, summarize_predictions


def load_group(prefix, condition, seeds, outdir):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Specify distinct, nonempty seeds")
    records, files, comparison = [], [], None
    for seed in seeds:
        path = Path(outdir) / f"{prefix}_{condition}_s{seed}.json"
        with path.open() as stream:
            run = json.load(stream)
        if not run.get("completed") or not run.get("protocol_version"):
            raise ValueError(f"Unverified or incomplete run: {path}")
        if run.get("seed") != seed or run.get("condition") != condition:
            raise ValueError(f"Run identity disagrees with filename: {path}")
        configuration = run.get("comparison")
        if not isinstance(configuration, dict) or not configuration:
            raise ValueError(f"Missing comparison configuration: {path}")
        if comparison is not None and configuration != comparison:
            raise ValueError(f"Incompatible comparison configuration: {path}")
        comparison = configuration
        samples = run["test"].get("records")
        if not samples or len(samples) != run["split"]["test"]:
            raise ValueError(f"Missing or truncated sample records: {path}")
        required = ("sample_id", "input_hash", "seed", "clean_target_id", "corrupt_target_id",
                    "baseline_clean_prediction", "baseline_corrupt_prediction", "steer", "delta_p2way")
        if any(any(key not in record for key in required) for record in samples):
            raise ValueError(f"Missing paired identity/baseline fields: {path}")
        if any(not record["sample_id"] or not record["input_hash"] for record in samples):
            raise ValueError(f"Empty paired sample identity: {path}")
        if any(record.get("seed") != seed for record in samples):
            raise ValueError(f"Sample seed metadata mismatch: {path}")
        records.extend(samples)
        files.append(str(path))
    align_records(records, records)
    return {"records": records, "files": files, "comparison": comparison}


def compare_groups(first, second, metric="steer", binary=True):
    if first["comparison"] != second["comparison"]:
        raise ValueError("Cannot compare incompatible model/task/layer/optimizer protocols")
    left, right = align_records(first["records"], second["records"])
    for one, two in zip(left, right):
        if any(one[key] != two[key] for key in ("baseline_clean_prediction", "baseline_corrupt_prediction")):
            raise ValueError("Unedited predictions differ between paired methods")
    if metric in ("same_sign_damage", "negative_edit_disruption"):
        if any(one["baseline_clean_correct"] != two["baseline_clean_correct"]
               for one, two in zip(left, right)):
            raise ValueError("Paired baseline correctness differs")
        left = [record for record in left if record["baseline_clean_correct"]]
        right = [record for record in right if record["baseline_clean_correct"]]
        if not left:
            return {"metric": metric, "n_unique": 0, "difference": None, "ci": None, "p": None}
    return paired_comparison(left, right, metric, binary=binary)


def analyze(prefix, conditions, seeds, outdir):
    if len(set(conditions)) != len(conditions) or not conditions:
        raise ValueError("Conditions must be distinct and nonempty")
    groups = {condition: load_group(prefix, condition, seeds, outdir) for condition in conditions}
    anchor = groups[conditions[0]]
    summaries = {}
    for condition, group in groups.items():
        compare_groups(anchor, group)
        summaries[condition] = {
            **summarize_predictions(group["records"]),
            "n_seeds": len(seeds), "source_files": group["files"],
            "delta_p2way": float(np.mean([record["delta_p2way"] for record in group["records"]])),
            "per_seed": {str(seed): summarize_predictions(
                [record for record in group["records"] if record["seed"] == seed]) for seed in seeds},
        }
    comparisons = {}
    for condition in conditions[1:]:
        comparisons[f"{conditions[0]}_minus_{condition}"] = {
            metric: compare_groups(anchor, groups[condition], metric, binary=metric != "delta_p2way")
            for metric in ("steer", "same_sign_damage", "negative_edit_disruption", "delta_p2way")
        }
    if "track8" in groups and "comp8" in groups:
        comparisons["track8_minus_comp8"] = {
            "delta_p2way": compare_groups(groups["track8"], groups["comp8"], "delta_p2way", binary=False)
        }
    return {"analysis_version": "paired_ids_v1", "comparison": anchor["comparison"],
            "summaries": summaries, "comparisons": comparisons,
            "limitations": ["Intervals conditional on the observed seeds.",
                            "Equal rates or p=1 are not population equivalence.",
                            "Exploratory contrasts require multiplicity-aware interpretation."]}


def legacy_descriptive(prefix, conditions, seeds, outdir):
    summaries = {}
    for condition in conditions:
        runs = []
        for seed in seeds:
            path = Path(outdir) / f"{prefix}_{condition}_s{seed}.json"
            with path.open() as stream:
                saved = json.load(stream)
            flags = saved["test"]["steer_flags"]
            if len(flags) != saved["split"]["test"]:
                raise ValueError(f"Truncated legacy flags: {path}")
            runs.append({"file": str(path), "seed": seed, "n_evaluations": len(flags),
                         "steer": float(np.mean(flags)),
                         "negative_edit_error": saved["test"]["break"],
                         "delta_p2way": saved["test"].get("delta_p2way")})
        summaries[condition] = runs
    return {"analysis_version": "legacy_descriptive_only", "summaries": summaries,
            "limitations": ["No sample IDs: independence and pairing unverified; no inferential p-values."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="diss")
    parser.add_argument("--conditions", nargs="+", default=["full", "track8", "comp8"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--output", required=True)
    parser.add_argument("--legacy-descriptive", action="store_true")
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    function = legacy_descriptive if args.legacy_descriptive else analyze
    result = function(args.prefix, args.conditions, args.seeds, args.outdir)
    payload = json.dumps(result, indent=2, allow_nan=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        stream.write(payload)
    print(f"saved {result['analysis_version']} -> {target}")


if __name__ == "__main__":
    main()