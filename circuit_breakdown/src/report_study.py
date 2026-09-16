"""Render detailed paper tables from a checksum-bound audited snapshot."""
from __future__ import annotations

import argparse
from pathlib import Path

from audit_study import file_sha256, read_json
from experiment_metrics import rate_ci


def estimate(value, bounds, scale=100, unit="%"):
    if value is None or bounds is None:
        return "undefined"
    digits = 1 if scale == 100 else 4
    return (f"{value * scale:.{digits}f}{unit} "
            f"[{bounds[0] * scale:.{digits}f}, {bounds[1] * scale:.{digits}f}]")


def point_change(value, bounds):
    return estimate(value, bounds, unit=" pp")


def add_table(lines, title, headers, rows):
    lines.extend([f"## {title}", "", "| " + " | ".join(headers) + " |",
                  "|" + "|".join("---" for header in headers) + "|"])
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    lines.append("")


def require_audit(snapshot, audit):
    if (audit.get("passed") is not True or audit.get("snapshot_matches_sources") is not True
            or audit.get("snapshot_sha256") != file_sha256(snapshot)):
        raise ValueError("Audit does not match this evidence snapshot")


def detailed_markdown(ledger):
    if ledger.get("matrix_complete") is not True or ledger.get("missing") != []:
        raise ValueError("Cannot report an incomplete study")
    lines = ["# Detailed Validated Results", "",
             "Generated from the completed frozen study snapshot. Rates include 95% intervals; "
             "accuracy differences are percentage points (pp), not relative percent changes.", "",
             "The CPU-only post-run audit recomputes stored predictions, counts and statistics. "
             "It does not rerun model forward passes, reconstruct missing logits, or prove historical "
             "weight provenance. See [the audit and reproduction guide](../docs/REPRODUCIBILITY.md).", "",
             "The two eligible tasks each contain 150 semantic test pairs shared across models and "
             "methods, in three disjoint 50-pair seed blocks. Repeated method/model evaluations are "
             "not extra independent semantic examples. Full-space edits are optimized separately for "
             "each target-informed example; this is not a reusable editor or demonstrated error repair.", "",
             "![Target override and same-sign damage](../results/final_validated_evidence_v1/validated_control.png)", ""]
    competence, baseline_rows, pooled, per_seed, additional, contrasts = [], [], [], [], [], []
    for row in ledger["study"]:
        model, task, dev = row["model"], row["task"], row["dev"]
        competence.append([model, task, dev["n"], estimate(dev["clean_accuracy"], dev["clean_ci"]),
                           estimate(dev["counterfactual_accuracy"], dev["counterfactual_ci"]),
                           "passed" if row["eligible"] else "failed; baseline only"])
        if "result" not in row:
            for run in row["baseline_test"]:
                baseline = run["baseline_test"]
                baseline_rows.append([model, task, run["seed"], baseline["n"],
                                      estimate(baseline["clean_accuracy"], baseline["clean_ci"]),
                                      estimate(baseline["counterfactual_accuracy"], baseline["counterfactual_ci"])])
            continue
        for method, summary in row["result"]["summaries"].items():
            pooled.append([model, task, method, summary["n"],
                           estimate(summary["steer"], summary["steer_ci"]),
                           summary["n_baseline_clean_correct"],
                           estimate(summary["same_sign_damage"], summary["same_sign_damage_ci"]),
                           estimate(summary["negative_edit_disruption"], summary["negative_edit_disruption_ci"])])
            additional.append([model, task, method, summary["n_new_target_eligible"],
                               estimate(summary["new_target_rate"], summary["new_target_rate_ci"]),
                               estimate(summary["third_token_rate"], summary["third_token_rate_ci"]),
                               f"{summary['delta_p2way']:.4f}"])
            for seed, values in summary["per_seed"].items():
                source = f"../results/validated_v1/{model}/{task}/diss_{method}_s{seed}.json"
                per_seed.append([model, task, method, f"[{seed}]({source})", values["n"],
                                 estimate(values["steer"], values["steer_ci"]),
                                 values["n_baseline_clean_correct"],
                                 estimate(values["same_sign_damage"], values["same_sign_damage_ci"]),
                                 estimate(values["negative_edit_disruption"], values["negative_edit_disruption_ci"])])
        for name, metrics in row["result"]["comparisons"].items():
            for metric, value in metrics.items():
                continuous = metric == "delta_p2way"
                contrast = (estimate(value["difference"], value["ci"], scale=1, unit="") if continuous
                            else point_change(value["difference"], value["ci"]))
                probability = "undefined" if value["p"] is None else f"{value['p']:.4g}"
                contrasts.append([model, task, name, metric, value["n_unique"], contrast, probability])
    add_table(lines, "Development Competence", ["Model", "Task", "Seed-0 dev pairs", "Clean accuracy", "Counterfactual accuracy", "Gate"], competence)
    lines.extend(["The 80% gate is applied to both accuracies on seed-0 development data. "
                  "Learning-rate selection uses four seed-0 development examples with equal candidate grids. "
                  "The other development blocks are not additional fitted calibration replicates.", ""])
    add_table(lines, "Hard-Task Baseline Tests", ["Model", "Task", "Seed", "Pairs", "Clean accuracy", "Counterfactual accuracy"], baseline_rows)
    lines.extend(["All container-swap development gates failed. These test rows describe the failed "
                  "generalization check and cannot support a causal-control conclusion. Baseline-only "
                  "files store flags but not item IDs; the audit checks their counts and rates, not "
                  "the original item ordering independently.", ""])
    add_table(lines, "Pooled Interventions", ["Model", "Task", "Method", "Pairs", "Target override", "Baseline-correct n", "Same-sign damage", "Negative disruption"], pooled)
    lines.extend(["Damage and disruption use only baseline-correct clean inputs. Target override uses "
                  "all counterfactual inputs. A zero observed damage rate is not a guarantee for unrelated inputs.", ""])
    add_table(lines, "Per-Seed Interventions", ["Model", "Task", "Method", "Seed / data", "Pairs", "Target override", "Baseline-correct n", "Same-sign damage", "Negative disruption"], per_seed)
    add_table(lines, "Additional Intervention Metrics", ["Model", "Task", "Method", "New-target eligible n", "New target rate", "Third-token rate", "Mean two-way probability change"], additional)
    add_table(lines, "Paired Contrasts", ["Model", "Task", "First minus second", "Metric", "Eligible pairs", "Difference [95% interval]", "p"], contrasts)
    lines.extend(["Binary contrasts use paired gain/loss Wilson bounds with a within-contrast Bonferroni "
                  "adjustment and exact McNemar tests. Continuous contrasts use the saved seed-stratified "
                  "paired bootstrap and sign-permutation procedure. These exploratory contrasts do not "
                  "have a study-wide multiplicity correction. Intervals are conditional on the observed "
                  "seed blocks; equal rates or p=1 do not establish equivalence.", ""])
    benchmark_rows, text_rows, norm_rows = [], [], []
    for run in ledger["capability"]:
        model, seed = run["comparison"]["model"], run["seed"]
        for condition, result in run["conditions"].items():
            for benchmark, values in result["benchmarks"].items():
                baseline = run["baseline_benchmarks"][benchmark]
                baseline_mean, baseline_ci = rate_ci([item["correct"] for item in baseline])
                benchmark_rows.append([model, seed, condition, benchmark, values["n_unique_items"],
                                       estimate(baseline_mean, baseline_ci),
                                       estimate(values["accuracy"], values["accuracy_ci"]),
                                       point_change(values["delta"], values["delta_ci"]),
                                       "met" if values["two_point_loss_bound_met"] else "not established"])
            text = result["text"]
            text_rows.append([model, seed, condition,
                              estimate(text["perplexity_ratio"], text["ratio_ci"], scale=1, unit=""),
                              "met" if text["ten_percent_ratio_bound_met"] else "not established",
                              ", ".join(f"{value:.5f}" for value in text["per_window_log_changes"])])
            norm_rows.append([model, seed, condition, ", ".join(str(layer) for layer in run["layers"]),
                              ", ".join(f"{value:.4f}" for value in result["edit_norms"]),
                              ", ".join(f"{value:.4f}" for value in run["cfg"]["norm_budget"])])
    add_table(lines, "Capability Accuracy and Controls", ["Model", "Edit seed", "Exposure", "Benchmark", "Items", "Baseline", "Edited", "Accuracy change [95% interval]", "2 pp loss bound"], benchmark_rows)
    lines.extend(["Each edit reuses the same 240 HellaSwag items and 200 ARC-Easy items. "
                  "The rows are not independent datasets and their confidence endpoints are not averaged. "
                  "The accuracy criterion requires the paired lower bound to be strictly above -2 pp. "
                  "Failure to meet that bound may reflect uncertainty, damage, or both. Global exposure "
                  "is a stress test, not an asserted upper bound on other deployment policies.", ""])
    add_table(lines, "Text Exposure", ["Model", "Edit seed", "Exposure", "Perplexity ratio [95% interval]", "10% ratio bound", "Window 1-5 mean NLL changes"], text_rows)
    lines.extend(["The ratio is the exponentiated mean of five paired window-level changes in mean token "
                  "negative log likelihood (NLL). The 95% interval bootstraps those five fixed windows; "
                  "it is not population-wide text uncertainty. All windows come from one public text. "
                  "The declared ratio upper bound must be strictly below 1.10. A result of 1 means "
                  "unchanged measured perplexity. Baseline and zero-control token arrays remain in the JSON.", ""])
    add_table(lines, "Capability Edit Norms", ["Model", "Seed", "Exposure", "Layers", "Recorded norms", "Per-layer budgets"], norm_rows)
    head_rows = []
    for run in ledger["heads"]:
        for label, field, count in (("Top8 minus depth-matched random faithfulness", "faithfulness_difference", run["n_valid_logit_gaps"]),
                                    ("Selected minus random receiver compensation", "compensation_vs_random", run["n_test"])):
            comparison = run[field]
            head_rows.append([run["model"], label, count,
                              estimate(comparison["difference"], comparison["ci"], scale=1, unit="") if comparison else "undefined",
                              f"{comparison['p']:.4g}" if comparison else "undefined"])
    add_table(lines, "Head Diagnostics With Uncertainty", ["Model", "Diagnostic", "Valid paired examples", "Difference [95% interval]", "p"], head_rows)
    lines.extend(["Head discovery uses 12 seed-0 training examples. Verification uses 24 held-out examples "
                  "from each seed, with five depth-matched random top8 sets. Faithfulness is normalized "
                  "logit-gap recovery, not question-answering accuracy. Compensation is a logit-margin "
                  "difference for a selected source/receiver intervention, not a full circuit or exhaustive "
                  "self-repair analysis. The recorded projection check uses natural replacement norms; "
                  "it is not the same intervention as budgeted additive steering.", "",
                  "## Evidence Limits", "",
                  "The historical model fingerprint used configuration and root weight-file sizes, "
                  "not hashes of weight contents. Head artifacts also lack run-time source/weight hashes. "
                  "Post-run checksums identify the files available now, not a retrospectively verified "
                  "remote revision. Continuous intervention deltas and head faithfulness cannot be "
                  "reconstructed from missing original full logits. No claim of optimizer convergence, "
                  "universal steering impossibility, general capability preservation, or publication "
                  "readiness follows from completion of the consistency audit.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    require_audit(args.snapshot / "ledger.json", read_json(args.audit))
    result = detailed_markdown(read_json(args.snapshot / "ledger.json"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(result)
    print(f"Saved detailed tables -> {args.output}")


if __name__ == "__main__":
    main()