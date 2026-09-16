"""Build an evidence ledger and figures only from saved validated artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from study_data import write_new


MODELS = ["llama-3.2-3b", "phi-3.5-mini", "nemotron-mini-4b"]
TASKS = ["intermediate", "transfer", "container_swap"]


def read(path):
    with path.open() as stream:
        return json.load(stream)


def collect(study, capability, heads):
    ledger = {"study": [], "capability": [], "heads": [], "missing": []}
    for model in MODELS:
        for task in TASKS:
            root = Path(study) / model / task
            calibration = root / "calibration.json"
            if not calibration.exists():
                ledger["missing"].append(str(calibration))
                continue
            saved = read(calibration)
            if not saved.get("completed"):
                raise ValueError(f"Incomplete calibration: {calibration}")
            row = {"model": model, "task": task, "dev": saved["baseline_dev"],
                   "eligible": saved["eligible_for_tracking_claim"], "source": str(calibration)}
            if row["eligible"]:
                summary = root / "paired_summary.json"
                if summary.exists():
                    row["result"] = read(summary)
                    row["source"] = str(summary)
                else:
                    ledger["missing"].append(str(summary))
            else:
                row["baseline_test"] = []
                for seed in [0, 1, 2]:
                    baseline = root / f"baseline_s{seed}.json"
                    if baseline.exists():
                        row["baseline_test"].append(read(baseline))
                    else:
                        ledger["missing"].append(str(baseline))
            ledger["study"].append(row)
        for seed in [0, 1, 2]:
            path = Path(capability) / model / f"capability_s{seed}.json"
            if path.exists():
                ledger["capability"].append({"source": str(path), **read(path)})
            else:
                ledger["missing"].append(str(path))
        path = Path(heads) / f"heads_{model}_intermediate.json"
        if path.exists():
            ledger["heads"].append({"source": str(path), **read(path)})
        else:
            ledger["missing"].append(str(path))
    ledger["matrix_complete"] = not ledger["missing"]
    return ledger


def percentage(value):
    return "undefined" if value is None else f"{value * 100:.1f}%"


def markdown(ledger):
    lines = ["# Validated Evidence Ledger", "",
             "Status: " + ("declared matrix complete" if ledger["matrix_complete"] else "INCOMPLETE; missing artifacts listed below"),
             "", "Generated from saved artifacts, not inferred from log messages or planned runs.",
             "", "## Task Competence", "",
             "| Model | Task | Dev clean | Dev counterfactual | Causal-control gate |",
             "|---|---|---:|---:|---|"]
    for row in ledger["study"]:
        lines.append(f"| {row['model']} | {row['task']} | {percentage(row['dev']['clean_accuracy'])} | "
                     f"{percentage(row['dev']['counterfactual_accuracy'])} | {'passed' if row['eligible'] else 'failed; baseline-only'} |")
    lines.extend(["", "## Locked Intervention Results", "",
                  "| Model | Task | Condition | Unique test pairs | Target override | Same-sign damage | Negative disruption |",
                  "|---|---|---|---:|---:|---:|---:|"])
    for row in ledger["study"]:
        if "result" not in row:
            continue
        for condition, summary in row["result"]["summaries"].items():
            lines.append(f"| {row['model']} | {row['task']} | {condition} | {summary['n']} | "
                         f"{percentage(summary['steer'])} | {percentage(summary['same_sign_damage'])} | "
                         f"{percentage(summary['negative_edit_disruption'])} |")
    lines.extend(["", "Damage and disruption are conditioned on baseline-correct clean answers; denominators and intervals are in the JSON ledger.",
                  "", "## Capability Exposure", "",
                  "| Model | Edit seed | Benchmark | Active-prefix accuracy change (pp) | Paired interval (pp) |",
                  "|---|---:|---|---:|---|"])
    for run in ledger["capability"]:
        for benchmark, result in run["conditions"]["active_prefix"]["benchmarks"].items():
            interval = result["delta_ci"]
            lines.append(f"| {run['comparison']['model']} | {run['seed']} | {benchmark} | "
                         f"{result['delta'] * 100:.1f} | [{interval[0] * 100:.1f}, {interval[1] * 100:.1f}] |")
    lines.extend(["", "Accuracy changes and interval endpoints are percentage points (pp), not relative percent changes.",
                  "Each edit reuses the same benchmark items; rows are not independent datasets. Global exposure and per-window text losses remain separate in source artifacts.",
                  "", "## Mechanism Diagnostics", ""])
    for run in ledger["heads"]:
        if "faithfulness_difference" not in run:
            lines.append(f"- {run.get('task')}: competence gate not met; no mechanism claim.")
            continue
        faith = run["faithfulness_difference"]
        comp = run["compensation_vs_random"]
        lines.append(f"- {run['model']}: held-out top8-minus-random faithfulness {faith['difference']:.4f}; "
                     f"selected receiver compensation minus random {comp['difference']:.4f}. "
                     "This is a bounded diagnostic, not a complete circuit.")
    lines.extend(["", "## Missing Artifacts", ""])
    lines.extend(f"- `{path}`" for path in ledger["missing"])
    if not ledger["missing"]:
        lines.append("None in the declared matrix. This does not mean every scientific hypothesis was supported.")
    return "\n".join(lines) + "\n"


def figures(ledger, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [row for row in ledger["study"] if "result" in row]
    if not rows:
        return
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"full": "#1c7c54", "track8": "#c7522a", "comp8": "#4665a3"}
    for index, row in enumerate(rows):
        for condition, summary in row["result"]["summaries"].items():
            offset = {"full": -.22, "track8": 0, "comp8": .22}[condition]
            lower, upper = summary["steer_ci"]
            error = [[max(0, summary["steer"] - lower)], [max(0, upper - summary["steer"])]]
            axes[0].bar(index + offset, summary["steer"], width=.21, color=colors[condition],
                        yerr=error, capsize=2,
                        label=condition if index == 0 else None)
            if summary["same_sign_damage"] is not None:
                damage = summary["same_sign_damage"]
                damage_ci = summary["same_sign_damage_ci"]
                axes[1].errorbar(damage, summary["steer"], color=colors[condition],
                                 marker=["o", "s", "^"][MODELS.index(row["model"])],
                                 xerr=[[max(0, damage - damage_ci[0])], [max(0, damage_ci[1] - damage)]],
                                 yerr=error, linestyle="none", alpha=.7, capsize=2)
    axes[0].set_xticks(range(len(rows)), [f"{row['model']}\n{row['task']}" for row in rows],
                      rotation=35, ha="right", fontsize=7)
    axes[0].set_ylabel("Target override rate")
    axes[0].legend()
    axes[1].set_xlabel("Same-sign damage among baseline-correct inputs")
    axes[1].set_ylabel("Target override rate")
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[Line2D([], [], color="black", marker=marker, linestyle="none", label=model)
                            for model, marker in zip(MODELS, ["o", "s", "^"])], fontsize=7, loc="best")
    for axis in axes:
        axis.set_ylim(-.02, 1.02)
    figure.tight_layout()
    figure.savefig(outdir / "validated_control.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("study", "capability", "heads", "output-dir"):
        parser.add_argument(f"--{argument}", required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    ledger = collect(args.study, args.capability, args.heads)
    if args.require_complete and not ledger["matrix_complete"]:
        raise ValueError(f"Missing {len(ledger['missing'])} declared artifacts")
    root = Path(args.output_dir)
    if root.exists():
        raise FileExistsError(f"Choose a fresh evidence snapshot directory: {root}")
    write_new(root / "ledger.json", ledger)
    with (root / "RESULTS.md").open("x") as stream:
        stream.write(markdown(ledger))
    figures(ledger, root)
    print(f"saved evidence snapshot -> {root}; complete={ledger['matrix_complete']}")


if __name__ == "__main__":
    main()