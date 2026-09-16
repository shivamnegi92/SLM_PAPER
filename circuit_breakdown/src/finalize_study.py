"""Insert completed artifact-derived results into the manuscript and execution log."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from collect_study import markdown, percentage


def results_prose(ledger):
    if not ledger.get("matrix_complete") or ledger.get("missing"):
        raise ValueError("Cannot finalize an incomplete evidence matrix")
    lines = ["The declared local experiment matrix is complete. This records",
             "execution, not universal control, equivalence, or publication readiness.",
             "The summary is in [the validated results supplement](VALIDATED_RESULTS.md);",
             "per-seed tables, controls, intervals and audit limitations are in",
             "[the detailed results](RESULTS_DETAILS.md).", ""]
    for row in ledger["study"]:
        label = f"{row['model']} / {row['task']}"
        if "result" not in row:
            lines.append(f"**{label}:** the development competence gate was not met "
                         f"(clean {percentage(row['dev']['clean_accuracy'])}, counterfactual "
                         f"{percentage(row['dev']['counterfactual_accuracy'])}). "
                         "Baseline-only test outcomes are retained; no causal-control conclusion is drawn.")
            lines.append("")
            continue
        result = row["result"]
        full, tracking = result["summaries"]["full"], result["summaries"]["track8"]
        contrast = result["comparisons"]["full_minus_track8"]["steer"]
        interval = contrast["ci"]
        interpretation = ("The interval supports a full-space advantage in this tested regime."
                          if interval[0] > 0 else "The interval supports a tracking-subspace advantage in this tested regime."
                          if interval[1] < 0 else "The contrast is unresolved at this precision; this is not evidence of equivalence.")
        lines.append(f"**{label}:** full-space override is {percentage(full['steer'])} "
                     f"and tracking-subspace override is {percentage(tracking['steer'])}, "
                     f"on {full['n']} unique test pairs across {full['n_seeds']} seeds. "
                     f"The paired difference is {contrast['difference'] * 100:.1f} percentage points "
                     f"with 95% interval [{interval[0] * 100:.1f}, {interval[1] * 100:.1f}] percentage points. "
                     f"{interpretation} Same-sign damage is reported separately from negative-edit disruption.")
        lines.append("")
    checks = []
    for run in ledger["capability"]:
        active = run["conditions"]["active_prefix"]
        checks.extend(value["two_point_loss_bound_met"] for value in active["benchmarks"].values())
        checks.append(active["text"]["ten_percent_ratio_bound_met"])
    if checks and all(checks):
        lines.append("The declared capability tolerances were met for the selected edits and measured inputs. "
                     "This remains conditional on those edits, benchmark items and fixed text windows; "
                     "it is not a guarantee for deployment or arbitrary text.")
    else:
        lines.append("The declared capability-preservation criteria were not all established. "
                     "Report the observed costs and uncertainty rather than claiming negligible degradation. "
                     "A failed bound can reflect imprecision or measured damage; inspect the paired intervals.")
    lines.extend(["", "The head results are held-out component and selected-receiver diagnostics, "
                  "not an exhaustive self-repair analysis or a complete causal graph. "
                  "Three checkpoints do not isolate architecture from training and tokenization differences."])
    return "\n".join(lines)


def replace_results(manuscript, results):
    start, end = "<!-- validated-results:start -->", "<!-- validated-results:end -->"
    if manuscript.count(start) != 1 or manuscript.count(end) != 1:
        raise ValueError("Manuscript requires exactly one generated-results region")
    prefix, remaining = manuscript.split(start)
    _, suffix = remaining.split(end)
    return prefix + start + "\n" + results + "\n" + end + suffix


def run(snapshot, paper, plan):
    snapshot, paper, plan = Path(snapshot), Path(paper), Path(plan)
    with (snapshot / "ledger.json").open() as stream:
        ledger = json.load(stream)
    prose = results_prose(ledger)
    manuscript_path = paper / "MANUSCRIPT.md"
    updated = replace_results(manuscript_path.read_text(), prose)
    supplement = paper / "VALIDATED_RESULTS.md"
    if supplement.exists():
        raise FileExistsError(supplement)
    with supplement.open("x") as stream:
        stream.write(markdown(ledger))
    manuscript_path.write_text(updated)
    current = plan.read_text()
    current = current.replace("*(RUNNING)*", "*(DECLARED PROTOCOL EXECUTED; SEE EVIDENCE)*")
    current = current.replace("*(IMPLEMENTED; MODEL RUNS QUEUED)*", "*(DECLARED PROTOCOL EXECUTED; SEE GATES)*")
    current = current.replace("*(QUEUED WITH COMPETENCE GATE)*", "*(EXECUTED WITH COMPETENCE GATE)*")
    current += ("\n## Completed Evidence Snapshot\n\n"
                "The declared experiment matrix passed the missing-artifact gate. "
                "See [paper/VALIDATED_RESULTS.md](paper/VALIDATED_RESULTS.md) and "
                "[paper/MANUSCRIPT.md](paper/MANUSCRIPT.md). "
                "Earlier progress notes describe the execution history. Some task competence "
                "or capability-preservation criteria may be unsupported; those are reported "
                "outcomes, not silently removed cases. No external submission was made.\n")
    plan.write_text(current)
    print(f"Finalized manuscript and evidence supplement from {snapshot}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--paper", required=True)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    run(args.snapshot, args.paper, args.plan)


if __name__ == "__main__":
    main()