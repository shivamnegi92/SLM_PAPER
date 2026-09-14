"""Figure: graded, specific answer control alongside zero semantic fidelity.

Left panel  -- P(Y_donor) as a function of intervention magnitude alpha, for
               the real donor delta versus a norm-matched random direction.
               The real donor rises smoothly; the norm-matched control stays
               flat at zero. Establishes SPECIFIC, GRADED answer control.
Right panel -- completeness versus selectivity from the cross-question panel,
               with Wilson intervals. Selectivity is zero.

Placing these side by side is the paper's core argument in one image: the
intervention earns every behavioral credential on the left and fails semantic
fidelity on the right.

Reads the JSON emitted by the magnitude sweep and the panel runs rather than
recomputing, so the figure can never silently disagree with the numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    with Path(path).open() as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--magnitude", type=Path, required=True,
                        help="JSON from the magnitude sweep")
    parser.add_argument("--panel", type=Path, required=True,
                        help="JSON with completeness/selectivity per model")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    magnitude = load(args.magnitude)
    panel = load(args.panel)

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    alphas = magnitude["alphas"]
    left.plot(alphas, magnitude["p_donor_real"], marker="o", linewidth=2,
              label="real donor direction", color="#1f77b4")
    left.plot(alphas, magnitude["p_donor_norm_matched"], marker="s", linewidth=2,
              linestyle="--", label="norm-matched random", color="#d62728")
    left.set_xlabel(r"intervention magnitude $\alpha$")
    left.set_ylabel(r"$P(Y_{\mathrm{donor}})$")
    left.set_title("Graded, direction-specific answer control")
    left.set_ylim(-0.03, 1.0)
    left.grid(alpha=0.3)
    left.legend(loc="upper left", fontsize=9)

    labels, values, errors, colors = [], [], [], []
    for model in panel["models"]:
        for metric, color in (("completeness", "#2ca02c"), ("selectivity", "#d62728")):
            entry = panel["models"][model][metric]
            labels.append(f"{model}\n{metric}")
            values.append(entry["rate"])
            errors.append([entry["rate"] - entry["ci"][0], entry["ci"][1] - entry["rate"]])
            colors.append(color)

    positions = range(len(values))
    right.bar(positions, values, color=colors, alpha=0.85,
              yerr=list(zip(*errors)), capsize=5)
    right.set_xticks(list(positions))
    right.set_xticklabels(labels, fontsize=8)
    right.set_ylabel("rate")
    right.set_ylim(0, 1.08)
    right.set_title("Completeness vs selectivity (95% Wilson)")
    right.grid(alpha=0.3, axis="y")
    # Label above the CI cap, not the bar top, so text never collides with the
    # whisker (it did when placed at value + 0.03).
    for position, value, error in zip(positions, values, errors):
        right.text(position, value + error[1] + 0.035, f"{value:.0%}",
                   ha="center", fontsize=9, fontweight="bold")

    figure.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
