"""Build the three paper figures from paper/figure_data.json.

The JSON is a hand transcription of results/frozen_e91985c/ (commit e91985c);
the panel runs predate the --out flag and cannot be regenerated, so figures
read frozen values rather than recomputing anything.

Figure 1  progressive control stack: four credentials pass, selectivity fails.
Figure 2  the concrete semantic failure, in words, using Phi's real numbers.
Figure 3  per-model replication with Wilson intervals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PASS_COLOR = "#2ca02c"
FAIL_COLOR = "#d62728"
INK = "#222222"


def figure_one(data, out):
    """The control stack: each credential passes, then selectivity fails."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                                         gridspec_kw={"width_ratios": [1.05, 1]})

    steps = [
        ("Effectiveness", "answer flips to the donor's value", True),
        ("Donor specificity", r"$DS_{cross} = +0.718$; non-injected 0.000", True),
        ("Norm control", "norm-matched random: 0.000 at every $\\alpha$", True),
        ("Dose response", "graded in $\\alpha$, threshold $\\approx$ 0.25", True),
        ("Selectivity", "0/240 in both models", False),
    ]
    y = 0.92
    for index, (name, detail, passed) in enumerate(steps):
        color = PASS_COLOR if passed else FAIL_COLOR
        mark = "PASS" if passed else "FAIL"
        box = FancyBboxPatch((0.04, y - 0.105), 0.92, 0.125,
                             boxstyle="round,pad=0.012",
                             linewidth=2.0, edgecolor=color,
                             facecolor=color, alpha=0.12 if passed else 0.22,
                             transform=left.transAxes)
        left.add_patch(box)
        left.text(0.08, y - 0.021, name, transform=left.transAxes,
                  fontsize=11.5, fontweight="bold", color=INK, va="center")
        left.text(0.08, y - 0.072, detail, transform=left.transAxes,
                  fontsize=8.8, color="#444444", va="center")
        left.text(0.92, y - 0.045, mark, transform=left.transAxes, fontsize=11,
                  fontweight="bold", color=color, va="center", ha="right")
        if index < len(steps) - 1:
            left.add_patch(FancyArrowPatch(
                (0.5, y - 0.113), (0.5, y - 0.167),
                transform=left.transAxes, arrowstyle="-|>", mutation_scale=13,
                linewidth=1.4, color="#888888"))
        y -= 0.183
    left.set_title("A progressively strengthened control stack", fontsize=12, pad=12)
    left.axis("off")

    alphas = data["magnitude"]["alphas"]
    right.plot(alphas, data["magnitude"]["p_donor_real"], marker="o", linewidth=2.2,
               color="#1f77b4", label="real donor direction")
    right.plot(alphas, data["magnitude"]["p_donor_norm_matched"], marker="s",
               linewidth=2.2, linestyle="--", color=FAIL_COLOR,
               label="norm-matched random")
    right.set_xlabel(r"intervention magnitude $\alpha$", fontsize=10.5)
    right.set_ylabel(r"$P(Y_{\mathrm{donor}})$", fontsize=10.5)
    right.set_title("Graded and direction-specific,\nnot a magnitude artifact", fontsize=12)
    right.set_ylim(-0.04, 0.82)
    right.grid(alpha=0.3)
    right.legend(loc="upper left", fontsize=9.5)
    right.annotate("identical norm,\nno effect", xy=(0.75, 0.0), xytext=(0.52, 0.17),
                   fontsize=9, color=FAIL_COLOR,
                   arrowprops=dict(arrowstyle="->", color=FAIL_COLOR, linewidth=1.3))

    figure.tight_layout()
    figure.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {out}")


def figure_two(data, out):
    """The semantic failure in concrete terms. Rates are Phi's real measured
    values; the names/object are an illustrative instantiation of the item
    template, not a transcript of one logged trial."""
    figure, axis = plt.subplots(figsize=(10.2, 5.0))
    axis.axis("off")
    phi = data["models"]["phi-3.5-mini"]["probes"]

    axis.text(0.5, 0.975, "One intervention, two questions",
              fontsize=13.5, fontweight="bold", ha="center", color=INK)
    axis.text(0.5, 0.915,
              r"intervention sets the current holder: $do(Z_{curr} := z'_{D})$"
              "   —   Phi-3.5-mini, n=120",
              fontsize=9.8, ha="center", color="#555555")

    panels = [
        (0.03, "Before intervention", "#f7f7f7", "#999999", [
            ("Who has the map?", "Ruby", True, f"{phi['current_holder']['competence']:.0%} correct"),
            ("Which object moved?", "map", True, f"{phi['object_identity']['competence']:.0%} correct"),
        ]),
        (0.52, "After donor intervention", "#fff4f4", FAIL_COLOR, [
            ("Who has the map?", "Ethan", True, "intended effect, 74.2%"),
            ("Which object moved?", "Ethan", False, "should be unchanged"),
        ]),
    ]
    for x0, title, face, edge, rows in panels:
        axis.add_patch(FancyBboxPatch((x0, 0.30), 0.45, 0.56,
                                      boxstyle="round,pad=0.015",
                                      linewidth=1.8, edgecolor=edge,
                                      facecolor=face, transform=axis.transAxes))
        axis.text(x0 + 0.225, 0.805, title, fontsize=11.5, fontweight="bold",
                  ha="center", color=INK)
        y = 0.685
        for question, answer, correct, note in rows:
            mark = "\u2713" if correct else "\u2717"
            color = PASS_COLOR if correct else FAIL_COLOR
            axis.text(x0 + 0.025, y, question, fontsize=10.2, color="#333333")
            axis.text(x0 + 0.025, y - 0.062, f"{answer}", fontsize=12.5,
                      fontweight="bold", color=color)
            axis.text(x0 + 0.118, y - 0.062, mark, fontsize=13,
                      fontweight="bold", color=color)
            axis.text(x0 + 0.158, y - 0.060, note, fontsize=8.6, color="#666666")
            y -= 0.180

    axis.text(0.5, 0.135,
              "The model answers \"which object moved?\" perfectly before the intervention,\n"
              "and with a person's name afterwards. Selectivity: 0/240, 95% CI [0.0%, 1.6%].",
              fontsize=10, ha="center", va="center", color=INK,
              bbox=dict(boxstyle="round,pad=0.45", facecolor="#fffbe6",
                        edgecolor="#e0d48a"))
    axis.text(0.5, 0.015,
              "Percentages are measured; names and object are an illustrative "
              "instantiation of the item template.",
              fontsize=7.8, ha="center", va="center", color="#888888", style="italic")

    figure.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {out}")


def figure_three(data, out):
    """Per-model replication with Wilson intervals. Completeness bars are
    deliberately NOT compared across models: Llama graded 3 probes, Phi 4."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.3),
                                         gridspec_kw={"width_ratios": [1, 1]})

    for axis, metric, title in ((left, "completeness", "Completeness\n(should-change probes)"),
                                (right, "selectivity", "Selectivity\n(should-NOT-change probes)")):
        names, rates, errors = [], [], []
        for model in ("phi-3.5-mini", "llama-3.2-3b"):
            entry = data["models"][model][metric]
            probes = data["models"][model]["graded_probes"]
            names.append(f"{model}\n({probes} graded probes)")
            rates.append(entry["rate"])
            errors.append([entry["rate"] - entry["ci"][0], entry["ci"][1] - entry["rate"]])
        color = PASS_COLOR if metric == "completeness" else FAIL_COLOR
        positions = range(len(rates))
        axis.bar(positions, rates, color=color, alpha=0.85, width=0.55,
                 yerr=list(zip(*errors)), capsize=6, error_kw={"linewidth": 1.6})
        axis.set_xticks(list(positions))
        axis.set_xticklabels(names, fontsize=9)
        axis.set_ylim(0, 1.0)
        axis.set_ylabel("rate", fontsize=10.5)
        axis.set_title(title, fontsize=11.5)
        axis.grid(alpha=0.3, axis="y")
        for position, rate, error in zip(positions, rates, errors):
            entry = data["models"][("phi-3.5-mini", "llama-3.2-3b")[position]][metric]
            axis.text(position, rate + error[1] + 0.045,
                      f"{rate:.1%}\n{entry['successes']}/{entry['n']}",
                      ha="center", fontsize=9.5, fontweight="bold")

    left.text(0.5, -0.30,
              "Not comparable across models: different probe sets\n"
              "(Llama's current_holder was excluded at 46.7% competence)",
              transform=left.transAxes, fontsize=8.4, ha="center",
              color="#8a6d00", style="italic")
    right.text(0.5, -0.30,
              "Zero in both families, across 3 seeds each\n"
              "(per-seed: 0/40, 0/40, 0/40)",
              transform=right.transAxes, fontsize=8.4, ha="center",
              color="#666666", style="italic")

    figure.suptitle("Replication across two model families", fontsize=12.5, y=1.0)
    figure.tight_layout()
    figure.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--data", type=Path, default=here.parent / "paper/figure_data.json")
    parser.add_argument("--outdir", type=Path, default=here.parent / "paper/figures")
    args = parser.parse_args()

    with args.data.open() as stream:
        data = json.load(stream)
    args.outdir.mkdir(parents=True, exist_ok=True)

    figure_one(data, args.outdir / "fig1_control_stack.png")
    figure_two(data, args.outdir / "fig2_semantic_failure.png")
    figure_three(data, args.outdir / "fig3_replication.png")


if __name__ == "__main__":
    main()
