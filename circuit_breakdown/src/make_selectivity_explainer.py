"""Figure 4: what 'selectivity 0/240' actually measures.

A zero counter is indistinguishable from a broken counter unless you show it
firing. This diagram makes the measurement legible: the same comparison scores
24/24 with no intervention and 0/24 with the donor patch, so the zero is a
property of the intervention rather than of the instrument.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PASS = "#2ca02c"
FAIL = "#d62728"
INK = "#222222"
MUTED = "#666666"


def panel_box(axis, x, y, width, height, face, edge, lw=1.8):
    axis.add_patch(FancyBboxPatch((x, y), width, height,
                                  boxstyle="round,pad=0.012", linewidth=lw,
                                  edgecolor=edge, facecolor=face,
                                  transform=axis.transAxes))


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=here.parent / "paper/figures/fig4_what_selectivity_means.png")
    args = parser.parse_args()

    figure, axis = plt.subplots(figsize=(11.5, 7.6))
    axis.axis("off")

    axis.text(0.5, 0.975, "What \"selectivity = 0/240\" measures",
              fontsize=14, fontweight="bold", ha="center", color=INK)
    axis.text(0.5, 0.932,
              "The receiver is asked a question whose answer the intervention must NOT change.",
              fontsize=10, ha="center", color=MUTED)

    # ---- the setup ----------------------------------------------------
    panel_box(axis, 0.03, 0.665, 0.42, 0.215, "#eef5ff", "#5b8fd4")
    axis.text(0.24, 0.845, "RECEIVER  (what we patch into)", fontsize=9.6,
              fontweight="bold", ha="center", color="#2a5d9f")
    axis.text(0.055, 0.795, "Ruby has the map.  Ruby gives the map to Mia.",
              fontsize=8.6, color="#333333")
    axis.text(0.055, 0.762, "Mia gives the map to Noah.", fontsize=8.6, color="#333333")
    axis.text(0.055, 0.715, "Q: \"What object did Ruby have at the start?\"",
              fontsize=9.2, color=INK, style="italic")
    axis.text(0.055, 0.684, "correct answer:  map", fontsize=9.2,
              fontweight="bold", color="#2a5d9f")

    panel_box(axis, 0.55, 0.665, 0.42, 0.215, "#fff3e8", "#d98b3a")
    axis.text(0.76, 0.845, "DONOR  (whose state we transplant)", fontsize=9.6,
              fontweight="bold", ha="center", color="#a8621c")
    axis.text(0.575, 0.795, "Ethan has the lamp.  Ethan gives the lamp to Zoe.",
              fontsize=8.6, color="#333333")
    axis.text(0.575, 0.762, "Zoe gives the lamp to Kai.", fontsize=8.6, color="#333333")
    axis.text(0.575, 0.715, "Q: \"Who has the lamp?\"  (always this question)",
              fontsize=9.2, color=INK, style="italic")
    axis.text(0.575, 0.684, "donor's answer:  Kai", fontsize=9.2,
              fontweight="bold", color="#a8621c")

    axis.text(0.5, 0.625,
              "Disjoint names and objects, so any transported answer is a token "
              "absent from the receiver's own text.",
              fontsize=8.6, ha="center", color=MUTED, style="italic")

    # ---- the two measurements ------------------------------------------
    rows = [
        (0.365, "#f4faf4", PASS, "NO patch (control)",
         "\"What object did Ruby have at the start?\"",
         "map", "correct  \u2014  the check fires", "24/24", PASS),
        (0.145, "#fdf2f2", FAIL, "donor state patched in",
         "\"What object did Ruby have at the start?\"",
         "Kai", "a PERSON, not an object", "0/24", FAIL),
    ]
    for y, face, edge, label, question, answer, note, score, color in rows:
        panel_box(axis, 0.03, y, 0.94, 0.215, face, edge)
        axis.text(0.055, y + 0.168, label, fontsize=10.4, fontweight="bold", color=edge)
        axis.text(0.055, y + 0.118, question, fontsize=9.4, color="#333333", style="italic")
        axis.text(0.055, y + 0.062, "model answers:", fontsize=9, color=MUTED)
        axis.text(0.175, y + 0.060, answer, fontsize=13, fontweight="bold", color=color)
        axis.text(0.255, y + 0.062, note, fontsize=9, color=color)
        axis.text(0.80, y + 0.150, "receiver's value kept", fontsize=8.6,
                  ha="center", color=MUTED)
        axis.text(0.80, y + 0.070, score, fontsize=19, fontweight="bold",
                  ha="center", color=color)

    axis.add_patch(FancyArrowPatch((0.30, 0.357), (0.30, 0.333),
                                   transform=axis.transAxes, arrowstyle="-|>",
                                   mutation_scale=17, linewidth=1.8, color="#888888"))
    axis.text(0.325, 0.339, "only the patch changes", fontsize=8.6, color=MUTED)

    axis.text(0.5, 0.048,
              "The counter is not stuck at zero: the identical comparison scores 24/24 when nothing is patched.\n"
              "The intervention was meant to set only WHO HOLDS the object, yet it also destroys WHICH OBJECT exists.",
              fontsize=9.6, ha="center", va="center", color=INK,
              bbox=dict(boxstyle="round,pad=0.5", facecolor="#fffbe6", edgecolor="#e0d48a"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
