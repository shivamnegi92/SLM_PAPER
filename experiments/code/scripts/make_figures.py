"""Generate paper figures from the result JSONs. Outputs PDFs to
experiments/figures/ for direct \\includegraphics in the LaTeX draft.

Figures:
  fig_probe_sweep.pdf  - probe accuracy vs depth, all 5 datasets (C1)
  fig_pareto.pdf       - intent accuracy vs latency, depths 3/6/9/12 (C1)
  fig_latency_bars.pdf - generative vs encoder vs ours latency, log scale (C2)
  fig_parsefail.pdf    - parse-failure rate generative vs discriminative (C2)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = Path(__file__).resolve().parent.parent
RESULTS = CODE / "results"
FIG = CODE.parent / "figures"
FIG.mkdir(exist_ok=True)
DATASETS = ["atis", "snips", "massive", "clinc150", "banking77"]
MARKERS = {"atis": "o", "snips": "s", "massive": "^", "clinc150": "D", "banking77": "v"}


def load(name):
    p = RESULTS / name
    return json.load(open(p)) if p.exists() else None


def fig_probe_sweep():
    plt.figure(figsize=(5, 3.4))
    for ds in DATASETS:
        # Prefer the dense per-layer sweep when available.
        j = load(f"{ds}_probe_sweep_dense.json") or load(f"{ds}_probe_sweep.json")
        if not j:
            continue
        d2a = j["depth_to_probe_accuracy"]
        depths = sorted(int(k) for k in d2a)
        accs = [d2a[str(d)] * 100 for d in depths]
        plt.plot(depths, accs, marker=MARKERS[ds], label=ds)
    plt.axvline(3, color="grey", ls="--", lw=1, alpha=0.7)
    plt.text(3.2, plt.ylim()[0] + 1, "probe pick (d=3)", fontsize=8, color="grey")
    plt.xlabel("Backbone depth (layers)")
    plt.ylabel("Linear-probe intent acc (%)")
    plt.title("Probe accuracy saturates early (C1)")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(FIG / "fig_probe_sweep.pdf")
    plt.close()


def fig_pareto():
    plt.figure(figsize=(5, 3.4))
    for ds in DATASETS:
        xs, ys = [], []
        for d in (1, 2, 3, 6, 9, 12):
            j = load(f"{ds}_pruned_depth{d}.json")
            if not j:
                continue
            xs.append(j["latency_ms"]["p50"])
            ys.append(j["intent_accuracy"] * 100)
        if xs:
            plt.plot(xs, ys, marker=MARKERS[ds], label=ds, alpha=0.85)
    plt.xlabel("P50 latency (ms, CPU)")
    plt.ylabel("Intent accuracy (%)")
    plt.title("Depth Pareto: accuracy vs latency (C1)")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(FIG / "fig_pareto.pdf")
    plt.close()


def fig_latency_bars():
    gen = load("atis_generative.json")
    enc = load("atis_encoder_5k.json") or load("atis_encoder.json")
    ours = load("atis_pruned_depth3.json")
    labels, vals = [], []
    if gen: labels.append("Generative\n(gpt2)"); vals.append(gen["latency_ms"]["p50"])
    if enc: labels.append("Encoder\n(DistilBERT)"); vals.append(enc["latency_ms"]["p50"])
    if ours: labels.append("Ours\n(depth-3)"); vals.append(ours["latency_ms"]["p50"])
    plt.figure(figsize=(4, 3.4))
    bars = plt.bar(labels, vals, color=["#c44", "#48c", "#4a4"])
    plt.yscale("log")
    plt.ylabel("P50 latency (ms, log scale)")
    plt.title("Inference latency on ATIS (C2)")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v * 1.1, f"{v:.1f}ms",
                 ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "fig_latency_bars.pdf")
    plt.close()


def fig_parsefail():
    rows = []
    for ds in ["atis", "snips", "clinc150"]:
        g = load(f"{ds}_generative.json")
        if g:
            rows.append((ds, g["parse_failure_rate"] * 100))
    if not rows:
        return
    plt.figure(figsize=(4, 3.4))
    labels = [r[0] for r in rows]
    x = range(len(labels))
    plt.bar([i - 0.2 for i in x], [r[1] for r in rows], width=0.4,
            label="Generative", color="#c44")
    plt.bar([i + 0.2 for i in x], [0 for _ in rows], width=0.4,
            label="Ours (by construction)", color="#4a4")
    plt.xticks(list(x), labels)
    plt.ylabel("Parse-failure rate (%)")
    plt.title("Structured-output reliability (C2)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "fig_parsefail.pdf")
    plt.close()


if __name__ == "__main__":
    fig_probe_sweep()
    fig_pareto()
    fig_latency_bars()
    fig_parsefail()
    print(f"figures written to {FIG}")
    for f in sorted(FIG.glob("*.pdf")):
        print(" ", f.name)
