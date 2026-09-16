"""Spearman correlation between frozen-probe accuracy and fine-tuned intent
accuracy at the same (dataset, depth). Answers the reviewer's "does the probe
actually predict downstream pruning quality?" question quantitatively.

Reads:
  results/{ds}_probe_sweep.json               (sparse: {L/4, L/2, 3L/4, L})
  results/{ds}_probe_sweep_dense.json         (dense: 1..L, if present)
  results/{ds}_pruned_depth{d}.json           (fine-tuned)

Writes a markdown block to stdout with per-dataset and pooled rho + p-value.
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
DATASETS = ["atis", "snips", "massive", "clinc150", "banking77"]
FT_DEPTHS = (1, 2, 3, 6, 9, 12)


def load(name):
    p = RESULTS / name
    return json.load(open(p)) if p.exists() else None


def spearman(xs, ys):
    """Spearman rho + two-sided p-value via scipy if available, else rho only."""
    try:
        from scipy import stats  # type: ignore
        r = stats.spearmanr(xs, ys)
        return float(r.correlation), float(r.pvalue)
    except Exception:
        n = len(xs)
        if n < 2:
            return float("nan"), float("nan")
        rank = lambda vs: [sorted(vs).index(v) + 1 for v in vs]
        rx, ry = rank(xs), rank(ys)
        d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
        rho = 1 - 6 * d2 / (n * (n * n - 1))
        return rho, float("nan")


def per_dataset_pairs(ds: str):
    probe_dense = load(f"{ds}_probe_sweep_dense.json")
    probe_sparse = load(f"{ds}_probe_sweep.json")
    probe = probe_dense or probe_sparse
    if probe is None:
        return []
    d2p = {int(k): v for k, v in probe["depth_to_probe_accuracy"].items()}
    pairs = []
    for d in FT_DEPTHS:
        ft = load(f"{ds}_pruned_depth{d}.json")
        if ft is None or d not in d2p:
            continue
        pairs.append((d2p[d], ft["intent_accuracy"]))
    return pairs


def main():
    print("## Table 7 — Spearman correlation: frozen probe vs fine-tuned intent accuracy\n")
    print("| Dataset | n depths | rho | p |")
    print("|---|---:|---:|---:|")
    pooled_x, pooled_y = [], []
    per_ds_rho = []
    for ds in DATASETS:
        pairs = per_dataset_pairs(ds)
        if len(pairs) < 2:
            print(f"| {ds} | {len(pairs)} | -- | -- |")
            continue
        xs, ys = zip(*pairs)
        rho, p = spearman(list(xs), list(ys))
        pooled_x.extend(xs); pooled_y.extend(ys)
        per_ds_rho.append(rho)
        pstr = f"{p:.3f}" if p == p else "--"
        print(f"| {ds} | {len(pairs)} | {rho:+.3f} | {pstr} |")
    if per_ds_rho:
        mean_r = sum(per_ds_rho) / len(per_ds_rho)
        srt = sorted(per_ds_rho)
        median_r = srt[len(srt) // 2] if len(srt) % 2 else (srt[len(srt) // 2 - 1] + srt[len(srt) // 2]) / 2
        print(f"| **within-dataset mean** | -- | **{mean_r:+.3f}** | -- |")
        print(f"| **within-dataset median** | -- | **{median_r:+.3f}** | -- |")
    if len(pooled_x) >= 2:
        rho, p = spearman(pooled_x, pooled_y)
        pstr = f"{p:.3f}" if p == p else "--"
        print(f"| **pooled** (see caveat) | {len(pooled_x)} | **{rho:+.3f}** | {pstr} |")
    print()
    print("_Caveat: pooled rho conflates within-dataset ranking with between-dataset difficulty. "
          "Within-dataset rho is the metric aligned with C1 (depth selection within a dataset). "
          "Per-dataset n is small so individual values are noisy._\n")

    print("## Table 7b — Probe-selected depth vs empirically shallowest-acceptable fine-tuned depth\n")
    print("| Dataset | dense probe pick | fine-tune best depth | fine-tune best acc | shallowest within 1pt |")
    print("|---|---:|---:|---:|---:|")
    for ds in DATASETS:
        probe = load(f"{ds}_probe_sweep_dense.json") or load(f"{ds}_probe_sweep.json")
        if probe is None:
            continue
        ft_by_d = {}
        for d in FT_DEPTHS:
            j = load(f"{ds}_pruned_depth{d}.json")
            if j is not None:
                ft_by_d[d] = j["intent_accuracy"]
        if not ft_by_d:
            continue
        best_d = max(ft_by_d, key=ft_by_d.get)
        best_acc = ft_by_d[best_d]
        shallowest_ok = min([d for d, a in ft_by_d.items() if a >= best_acc - 0.01],
                             default=best_d)
        print(f"| {ds} | {probe.get('recommended_depth', '--')} | {best_d} | "
              f"{best_acc*100:.2f} | {shallowest_ok} |")


if __name__ == "__main__":
    main()
