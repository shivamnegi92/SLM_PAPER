"""Aggregate all result JSONs into the paper's tables (Markdown).

Produces:
 - Table 1: fair-budget head-to-head (B2 encoder-5k vs ours depth3/depth12), matched 5000-example budget.
 - Table 2: depth Pareto (intent/slot/latency at depths 3/6/9/12) per dataset.
 - Table 3: CRF vs softmax slot head at depth 3.
 - Table 4: multi-seed mean +/- std for the headline (ours depth3 vs depth12).

Run: python scripts/aggregate_results.py > ../RESULTS_FINAL_TABLES.md
"""
from __future__ import annotations

import json
import glob
from pathlib import Path

from slmpaper.aggregate import mean_std

RESULTS = Path(__file__).resolve().parent.parent / "results"
DATASETS = ["atis", "snips", "massive", "clinc150", "banking77"]


def load(name):
    p = RESULTS / name
    return json.load(open(p)) if p.exists() else None


def slot_f1(j):
    return j["slot_f1"]["f1"] if j and j.get("n_tags", 1) > 1 else None


def fmt(x, pct=True):
    if x is None:
        return "  --  "
    return f"{x*100:.2f}" if pct else f"{x:.2f}"


def table1_fair_budget():
    print("## Table 1 — Fair head-to-head (matched 5000-example budget)\n")
    print("| Dataset | B2 encoder acc | B2 slot F1 | ours d3 acc | ours d3 slotF1 | ours d3 P50 | ours d12 acc | ours d12 slotF1 | ours d12 P50 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for ds in DATASETS:
        enc = load(f"{ds}_encoder_5k.json")
        d3 = load(f"{ds}_pruned_depth3.json")
        d12 = load(f"{ds}_pruned_depth12.json")
        row = [ds]
        row += [fmt(enc["intent_accuracy"]) if enc else "--", fmt(slot_f1(enc)) if enc else "--"]
        row += [fmt(d3["intent_accuracy"]), fmt(slot_f1(d3)), f"{d3['latency_ms']['p50']:.2f}"]
        row += [fmt(d12["intent_accuracy"]), fmt(slot_f1(d12)), f"{d12['latency_ms']['p50']:.2f}"]
        print("| " + " | ".join(str(x) for x in row) + " |")
    print()


def table2_pareto():
    print("## Table 2 — Depth Pareto (ours, per dataset)\n")
    print("| Dataset | Depth | Intent Acc | Slot F1 | P50 (ms) |")
    print("|---|---:|---:|---:|---:|")
    for ds in DATASETS:
        for d in (3, 6, 9, 12):
            j = load(f"{ds}_pruned_depth{d}.json")
            if not j:
                continue
            print(f"| {ds} | {d} | {fmt(j['intent_accuracy'])} | {fmt(slot_f1(j))} | {j['latency_ms']['p50']:.2f} |")
    print()


def table3_crf():
    print("## Table 3 — CRF vs softmax slot head (depth 3)\n")
    print("| Dataset | Softmax slot F1 | CRF slot F1 | Delta |")
    print("|---|---:|---:|---:|")
    for ds in ["atis", "snips", "massive"]:
        sm = load(f"{ds}_pruned_depth3.json")
        crf = load(f"{ds}_pruned_depth3_crf.json")
        if not (sm and crf):
            continue
        a, b = slot_f1(sm), slot_f1(crf)
        delta = (b - a) * 100 if (a is not None and b is not None) else None
        ds_delta = f"{delta:+.2f}" if delta is not None else "--"
        print(f"| {ds} | {fmt(a)} | {fmt(b)} | {ds_delta} |")
    print()


def table4_seeds():
    print("## Table 4 — Multi-seed (mean +/- std over seeds {42,1,2}), ours\n")
    print("| Dataset | Depth | Intent Acc (mean+/-std) | Slot F1 (mean+/-std) |")
    print("|---|---:|---:|---:|")
    for ds in DATASETS:
        for d in (3, 12):
            files = [f"{ds}_pruned_depth{d}.json",           # seed 42
                     f"{ds}_pruned_depth{d}_seed1.json",
                     f"{ds}_pruned_depth{d}_seed2.json"]
            js = [load(f) for f in files]
            js = [j for j in js if j]
            if not js:
                continue
            accs = [j["intent_accuracy"] for j in js]
            am, asd = mean_std(accs)
            sfs = [slot_f1(j) for j in js if slot_f1(j) is not None]
            if sfs:
                sm, ssd = mean_std(sfs)
                slot_str = f"{sm*100:.2f} +/- {ssd*100:.2f}"
            else:
                slot_str = "--"
            print(f"| {ds} | {d} | {am*100:.2f} +/- {asd*100:.2f} (n={len(js)}) | {slot_str} |")
    print()


if __name__ == "__main__":
    print("# SLM_PAPER — Final Aggregated Tables\n")
    print("Backbone gpt2-124M pruned; matched 5000-example budget; CPU P50 latency.\n")
    table1_fair_budget()
    table2_pareto()
    table3_crf()
    table4_seeds()
