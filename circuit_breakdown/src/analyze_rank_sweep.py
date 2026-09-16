"""Descriptive audit of legacy rank sweeps without unverified paired inference.

Select seeds explicitly: old rank8 used three while most other ranks used two.
This entry point preserves raw rates and refuses to manufacture sample pairing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from analyze_dissociation import legacy_descriptive
from study_data import write_new


def analyze_ranks(outdir, ranks, seeds):
    if not ranks or not seeds or len(set(ranks)) != len(ranks) or len(set(seeds)) != len(seeds):
        raise ValueError("Specify distinct nonempty ranks and seeds")
    rows = []
    for rank in ranks:
        result = legacy_descriptive("diss", [f"track{rank}", f"comp{rank}"], seeds, outdir)
        rows.append({"rank": rank, **result})
    return {"analysis_version": "legacy_rank_descriptive_v1", "ranks": rows,
            "limitations": ["No sample IDs: no validated paired p-values or independent sample-count claim.",
                            "Finite ranks and ratios near a zero control mean do not establish a scaling law.",
                            "Legacy break is unconditional negative-edit error, not same-sign harm."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 4, 8, 16, 26])
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    result = analyze_ranks(args.outdir, args.ranks, args.seeds)
    write_new(args.output, result)
    print(f"Saved descriptive rank audit -> {args.output}; no significance claims")


if __name__ == "__main__":
    main()