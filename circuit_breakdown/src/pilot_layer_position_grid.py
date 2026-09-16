"""Pilot: is a layer x position selectivity grid worth running overnight?

Motivation. The frozen study patches ONE site (layers [18,20,22,24], final
pre-answer token) and finds selectivity 0/240. A reviewer can fairly reply:
"one badly-chosen site failed, so what?" The answer requires knowing whether
ANY site is selective -- i.e. changes the target variable while preserving
unrelated ones. Every cell of a layer x position grid is exactly that
measurement, so the grid doubles as a search for a positive result.

This is the cheap viability check before committing ~8 GPU-hours. It asks one
question: ARE THE CELLS INTERPRETABLE?

THE GATE. Selectivity is only meaningful where the intervention actually does
something. A cell with zero completeness trivially "preserves" the receiver's
value -- not because the intervention is surgical, but because it is inert.
Reporting that as perfect selectivity would be the same class of error as
reading a stuck counter as a real zero. So each cell reports:

    completeness  how often a should-CHANGE probe moves to the donor's value
    selectivity   how often a should-NOT-change probe keeps the receiver's
    status        INTERPRETABLE only if completeness >= MIN_COMPLETENESS

A cell that is inert is labelled DEAD, not selective. The pilot is worth
scaling only if a reasonable share of cells come back INTERPRETABLE -- and it
is worth scaling MOST if some interpretable cell shows nonzero selectivity,
since that would be the positive result the main-track framing needs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import causal_interchange as ci
from localize import Harness, pick_device
import test_cross_question_panel as panel

# A cell whose intervention lands below this does nothing, so its selectivity
# is vacuous rather than impressive.
MIN_COMPLETENESS = 0.20

CHANGE_PROBE = "current_holder"      # must move to the donor's value
PRESERVE_PROBES = ("original_holder", "object_identity")  # must keep receiver's


def evaluate_cell(harness, pairs, prefix, layers, position):
    """One (layers, position) cell: completeness and selectivity."""
    change_hits = 0
    preserve_hits = 0
    preserve_total = 0
    donor_leak = 0

    for donor, receiver in pairs:
        donor_text = prefix + panel.render(donor, panel.CANONICAL_QUESTION)
        donor_ids = harness.encode(donor_text)
        donor_state = ci.extract_activations(harness, layers, donor_ids, position)
        donor_value = donor["queried_chain"][-1]

        # should CHANGE
        template = panel.PROBES[CHANGE_PROBE][0]
        receiver_ids = harness.encode(prefix + panel.render(receiver, template))
        receiver_state = ci.extract_activations(harness, layers, receiver_ids, position)
        deltas = [d - r for d, r in zip(donor_state, receiver_state)]
        prediction = ci.predict(harness, layers, receiver_ids, position, deltas)["prediction"]
        change_hits += int(prediction == harness.first_id(donor_value))

        # should NOT change
        for probe in PRESERVE_PROBES:
            template = panel.PROBES[probe][0]
            receiver_ids = harness.encode(prefix + panel.render(receiver, template))
            receiver_state = ci.extract_activations(harness, layers, receiver_ids, position)
            deltas = [d - r for d, r in zip(donor_state, receiver_state)]
            prediction = ci.predict(harness, layers, receiver_ids, position, deltas)["prediction"]
            expected = panel.unpatched_answer(receiver, probe, panel.ROUNDS)
            preserve_hits += int(prediction == harness.first_id(expected))
            preserve_total += 1
            donor_leak += int(prediction == harness.first_id(donor_value))

    n = len(pairs)
    return {"completeness": change_hits / n,
            "selectivity": preserve_hits / preserve_total,
            "donor_leak": donor_leak / preserve_total,
            "change_hits": change_hits, "n": n,
            "preserve_hits": preserve_hits, "preserve_total": preserve_total}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="phi-3.5-mini")
    parser.add_argument("--n", type=int, default=16, help="pairs per cell")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    harness = Harness(str(project.parent / args.model), pick_device(args.device))
    harness.model.requires_grad_(False)

    depth = harness.model.config.num_hidden_layers
    layer_sets = {
        "late [18,20,22,24]": [18, 20, 22, 24],          # the frozen study's site
        "mid  [10,12,14,16]": [10, 12, 14, 16],
    }
    positions = [-1, -2, -4]

    prefix = panel.build_prefix(harness, 21, panel.ROUNDS, 2)
    pairs = panel.build_pairs(harness, args.n, 21, panel.ROUNDS)

    print(f"model={args.model}  depth={depth}  n={args.n} pairs/cell")
    print(f"change probe: {CHANGE_PROBE}   preserve probes: {', '.join(PRESERVE_PROBES)}")
    print(f"a cell is INTERPRETABLE only if completeness >= {MIN_COMPLETENESS:.0%}\n")
    print(f"  {'layers':<20} {'pos':>4} {'complete':>9} {'select':>8} {'leak':>7}  status")
    print("  " + "-" * 62)

    cells = []
    for label, layers in layer_sets.items():
        for position in positions:
            result = evaluate_cell(harness, pairs, prefix, layers, position)
            interpretable = result["completeness"] >= MIN_COMPLETENESS
            if not interpretable:
                status = "DEAD (inert; selectivity vacuous)"
            elif result["selectivity"] > 0:
                status = "*** INTERPRETABLE + SELECTIVE ***"
            else:
                status = "interpretable, 0 selectivity"
            result.update({"layers": label, "position": position,
                           "interpretable": interpretable, "status": status})
            cells.append(result)
            print(f"  {label:<20} {position:>4} {result['completeness']:>8.1%} "
                  f"{result['selectivity']:>7.1%} {result['donor_leak']:>6.1%}  {status}",
                  flush=True)

    live = [c for c in cells if c["interpretable"]]
    selective = [c for c in live if c["selectivity"] > 0]

    print("\n" + "=" * 66)
    print(f"  interpretable cells : {len(live)}/{len(cells)}")
    print(f"  of those, selective : {len(selective)}/{len(live) if live else 0}")
    print("=" * 66)

    print("\n--- verdict ---")
    if not live:
        print("  Every cell is inert at this n. The grid would be uninterpretable:")
        print("  zero-completeness cells produce vacuous 100% selectivity.")
        print("  DO NOT scale. Fix the intervention or the gate first.")
    elif selective:
        print(f"  {len(selective)} cell(s) both do something AND preserve unrelated values.")
        print("  This is the positive result the main-track framing needs.")
        print("  SCALE THE GRID -- and confirm these cells at larger n first.")
    else:
        print(f"  {len(live)}/{len(cells)} cells are interpretable, and every one has zero")
        print("  selectivity. No positive result here, but the grid is measurable,")
        print("  so scaling would buy a much stronger NEGATIVE claim: selectivity")
        print("  fails across the whole accessible layer x position space, not just")
        print("  at one site. Worth an overnight run; will not yield a method.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as stream:
            json.dump({"model": args.model, "n": args.n,
                       "min_completeness": MIN_COMPLETENESS, "cells": cells},
                      stream, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
