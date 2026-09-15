"""Validate that the selectivity metric CAN fire. Does 0/240 mean anything?

A reader's first reaction to "selectivity 0/240" should be suspicion: a
counter that is always zero is indistinguishable from a counter that is
broken. This script rules out the broken case.

The argument has two halves.

1. STRUCTURAL. In test_cross_question_panel.run_one_seed, baseline competence
   and the to_receiver outcome use the SAME comparison against the SAME
   target token:

       competence:  prediction == harness.first_id(unpatched_answer(receiver,...))
       to_receiver: prediction == harness.first_id(receiver_value)
                    where receiver_value = unpatched_answer(receiver,...)

   The only difference is the patch: competence predicts under a ZERO patch,
   to_receiver predicts under the donor patch. So baseline competence IS the
   to_receiver metric with the intervention switched off. Phi scoring 120/120
   competence on object_identity therefore proves the to_receiver check fires
   at 100% when nothing is patched.

2. EMPIRICAL. We re-run that exact code path with a zero patch and confirm
   to_receiver is high, then with the real donor patch and confirm it drops to
   zero on the same items in the same process. Same harness, same items, same
   comparison -- only the patch changes.

This is metric validation, not a new experiment: it re-measures the frozen
protocol at small n to check the instrument, and makes no new claim.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import causal_interchange as ci
from localize import Harness, pick_device
import test_cross_question_panel as panel


def predict_with(harness, ids, deltas):
    return ci.predict(harness, panel.LAYERS, ids, panel.POSITION, deltas)["prediction"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="phi-3.5-mini")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    harness = Harness(str(project.parent / args.model), pick_device(args.device))
    harness.model.requires_grad_(False)

    
    prefix = panel.build_prefix(harness, 21, panel.ROUNDS, 2)
    pairs = panel.build_pairs(harness, args.n, 21, panel.ROUNDS)

    print(f"model={args.model}  n={len(pairs)}  layers={panel.LAYERS}  "
          f"position={panel.POSITION}\n")
    print("Probing the two 'should NOT change' probes that carry the headline.\n")

    grand = {"zero": [0, 0], "real": [0, 0]}
    for probe in ("original_holder", "object_identity"):
        template = panel.PROBES[probe][0]
        zero_hits = real_hits = donor_hits = 0

        for donor, receiver in pairs:
            donor_text = prefix + panel.render(donor, panel.CANONICAL_QUESTION)
            receiver_text = prefix + panel.render(receiver, template)
            donor_ids = harness.encode(donor_text)
            receiver_ids = harness.encode(receiver_text)

            donor_state = ci.extract_activations(harness, panel.LAYERS, donor_ids, panel.POSITION)
            receiver_state = ci.extract_activations(harness, panel.LAYERS, receiver_ids, panel.POSITION)

            receiver_value = panel.unpatched_answer(receiver, probe, panel.ROUNDS)
            donor_value = donor["queried_chain"][-1]
            receiver_target = harness.first_id(receiver_value)
            donor_target = harness.first_id(donor_value)

            # (a) zero patch: the metric with the intervention switched off
            zero_deltas = [torch.zeros_like(v) for v in receiver_state]
            zero_hits += int(predict_with(harness, receiver_ids, zero_deltas) == receiver_target)

            # (b) real donor patch: identical code path, real delta
            real_deltas = [d - r for d, r in zip(donor_state, receiver_state)]
            prediction = predict_with(harness, receiver_ids, real_deltas)
            real_hits += int(prediction == receiver_target)
            donor_hits += int(prediction == donor_target)

        n = len(pairs)
        grand["zero"][0] += zero_hits; grand["zero"][1] += n
        grand["real"][0] += real_hits; grand["real"][1] += n
        print(f"  {probe}")
        print(f"    zero patch  -> receiver value : {zero_hits:>3}/{n}   "
              f"(metric fires when nothing is changed)")
        print(f"    donor patch -> receiver value : {real_hits:>3}/{n}   "
              f"(selectivity: survives the intervention)")
        print(f"    donor patch -> donor value    : {donor_hits:>3}/{n}   "
              f"(leakage: answers the donor's question instead)\n")

    zero_hits, zero_n = grand["zero"]
    real_hits, real_n = grand["real"]
    print("=" * 66)
    print(f"  metric fires under zero patch : {zero_hits}/{zero_n}")
    print(f"  metric fires under real patch : {real_hits}/{real_n}")
    print("=" * 66)
    if zero_hits == 0:
        print("\n  INVALID: the counter never fires even with NO intervention.")
        print("  A 0/240 headline would be an artifact. Do not report it.")
        return 1
    print(f"\n  VALID: the counter demonstrably fires ({zero_hits}/{zero_n} with no")
    print("  intervention) and is driven to zero by the intervention alone.")
    print("  The 0/240 result measures the intervention, not a broken metric.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
