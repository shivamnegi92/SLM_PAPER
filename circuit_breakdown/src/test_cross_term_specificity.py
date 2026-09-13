"""Cross-term specificity test -- the measurement the earlier random-donor
control got wrong.

Earlier framing compared:
    patch donor D   -> P(Y_D)      (succeeds)
    patch random R  -> P(Y_R)      (also succeeds)
and concluded "no specificity". That conclusion does not follow: both are
legitimate interchange interventions with different donors, and the high
level causal model predicts BOTH should succeed. Two separate forward passes
each hitting their own target is evidence the intervention WORKS, not
evidence it is indiscriminate.

The real question is the cross-term, within a SINGLE patched forward pass:

    patch with donor D, then compare  P(Y_D)  vs  P(Y_other)

where Y_other is a different donor's answer that was NOT injected. If the
output tracks precisely which donor was injected, we expect
P(Y_D) >> P(Y_other). Donor specificity is therefore measured WITHIN one
intervention, not ACROSS two:

    DS_cross = P(Y_D | do(D)) - P(Y_other | do(D))

Also reports, for the same single pass, whether the receiver's own original
answer is suppressed: P(Y_R | do(D)) should fall relative to baseline if the
donor state genuinely displaces the receiver's tracked state.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "src")

import random as pyrandom

import numpy as np
import torch

from localize import Harness, pick_device
import dataset
import interchange_dataset as ic
import causal_interchange as ci
from study_data import load_manifest

LAYERS = [18, 20, 22, 24]
POSITION = -1
N_ITEMS = 16
ITEM_SEED = 7
ROUNDS = 2
DERANGEMENT_SEED = 12345


def collision_free_derangement(items, seed):
    rng = pyrandom.Random(seed)
    n = len(items)
    order = list(range(n))
    for _ in range(2000):
        rng.shuffle(order)
        if (all(order[i] != i for i in range(n))
                and all(items[order[i]].donor_answer != items[i].receiver_answer for i in range(n))
                and all(items[order[i]].donor_answer != items[i].donor_answer for i in range(n))):
            return order
    raise RuntimeError("No valid derangement found")


def main():
    h = Harness("../llama-3.2-3b", pick_device("auto"))
    h.model.requires_grad_(False)
    dataset.restrict_to_single_token(h.tok)

    manifest = load_manifest("data/validated_manifest_v1.json")
    splits = manifest["tasks"]["transfer"]["1"]
    prefix = "".join(f"{p['clean_prompt']} {p['clean_target']}.\n" for p in splits["fewshot"])

    items = ic.generate("disjoint", N_ITEMS, seed=ITEM_SEED, rounds=ROUNDS, tokenizer=h.tok)
    order = collision_free_derangement(items, DERANGEMENT_SEED)

    print(f"{'i':>2} {'recv':>7} {'donor':>7} {'other':>7} | "
          f"{'P(Yd)base':>9} {'P(Yd)patch':>10} | {'P(Yo)base':>9} {'P(Yo)patch':>10} | "
          f"{'P(Yr)base':>9} {'P(Yr)patch':>10} | {'top1':>7}")

    rows = []
    for i, item in enumerate(items):
        other = items[order[i]]  # a donor that was NOT injected
        receiver_ids = h.encode(prefix + item.receiver_prompt)
        donor_ids = h.encode(prefix + item.donor_prompt)

        receiver_state = ci.extract_activations(h, LAYERS, receiver_ids, POSITION)
        donor_state = ci.extract_activations(h, LAYERS, donor_ids, POSITION)
        vectors = ci.interchange_vectors(donor_state, receiver_state, None)
        zero_vectors = [torch.zeros_like(v) for v in receiver_state]

        baseline = ci.predict(h, LAYERS, receiver_ids, POSITION, zero_vectors)
        patched = ci.predict(h, LAYERS, receiver_ids, POSITION, vectors)

        donor_id = h.first_id(item.donor_answer)
        other_id = h.first_id(other.donor_answer)
        receiver_id = h.first_id(item.receiver_answer)

        row = {
            "p_donor_base": float(baseline["probabilities"][donor_id]),
            "p_donor_patch": float(patched["probabilities"][donor_id]),
            "p_other_base": float(baseline["probabilities"][other_id]),
            "p_other_patch": float(patched["probabilities"][other_id]),
            "p_receiver_base": float(baseline["probabilities"][receiver_id]),
            "p_receiver_patch": float(patched["probabilities"][receiver_id]),
            "top1_is_donor": int(patched["prediction"] == donor_id),
            "top1_is_other": int(patched["prediction"] == other_id),
        }
        rows.append(row)
        top1 = ("donor" if row["top1_is_donor"] else
                "OTHER" if row["top1_is_other"] else "neither")
        print(f"{i:>2} {item.receiver_answer:>7} {item.donor_answer:>7} {other.donor_answer:>7} | "
              f"{row['p_donor_base']:>9.3f} {row['p_donor_patch']:>10.3f} | "
              f"{row['p_other_base']:>9.3f} {row['p_other_patch']:>10.3f} | "
              f"{row['p_receiver_base']:>9.3f} {row['p_receiver_patch']:>10.3f} | {top1:>7}")

    n = len(rows)
    mean = lambda key: float(np.mean([r[key] for r in rows]))
    ds_cross = mean("p_donor_patch") - mean("p_other_patch")

    print(f"\n=== CROSS-TERM SPECIFICITY (single patched pass, n={n}) ===")
    print(f"P(Y_donor | do(D)):    {mean('p_donor_patch'):.3f}   (baseline {mean('p_donor_base'):.3f})")
    print(f"P(Y_other | do(D)):    {mean('p_other_patch'):.3f}   (baseline {mean('p_other_base'):.3f})")
    print(f"P(Y_receiver | do(D)): {mean('p_receiver_patch'):.3f}   (baseline {mean('p_receiver_base'):.3f})")
    print(f"\nDS_cross = P(Y_donor|do(D)) - P(Y_other|do(D)) = {ds_cross:+.3f}")
    print(f"receiver suppression = {mean('p_receiver_patch') - mean('p_receiver_base'):+.3f}")
    print(f"top-1 is injected donor: {sum(r['top1_is_donor'] for r in rows)}/{n}")
    print(f"top-1 is NON-injected other donor: {sum(r['top1_is_other'] for r in rows)}/{n}")


if __name__ == "__main__":
    main()
