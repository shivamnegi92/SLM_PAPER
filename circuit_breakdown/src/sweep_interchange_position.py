"""Sweep candidate interchange positions to find where full-space interchange
shows genuine donor specificity (correct-donor IIA >> random-donor IIA),
before re-testing compact bases against a validated protocol.

The completed run patched at position -1 (the '?' token) and found
correct=75%, random=72% -- no specificity; a full swap there appears to
hijack the tail computation regardless of which donor is used. This sweep
tests earlier positions on the SAME 4-layer set, on a reduced item count for
speed (12 pairs, not 32 -- this is a search, not a confirmatory run).
"""
from __future__ import annotations

import sys
sys.path.insert(0, "src")

import random as pyrandom

import torch

from localize import Harness, pick_device
import dataset
import interchange_dataset as ic
import causal_interchange as ci
from study_data import load_manifest

LAYERS = [18, 20, 22, 24]
N_ITEMS = 12
ITEM_SEED = 7
ROUNDS = 2
DERANGEMENT_SEED = 12345
POSITIONS = [-1, -2, -3, -4, -5, -6, -7]
POSITION_LABELS = {-1: "'?'", -2: "object", -3: "'the'", -4: "'has'",
                   -5: "'Who'", -6: "'.' (end of body)", -7: "recipient name"}


def collision_free_derangement(items, seed):
    rng = pyrandom.Random(seed)
    n = len(items)
    order = list(range(n))
    for attempt in range(2000):
        rng.shuffle(order)
        if (all(order[i] != i for i in range(n))
                and all(items[order[i]].donor_answer != items[i].receiver_answer for i in range(n))
                and all(items[order[i]].donor_answer != items[i].donor_answer for i in range(n))):
            return order
    raise RuntimeError("No valid derangement found")


def evaluate_at_position(h, items, order, prefix, position):
    baseline_ok = correct_ok = random_ok = 0
    for i, item in enumerate(items):
        random_donor = items[order[i]]
        receiver_ids = h.encode(prefix + item.receiver_prompt)
        donor_ids = h.encode(prefix + item.donor_prompt)
        random_ids = h.encode(prefix + random_donor.donor_prompt)

        receiver_state = ci.extract_activations(h, LAYERS, receiver_ids, position)
        donor_state = ci.extract_activations(h, LAYERS, donor_ids, position)
        random_state = ci.extract_activations(h, LAYERS, random_ids, position)

        zero_vectors = [torch.zeros_like(v) for v in receiver_state]
        correct_vectors = ci.interchange_vectors(donor_state, receiver_state, None)
        random_vectors = ci.interchange_vectors(random_state, receiver_state, None)

        baseline = ci.predict(h, LAYERS, receiver_ids, position, zero_vectors)
        correct_patched = ci.predict(h, LAYERS, receiver_ids, position, correct_vectors)
        random_patched = ci.predict(h, LAYERS, receiver_ids, position, random_vectors)

        receiver_id = h.first_id(item.receiver_answer)
        donor_id = h.first_id(item.donor_answer)
        random_id = h.first_id(random_donor.donor_answer)

        baseline_ok += int(baseline["prediction"] == receiver_id)
        correct_ok += int(correct_patched["prediction"] == donor_id)
        random_ok += int(random_patched["prediction"] == random_id)
    return baseline_ok, correct_ok, random_ok


def main():
    h = Harness("../llama-3.2-3b", pick_device("auto"))
    h.model.requires_grad_(False)
    dataset.restrict_to_single_token(h.tok)

    manifest = load_manifest("data/validated_manifest_v1.json")
    splits = manifest["tasks"]["transfer"]["1"]
    prefix = "".join(f"{p['clean_prompt']} {p['clean_target']}.\n" for p in splits["fewshot"])

    items = ic.generate("disjoint", N_ITEMS, seed=ITEM_SEED, rounds=ROUNDS, tokenizer=h.tok)
    order = collision_free_derangement(items, DERANGEMENT_SEED)

    print(f"{'position':>10} {'token':>18} {'baseline':>9} {'correct':>8} {'random':>7} {'specificity':>12}")
    for position in POSITIONS:
        baseline_ok, correct_ok, random_ok = evaluate_at_position(h, items, order, prefix, position)
        specificity = correct_ok - random_ok
        print(f"{position:>10} {POSITION_LABELS[position]:>18} {baseline_ok:>5}/{N_ITEMS:<3} "
              f"{correct_ok:>4}/{N_ITEMS:<3} {random_ok:>4}/{N_ITEMS:<2} "
              f"{specificity:>+5} / {N_ITEMS}")


if __name__ == "__main__":
    main()
