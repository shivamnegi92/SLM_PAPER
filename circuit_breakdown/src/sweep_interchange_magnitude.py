"""Magnitude sweep at the final-token interchange position: is the apparent
full-space effect a genuine dose-response (real causal signal, diluted by
scale) or a threshold/hijack effect (real only at alpha=1, and non-specific
even there)?

For each alpha in {0, 0.125, 0.25, 0.5, 0.75, 1.0}, at position -1 only
(the only position with ANY effect per the prior sweep), layers [18,20,22,24]
(unchanged, no basis refit):

  correct donor:  h_R + alpha * (h_D - h_R),        target = Y_D
  random donor:   h_R + alpha * (h_Rand - h_R),     target = Y_Rand
  norm-matched
  random direction: h_R + alpha * r, ||r|| = ||h_D - h_R||,  target = Y_D

Donor specificity is measured continuously via probability deltas, not raw
top-1 IIA, so partial effects at intermediate alpha are visible even when no
top-1 flip occurs:

  Delta_donor  = P(Y=Y_D    | patched, correct donor)   - P(Y=Y_D    | baseline)
  Delta_random = P(Y=Y_Rand | patched, random donor)    - P(Y=Y_Rand | baseline)
  DS           = Delta_donor - Delta_random

  Delta_matched_norm = P(Y=Y_D | patched, norm-matched random direction) - P(Y=Y_D | baseline)
  direction_specificity = Delta_donor - Delta_matched_norm

DS near zero at every alpha => threshold/hijack, not graded causal control.
direction_specificity near zero => effect is about injected MAGNITUDE, not
the donor's specific direction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
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
ALPHAS = [0.0, 0.125, 0.25, 0.5, 0.75, 1.0]


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


def norm_matched_random(reference_vectors, seed):
    """Per-layer Gaussian direction, globally rescaled so the COMBINED norm
    across all layers matches `reference_vectors` exactly."""
    generator = torch.Generator().manual_seed(seed)
    raw = [torch.randn(v.shape, generator=generator, dtype=torch.float64).float().to(v.device) for v in reference_vectors]
    reference_norm = float(sum((v.float() ** 2).sum() for v in reference_vectors) ** 0.5)
    raw_norm = float(sum((v ** 2).sum() for v in raw) ** 0.5)
    scale = reference_norm / raw_norm if raw_norm > 0 else 0.0
    return [v * scale for v in raw]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None,
                        help="optional JSON path so the headline figure reads "
                             "these exact numbers rather than recomputing them")
    args = parser.parse_args()

    h = Harness("../llama-3.2-3b", pick_device("auto"))
    h.model.requires_grad_(False)
    dataset.restrict_to_single_token(h.tok)

    manifest = load_manifest("data/validated_manifest_v1.json")
    splits = manifest["tasks"]["transfer"]["1"]
    prefix = "".join(f"{p['clean_prompt']} {p['clean_target']}.\n" for p in splits["fewshot"])

    items = ic.generate("disjoint", N_ITEMS, seed=ITEM_SEED, rounds=ROUNDS, tokenizer=h.tok)
    order = collision_free_derangement(items, DERANGEMENT_SEED)

    # Precompute per-item states/ids once; reused across every alpha.
    prepared = []
    for i, item in enumerate(items):
        random_donor = items[order[i]]
        receiver_ids = h.encode(prefix + item.receiver_prompt)
        donor_ids = h.encode(prefix + item.donor_prompt)
        random_ids = h.encode(prefix + random_donor.donor_prompt)

        receiver_state = ci.extract_activations(h, LAYERS, receiver_ids, POSITION)
        donor_state = ci.extract_activations(h, LAYERS, donor_ids, POSITION)
        random_state = ci.extract_activations(h, LAYERS, random_ids, POSITION)

        correct_delta = [d - r for d, r in zip(donor_state, receiver_state)]
        random_delta = [d - r for d, r in zip(random_state, receiver_state)]
        matched_direction = norm_matched_random(correct_delta, seed=1000 + i)

        prepared.append({
            "receiver_ids": receiver_ids, "receiver_state": receiver_state,
            "correct_delta": correct_delta, "random_delta": random_delta,
            "matched_direction": matched_direction,
            "donor_id": h.first_id(item.donor_answer),
            "random_id": h.first_id(random_donor.donor_answer),
            "receiver_id": h.first_id(item.receiver_answer),
        })

    print(f"{'alpha':>6} {'baseline_ok':>11} {'IIA_correct':>11} {'IIA_random':>10} "
          f"{'d_donor':>8} {'d_random':>9} {'DS':>7} {'d_matched':>10} {'dir_spec':>9}")

    record = {"model": "llama-3.2-3b", "layers": LAYERS, "position": POSITION,
              "n_items": N_ITEMS, "alphas": [], "p_donor_real": [],
              "p_donor_norm_matched": [], "donor_specificity": []}

    for alpha in ALPHAS:
        baseline_ok = iia_correct = iia_random = 0
        p_donor_baseline, p_donor_patched = [], []
        p_random_baseline, p_random_patched = [], []
        p_donor_matched_patched = []

        for row in prepared:
            zero_vectors = [torch.zeros_like(v) for v in row["receiver_state"]]
            correct_vectors = [alpha * v for v in row["correct_delta"]]
            random_vectors = [alpha * v for v in row["random_delta"]]
            matched_vectors = [alpha * v for v in row["matched_direction"]]

            baseline = ci.predict(h, LAYERS, row["receiver_ids"], POSITION, zero_vectors)
            correct_patched = ci.predict(h, LAYERS, row["receiver_ids"], POSITION, correct_vectors)
            random_patched = ci.predict(h, LAYERS, row["receiver_ids"], POSITION, random_vectors)
            matched_patched = ci.predict(h, LAYERS, row["receiver_ids"], POSITION, matched_vectors)

            baseline_ok += int(baseline["prediction"] == row["receiver_id"])
            iia_correct += int(correct_patched["prediction"] == row["donor_id"])
            iia_random += int(random_patched["prediction"] == row["random_id"])

            p_donor_baseline.append(float(baseline["probabilities"][row["donor_id"]]))
            p_donor_patched.append(float(correct_patched["probabilities"][row["donor_id"]]))
            p_random_baseline.append(float(baseline["probabilities"][row["random_id"]]))
            p_random_patched.append(float(random_patched["probabilities"][row["random_id"]]))
            p_donor_matched_patched.append(float(matched_patched["probabilities"][row["donor_id"]]))

        delta_donor = float(np.mean(p_donor_patched) - np.mean(p_donor_baseline))
        delta_random = float(np.mean(p_random_patched) - np.mean(p_random_baseline))
        delta_matched = float(np.mean(p_donor_matched_patched) - np.mean(p_donor_baseline))
        ds = delta_donor - delta_random
        direction_specificity = delta_donor - delta_matched

        print(f"{alpha:>6.3f} {baseline_ok:>7}/{N_ITEMS:<3} {iia_correct:>7}/{N_ITEMS:<3} "
              f"{iia_random:>6}/{N_ITEMS:<3} {delta_donor:>+8.3f} {delta_random:>+9.3f} "
              f"{ds:>+7.3f} {delta_matched:>+10.3f} {direction_specificity:>+9.3f}")

        # Absolute probabilities (not deltas) so the figure shows the curve the
        # reader expects: P(Y_donor) rising with alpha for the real direction
        # while the norm-matched control stays flat.
        record["alphas"].append(alpha)
        record["p_donor_real"].append(float(np.mean(p_donor_patched)))
        record["p_donor_norm_matched"].append(float(np.mean(p_donor_matched_patched)))
        record["donor_specificity"].append(ds)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as stream:
            json.dump(record, stream, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
