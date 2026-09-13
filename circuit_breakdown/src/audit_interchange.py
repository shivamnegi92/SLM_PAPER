"""Audit the causal_interchange_v1 pairs: why is random-donor IIA (23/32)
almost as high as correct-donor IIA (24/32)?

For each of the 32 disjoint items (regenerated deterministically from the
same seed as the completed run), reports the full row the reviewer asked
for, plus a no-op control, plus IIA recomputed under two conditionings:
  - baseline receiver prediction is CORRECT (task actually being solved)
  - receiver / donor / random-donor answers are MUTUALLY DISTINCT (no label
    collision between the correct donor and the randomly assigned one)

This only evaluates the FULL (oracle) basis, since every candidate basis is
already a flat 0/32 -- the open question is specific to full-space.
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
from run_convergence_pilot import pilot_records

PROTOCOL = {
    "task": "transfer", "development_seed": 1, "training_seed": 1,
    "development_indices": list(range(8)),
}
LAYERS = [18, 20, 22, 24]
N_ITEMS = 32
ITEM_SEED = 7
ROUNDS = 2
DERANGEMENT_SEED = 12345  # NEW: fixed, was unseeded in the completed run


def derangement_no_answer_collision(items, seed):
    """Full derangement of donors, with BOTH constraints enforced:
    random_donor_answer != receiver_answer (already had this)
    random_donor_answer != correct_donor_answer (MISSING in the original run --
    likely the dominant explanation for random IIA tracking correct IIA).
    """
    rng = pyrandom.Random(seed)
    n = len(items)
    order = list(range(n))
    for attempt in range(2000):
        rng.shuffle(order)
        ok = all(order[i] != i for i in range(n))
        ok = ok and all(items[order[i]].donor_answer != items[i].receiver_answer for i in range(n))
        ok = ok and all(items[order[i]].donor_answer != items[i].donor_answer for i in range(n))
        if ok:
            return order
    raise RuntimeError(f"No valid collision-free derangement found in {attempt + 1} attempts")


def main():
    h = Harness("../llama-3.2-3b", pick_device("auto"))
    h.model.requires_grad_(False)
    dataset.restrict_to_single_token(h.tok)

    manifest = load_manifest("data/validated_manifest_v1.json")
    splits = manifest["tasks"]["transfer"]["1"]
    prefix = "".join(f"{p['clean_prompt']} {p['clean_target']}.\n" for p in splits["fewshot"])

    items = ic.generate("disjoint", N_ITEMS, seed=ITEM_SEED, rounds=ROUNDS, tokenizer=h.tok)

    # First: check the ORIGINAL (unseeded, in-run) derangement's collision rate
    # by reconstructing what the original code's constraint allowed.
    order_original_style = None
    for attempt in range(2000):
        candidate = list(range(N_ITEMS))
        pyrandom.shuffle(candidate)
        if (all(candidate[i] != i for i in range(N_ITEMS))
                and all(items[candidate[i]].donor_answer != items[i].receiver_answer for i in range(N_ITEMS))):
            order_original_style = candidate
            break
    collisions = sum(1 for i in range(N_ITEMS)
                     if items[order_original_style[i]].donor_answer == items[i].donor_answer)
    print(f"ORIGINAL-STYLE derangement (one valid sample): "
          f"{collisions}/{N_ITEMS} items have random_donor_answer == correct_donor_answer")
    print("(this collision was NOT prevented in the completed run -- likely dominant confound)\n")

    order = derangement_no_answer_collision(items, DERANGEMENT_SEED)

    print(f"{'i':>2} {'recv_ans':>9} {'base_ok':>7} {'donor_ans':>9} {'rand_ans':>9} "
          f"{'d!=r':>4} {'rd!=r':>5} {'rd!=d':>5} {'corr_patch':>10} {'rand_patch':>10} "
          f"{'noop_patch':>10} {'corr_ok':>7} {'rand_ok':>7} {'noop_ok':>7}")

    rows = []
    for i, item in enumerate(items):
        receiver_id = h.first_id(item.receiver_answer)
        donor_id = h.first_id(item.donor_answer)
        random_donor = items[order[i]]
        random_donor_id = h.first_id(random_donor.donor_answer)

        receiver_ids = h.encode(prefix + item.receiver_prompt)
        donor_ids = h.encode(prefix + item.donor_prompt)
        random_donor_ids = h.encode(prefix + random_donor.donor_prompt)

        receiver_state = ci.extract_activations(h, LAYERS, receiver_ids, -1)
        donor_state = ci.extract_activations(h, LAYERS, donor_ids, -1)
        random_donor_state = ci.extract_activations(h, LAYERS, random_donor_ids, -1)

        zero_vectors = [torch.zeros_like(v) for v in receiver_state]
        correct_vectors = ci.interchange_vectors(donor_state, receiver_state, None)
        random_vectors = ci.interchange_vectors(random_donor_state, receiver_state, None)
        noop_vectors = ci.interchange_vectors(receiver_state, receiver_state, None)  # true no-op: delta = 0

        baseline = ci.predict(h, LAYERS, receiver_ids, -1, zero_vectors)
        correct_patched = ci.predict(h, LAYERS, receiver_ids, -1, correct_vectors)
        random_patched = ci.predict(h, LAYERS, receiver_ids, -1, random_vectors)
        noop_patched = ci.predict(h, LAYERS, receiver_ids, -1, noop_vectors)

        baseline_ok = int(baseline["prediction"] == receiver_id)
        corr_ok = int(correct_patched["prediction"] == donor_id)
        rand_ok = int(random_patched["prediction"] == random_donor_id)
        noop_ok = int(noop_patched["prediction"] == baseline["prediction"])
        d_ne_r = int(item.donor_answer != item.receiver_answer)
        rd_ne_r = int(random_donor.donor_answer != item.receiver_answer)
        rd_ne_d = int(random_donor.donor_answer != item.donor_answer)

        rows.append({
            "baseline_ok": baseline_ok, "corr_ok": corr_ok, "rand_ok": rand_ok, "noop_ok": noop_ok,
            "d_ne_r": d_ne_r, "rd_ne_r": rd_ne_r, "rd_ne_d": rd_ne_d,
            "donor_answer": item.donor_answer, "receiver_answer": item.receiver_answer,
            "random_donor_answer": random_donor.donor_answer,
        })
        print(f"{i:>2} {item.receiver_answer:>9} {baseline_ok:>7} {item.donor_answer:>9} "
              f"{random_donor.donor_answer:>9} {d_ne_r:>4} {rd_ne_r:>5} {rd_ne_d:>5} "
              f"{correct_patched['prediction'] == donor_id!s:>10} {random_patched['prediction'] == random_donor_id!s:>10} "
              f"{noop_patched['prediction'] == baseline['prediction']!s:>10} {corr_ok:>7} {rand_ok:>7} {noop_ok:>7}")

    n = len(rows)
    baseline_correct_n = sum(r["baseline_ok"] for r in rows)
    corr_iia = sum(r["corr_ok"] for r in rows)
    rand_iia = sum(r["rand_ok"] for r in rows)
    noop_iia = sum(r["noop_ok"] for r in rows)

    print(f"\n=== OVERALL (n={n}, collision-free random-donor derangement) ===")
    print(f"baseline correct: {baseline_correct_n}/{n}")
    print(f"correct-donor IIA: {corr_iia}/{n} ({100*corr_iia/n:.0f}%)")
    print(f"random-donor IIA (collision-free): {rand_iia}/{n} ({100*rand_iia/n:.0f}%)")
    print(f"no-op IIA (should be ~100%, sanity check on mechanism): {noop_iia}/{n} ({100*noop_iia/n:.0f}%)")
    print(f"SPECIFICITY (correct - random): {100*(corr_iia-rand_iia)/n:.0f} points")

    mutually_distinct = [r for r in rows if r["d_ne_r"] and r["rd_ne_r"] and r["rd_ne_d"]]
    print(f"\n=== conditional on mutually distinct targets (n={len(mutually_distinct)}) ===")
    print(f"correct-donor IIA: {sum(r['corr_ok'] for r in mutually_distinct)}/{len(mutually_distinct)}")
    print(f"random-donor IIA:  {sum(r['rand_ok'] for r in mutually_distinct)}/{len(mutually_distinct)}")

    baseline_correct_rows = [r for r in rows if r["baseline_ok"]]
    print(f"\n=== conditional on baseline receiver already correct (n={len(baseline_correct_rows)}) ===")
    print(f"correct-donor IIA: {sum(r['corr_ok'] for r in baseline_correct_rows)}/{len(baseline_correct_rows)}")
    print(f"random-donor IIA:  {sum(r['rand_ok'] for r in baseline_correct_rows)}/{len(baseline_correct_rows)}")


if __name__ == "__main__":
    main()
