"""Run the causal interchange pilot: does a candidate subspace transport a
donor's reasoning state into a receiver, at rank 32 on Llama, the sweet spot
identified from the predictive-vs-causal v2 matrix (tracking/PCA/learned_causal
all succeed 5-6/8 there under fresh optimization, while projection of the
known solution fails at every rank for every basis).

For every basis (full/oracle, tracking, pca, random, gradient_coordinate,
learned_causal), reports:
  - correct-donor interchange accuracy (IIA) on disjoint items
  - random-donor interchange accuracy (IIA), a full derangement so every item
    gets a donor that is NOT its constructed match
  - same-answer items: raw-state cosine (donor vs receiver) and whether an
    already-correct receiver survives the patch

Bases are fit on the SAME train split used throughout this project (disjoint
from both the steering dev set and these interchange items). Development-only
diagnostic; not a confirmatory test.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from audit_study import read_json
from causal_interchange import evaluate_disjoint_item, evaluate_same_answer_item
from dataset import restrict_to_single_token
from experiment_metrics import rate_ci
from intervene_pareto import fit_subspace_bases
from interchange_dataset import InterchangeItem, generate
from localize import Harness, pick_device
from predictive_vs_causal_v2 import gradient_coordinate_scores, learned_causal_basis
from run_convergence_pilot import pilot_records
from run_predictive_vs_causal_subspace import pca_bases, random_bases, state_matrices
from run_predictive_vs_causal_subspace_v2 import gradient_coordinate_bases
from study_data import fingerprint, load_manifest, write_new


def random_donor_pairs(items):
    """Full derangement: pair every item's receiver with a DIFFERENT item's
    donor. Raises if a valid derangement with no accidental answer match is
    not found within a bounded number of attempts (n is small; this is fast).
    """
    import random as pyrandom
    n = len(items)
    if n < 2:
        raise ValueError("Random-donor control needs at least two items")
    order = list(range(n))
    for _ in range(500):
        pyrandom.shuffle(order)
        if all(order[i] != i for i in range(n)) and all(
                items[order[i]].donor_answer != items[i].receiver_answer for i in range(n)):
            break
    else:
        raise RuntimeError("Could not find a valid random-donor derangement")
    paired = []
    for i in range(n):
        donor = items[order[i]]
        receiver = items[i]
        paired.append(InterchangeItem(
            kind="disjoint", rounds=receiver.rounds,
            donor_prompt=donor.donor_prompt, receiver_prompt=receiver.receiver_prompt,
            donor_answer=donor.donor_answer, receiver_answer=receiver.receiver_answer,
            donor_object=donor.donor_object, receiver_object=receiver.receiver_object,
            donor_names=donor.donor_names, receiver_names=receiver.receiver_names,
            metadata={"donor_chain": donor.metadata["donor_chain"],
                      "receiver_chain": receiver.metadata["receiver_chain"]}))
    return paired


def fit_all_bases(harness, train, layers, rank, clean_train, corrupt_train, gradient_scores, seed):
    tracking = fit_subspace_bases(harness, train, layers, rank)
    pca, _ = pca_bases(clean_train, corrupt_train, rank, harness.device)
    random = random_bases(harness.model.config.hidden_size, layers, rank, seed + rank, harness.device)
    gradient_coordinate, _ = gradient_coordinate_bases(gradient_scores, rank, harness.device)
    learned_causal_cfg = {"lr": 0.02, "seed": seed, "n_examples": 8,
                          "convergence_eps": 1e-3, "convergence_patience": 5, "max_steps": 50}
    learned_causal, _ = learned_causal_basis(harness, train[:8], layers, rank, learned_causal_cfg)
    return {"full": None, "tracking": tracking, "pca": pca, "random": random,
            "gradient_coordinate": gradient_coordinate, "learned_causal": learned_causal}


def summarize(rows, key):
    flags = [row[key] for row in rows]
    rate, interval = rate_ci(flags)
    return {"n": len(flags), "successes": int(sum(flags)), "rate": rate, "ci": interval}


def run(project, protocol_path, device):
    started = time.monotonic()
    protocol = read_json(protocol_path)
    output = project / protocol["output"]
    output.mkdir(parents=True, exist_ok=True)

    harness = Harness(str(project.parent / protocol["model"]), pick_device(device))
    harness.model.requires_grad_(False)
    restrict_to_single_token(harness.tok)

    manifest = load_manifest(project / protocol["manifest"])
    train, _ = pilot_records(harness, manifest, protocol)
    layers = protocol["layers"]
    splits = manifest["tasks"][protocol["task"]][str(protocol["development_seed"])]
    prefix = "".join(f"{pair['clean_prompt']} {pair['clean_target']}.\n" for pair in splits["fewshot"])

    clean_train = state_matrices(harness, train, layers, True)
    corrupt_train = state_matrices(harness, train, layers, False)
    gradient_scores = gradient_coordinate_scores(harness, train[:12], layers)
    bases = fit_all_bases(harness, train, layers, protocol["rank"], clean_train, corrupt_train,
                          gradient_scores, protocol["training_seed"])

    disjoint_items = generate("disjoint", protocol["n_items"], seed=protocol["item_seed"],
                              rounds=protocol["rounds"], tokenizer=harness.tok)
    same_answer_items = generate("same_answer", protocol["n_items"], seed=protocol["item_seed"],
                                 rounds=protocol["rounds"], tokenizer=harness.tok)
    random_donor_items = random_donor_pairs(disjoint_items)

    result = {"model": protocol["model"], "rank": protocol["rank"], "layers": layers,
              "n_items": protocol["n_items"], "disjoint": {}, "random_donor": {}, "same_answer": {}}
    for name, basis in bases.items():
        print(f"=== basis: {name} ===", flush=True)
        disjoint_rows = [evaluate_disjoint_item(harness, layers, item, basis, name, harness.first_id, prefix)
                         for item in disjoint_items]
        random_rows = [evaluate_disjoint_item(harness, layers, item, basis, name, harness.first_id, prefix)
                       for item in random_donor_items]
        same_rows = [evaluate_same_answer_item(harness, layers, item, basis, name, prefix)
                    for item in same_answer_items]

        result["disjoint"][name] = {
            "iia": summarize(disjoint_rows, "interchange_success"),
            "baseline_correct": summarize(disjoint_rows, "baseline_matches_receiver"),
            "mean_p_donor_baseline": float(np.mean([r["p_donor_answer_baseline"] for r in disjoint_rows])),
            "mean_p_donor_patched": float(np.mean([r["p_donor_answer_patched"] for r in disjoint_rows])),
        }
        result["random_donor"][name] = {"iia": summarize(random_rows, "interchange_success")}
        result["same_answer"][name] = {
            "mean_raw_state_cosine": float(np.mean([r["raw_state_cosine_donor_vs_receiver"] for r in same_rows])),
            "patch_preserves_answer": summarize(
                [{"kept": int(r["patched_prediction"] == r["baseline_prediction"])} for r in same_rows], "kept"),
        }
        print(f"  disjoint IIA: {result['disjoint'][name]['iia']['successes']}/{result['disjoint'][name]['iia']['n']}  "
              f"random-donor IIA: {result['random_donor'][name]['iia']['successes']}/{result['random_donor'][name]['iia']['n']}  "
              f"same-answer cosine: {result['same_answer'][name]['mean_raw_state_cosine']:.3f}", flush=True)

    result["elapsed_seconds"] = time.monotonic() - started
    result["protocol_sha256"] = fingerprint(protocol)
    write_new(output / "summary.json", result)
    print(f"Saved causal interchange pilot -> {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", type=Path,
                        default=Path(__file__).resolve().parents[1] / "protocols/causal_interchange_v1.json")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    run(args.project.resolve(), args.protocol.resolve(), args.device)


if __name__ == "__main__":
    main()
