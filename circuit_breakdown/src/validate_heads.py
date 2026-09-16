"""Held-out mover-head patching and selected downstream compensation checks."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import time

import numpy as np
import torch

from experiment_metrics import paired_comparison
from heads import HeadHarness
from intervene_pareto import fit_subspace_bases
from localize import pick_device
from run_validated_study import baseline_metrics
from study_data import load_manifest, manifest_records, write_new


@contextmanager
def replace_heads(harness, replacements):
    by_layer = {}
    for (layer, head), vector in replacements.items():
        if not 0 <= layer < len(harness.oprojs) or not 0 <= head < harness.n_heads:
            raise ValueError("Invalid head site")
        if vector.numel() != harness.head_dim:
            raise ValueError("Replacement must contain one head output")
        by_layer.setdefault(layer, []).append((head, vector))

    def make_hook(sites):
        def hook(module, inputs):
            tensor = inputs[0].clone()
            for head, vector in sites:
                start = head * harness.head_dim
                tensor[0, -1, start:start + harness.head_dim] = vector.to(tensor)
            return (tensor,) + inputs[1:]
        return hook

    handles = []
    try:
        for layer, sites in by_layer.items():
            handles.append(harness.oprojs[layer].register_forward_pre_hook(make_hook(sites)))
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def patched_logits(harness, ids, replacements):
    with replace_heads(harness, replacements):
        return harness.model(ids).logits[0, -1]


def matched_random(sites, n_heads, rng, exclude=()):
    result = []
    exclusions = set(exclude) | set(sites)
    for layer, count in Counter(layer for layer, head in sites).items():
        candidates = [head for head in range(n_heads) if (layer, head) not in exclusions]
        if len(candidates) < count:
            raise ValueError("Not enough disjoint depth-matched random heads")
        result.extend((layer, int(head)) for head in rng.choice(candidates, count, replace=False))
    return result


def cache_vectors(harness, cache, sites):
    return {(layer, head): cache[layer][0, -1, head * harness.head_dim:(head + 1) * harness.head_dim]
            for layer, head in sites}


@contextmanager
def activation_gradients(model):
    model.requires_grad_(False)
    def enable_gradient(module, inputs, output):
        return output.requires_grad_(True)
    handle = model.get_input_embeddings().register_forward_hook(enable_gradient)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def projected_patch(harness, ids, clean_outputs, layers, bases, position):
    if len(layers) != len(bases):
        raise ValueError("One projection basis is required per layer")

    def make_hook(clean, basis):
        def hook(module, inputs, output):
            tensor = (output[0] if isinstance(output, tuple) else output).clone()
            difference = clean[0, position].to(tensor) - tensor[0, position]
            projection = basis.to(tensor)
            tensor[0, position] += (difference @ projection.T) @ projection
            return (tensor,) + output[1:] if isinstance(output, tuple) else tensor
        return hook

    handles = []
    try:
        for layer, basis in zip(layers, bases):
            handles.append(harness.layers[layer].register_forward_hook(make_hook(clean_outputs[layer], basis)))
        return harness.model(ids).logits[0, -1]
    finally:
        for handle in handles:
            handle.remove()


def run(args):
    manifest = load_manifest(args.manifest)
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    harness = HeadHarness(args.model, pick_device(args.device))
    train = manifest_records(harness, manifest, args.task, 0)
    competence = baseline_metrics(harness, train["dev"])
    if min(competence["clean_accuracy"], competence["counterfactual_accuracy"]) < .8:
        write_new(destination, {"completed": True, "task": args.task,
                                "status": "baseline_competence_gate_not_met", "baseline_dev": competence})
        return
    started = time.monotonic()
    effects = np.zeros((len(harness.layers), harness.n_heads))
    with activation_gradients(harness.model):
        for index, record in enumerate(train["train"][:args.n_discovery]):
            effects += harness.head_attribution(record.clean_ids, record.corrupt_ids, record.cid, record.kid, -1)
            print(f"head discovery {index + 1}/{args.n_discovery}", flush=True)
    ranked = sorted(((layer, head) for layer in range(len(harness.layers)) for head in range(harness.n_heads)),
                    key=lambda site: effects[site], reverse=True)
    top = ranked[:8]
    source = next(site for site in ranked if site[0] < len(harness.layers) - 2)
    receivers = [site for site in ranked if site[0] > source[0]][:4]
    rng = np.random.default_rng(731)
    random_sets = [matched_random(top, harness.n_heads, rng) for repeat in range(5)]
    random_receivers = matched_random(receivers, harness.n_heads, rng, exclude=[source])
    harness.model.zero_grad(set_to_none=True)
    harness.model.requires_grad_(False)
    from run_validated_study import matched_layers
    projection_layers = matched_layers(len(harness.layers))
    tracking_bases = fit_subspace_bases(harness, train["train"], projection_layers, 8)
    control_bases = fit_subspace_bases(harness, train["train"], projection_layers, 8, complement=True, comp_dim=8)
    rows = []
    for seed in manifest["seeds"]:
        for index, record in enumerate(manifest_records(harness, manifest, args.task, seed)["test"][:args.n_test]):
            clean_cache = harness.cache_head_inputs(record.clean_ids)
            clean = harness.logits_last(record.clean_ids)
            corrupt = harness.logits_last(record.corrupt_ids)
            base = float(clean[record.cid] - clean[record.kid])
            corrupt_margin = float(corrupt[record.cid] - corrupt[record.kid])
            gap = base - corrupt_margin
            source_zero = {source: torch.zeros(harness.head_dim, device=harness.device)}
            receiver_clamp = cache_vectors(harness, clean_cache, receivers)
            random_clamp = cache_vectors(harness, clean_cache, random_receivers)

            def margin(replacements):
                logits = patched_logits(harness, record.clean_ids, replacements)
                return float(logits[record.cid] - logits[record.kid])

            ablated = margin(source_zero)
            frozen = margin({**source_zero, **receiver_clamp})
            frozen_random = margin({**source_zero, **random_clamp})
            top_metric = harness.patch_heads(record.corrupt_ids, clean_cache, top, record.cid, record.kid, -1)
            random_metrics = [harness.patch_heads(record.corrupt_ids, clean_cache, sites,
                                                  record.cid, record.kid, -1) for sites in random_sets]
            valid_gap = gap > 1e-5
            clean_outputs = harness.cache_layer_outputs(record.clean_ids)
            full_patch = harness.patch_sites(record.corrupt_ids, clean_outputs,
                                              [(layer, record.dpos) for layer in projection_layers],
                                              record.cid, record.kid)
            track_patch = projected_patch(harness, record.corrupt_ids, clean_outputs,
                                           projection_layers, tracking_bases, record.dpos)
            control_patch = projected_patch(harness, record.corrupt_ids, clean_outputs,
                                             projection_layers, control_bases, record.dpos)
            record_row = {
                "sample_id": record.sample_id, "input_hash": record.input_hash, "seed": seed,
                "clean_target_id": record.cid, "corrupt_target_id": record.kid,
                "baseline_clean_correct": int(clean.argmax() == record.cid),
                "baseline_corrupt_correct": int(corrupt.argmax() == record.kid),
                "clean_corrupt_logit_gap": gap,
                "top8_faithfulness": (top_metric - corrupt_margin) / gap if valid_gap else None,
                "random8_faithfulness": (float(np.mean(random_metrics)) - corrupt_margin) / gap if valid_gap else None,
                "source_ablation_loss": base - ablated,
                "receiver_clamp_compensation": ablated - frozen,
                "random_clamp_compensation": ablated - frozen_random,
                "baseline_receiver_clamp_delta": margin(receiver_clamp) - base,
                "full_residual_faithfulness": (full_patch - corrupt_margin) / gap if valid_gap else None,
                "tracking_projection_faithfulness": float(track_patch[record.cid] - track_patch[record.kid] - corrupt_margin) / gap if valid_gap else None,
                "control_projection_faithfulness": float(control_patch[record.cid] - control_patch[record.kid] - corrupt_margin) / gap if valid_gap else None,
            }
            if abs(record_row["baseline_receiver_clamp_delta"]) > 1e-4:
                raise ValueError("Unablated receiver clamp changed baseline; hook check failed")
            rows.append(record_row)
            if (index + 1) % 8 == 0:
                print(f"held-out heads seed{seed}: {index + 1}/{args.n_test}", flush=True)
    valid = [row for row in rows if row["top8_faithfulness"] is not None]
    top_rows = [{**row, "value": row["top8_faithfulness"]} for row in valid]
    random_rows = [{**row, "value": row["random8_faithfulness"]} for row in valid]
    compensation = [{**row, "value": row["receiver_clamp_compensation"]} for row in rows]
    random_compensation = [{**row, "value": row["random_clamp_compensation"]} for row in rows]
    result = {
        "completed": True, "protocol": "heldout_heads_compensation_v1", "model": Path(args.model).name,
        "task": args.task, "manifest_sha256": manifest["sha256"], "baseline_dev": competence,
        "top8": top, "source": source, "receivers": receivers, "random_sets": random_sets,
        "random_receivers": random_receivers, "n_discovery": args.n_discovery, "n_test": len(rows),
        "projection_layers": projection_layers,
        "projection_protocol": "sequential projected clean-minus-current replacement; natural patch norms, not steering budgets",
        "n_valid_logit_gaps": len(valid), "records": rows,
        "faithfulness_difference": paired_comparison(top_rows, random_rows, "value", binary=False) if valid else None,
        "compensation_vs_random": paired_comparison(compensation, random_compensation, "value", binary=False),
        "elapsed_seconds": time.monotonic() - started,
        "limitations": ["Selected source and receivers only; not an exhaustive self-repair search.",
                        "Zero ablation is an intervention, not a natural counterfactual.",
                        "This is component localization and a receiver-clamp diagnostic, not a complete circuit."]}
    write_new(destination, result)
    print(f"saved -> {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", default="intermediate")
    parser.add_argument("--device", default="auto", choices=["cpu", "mps", "auto"])
    parser.add_argument("--n-discovery", type=int, default=12)
    parser.add_argument("--n-test", type=int, default=24)
    run(parser.parse_args())


if __name__ == "__main__":
    main()