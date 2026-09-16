"""Run equal-rank predictive-vs-causal subspace diagnostics.

This development-only runner compares readout decodability and causal control
for tracking, PCA, random and gradient-causal coordinate subspaces at matched
rank. It uses the existing frozen manifest but writes separate outputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from audit_study import assert_matches, read_json
from intervene_pareto import fit_subspace_bases, forward_logits, optimize_sample
from localize import Harness, pick_device
from predictive_vs_causal_subspace import (
    centroid_probe_accuracy,
    centroid_probe_fit,
    coordinate_basis_from_scores,
    energy_fraction,
    pca_basis,
    project_vectors,
    random_basis,
    summarize_decode_control,
    validate_protocol,
)
from run_convergence_pilot import input_identity, pilot_records
from run_validated_study import configuration, control_set, effective_rate, layer_budgets
from study_data import fingerprint, load_manifest, write_new


def state_matrices(harness, records, layers, clean):
    matrices = {layer: [] for layer in layers}
    for record in records:
        outputs = harness.cache_layer_outputs(record.clean_ids if clean else record.corrupt_ids)
        for layer in layers:
            matrices[layer].append(outputs[layer][0, record.dpos].detach().float().cpu())
    return [torch.stack(matrices[layer], 0) for layer in layers]


def pca_bases(clean_states, corrupt_states, rank, device):
    bases, information = [], []
    for layer_index, (clean, corrupt) in enumerate(zip(clean_states, corrupt_states)):
        basis, info = pca_basis(torch.cat([clean, corrupt], 0), rank)
        bases.append(basis.to(device))
        information.append({"layer_index": layer_index, **info})
    return bases, information


def random_bases(hidden, layers, rank, seed, device):
    return [random_basis(hidden, rank, seed + layer).to(device) for layer in layers]


def causal_coordinate_scores(harness, records, layers):
    scores = [torch.zeros(harness.model.config.hidden_size) for _ in layers]
    for record in records:
        vectors = [torch.zeros(1, harness.model.config.hidden_size, device=harness.device, requires_grad=True)
                   for _ in layers]
        logits = forward_logits(harness, record.corrupt_ids, layers, vectors, record.dpos)
        loss = -F.log_softmax(logits.float(), dim=0)[record.cid]
        loss.backward()
        for index, vector in enumerate(vectors):
            scores[index] += vector.grad.detach().float().cpu().reshape(-1).abs()
    return [score / len(records) for score in scores]


def causal_bases(scores, rank, device):
    bases, indices = [], []
    for score in scores:
        basis, selected = coordinate_basis_from_scores(score, rank)
        bases.append(basis.to(device))
        indices.append(selected)
    return bases, indices


def evaluate(harness, record, layers, vectors=None):
    if vectors is None:
        with torch.no_grad():
            logits = harness.model(record.corrupt_ids).logits[0, -1]
    else:
        logits = forward_logits(harness, record.corrupt_ids, layers,
                                [vector.to(harness.device) for vector in vectors], record.dpos)
    probabilities = torch.softmax(logits.float(), dim=0)
    competitors = logits.clone()
    competitors[record.cid] = -torch.inf
    return {"target_probability": float(probabilities[record.cid]),
            "target_margin": float(logits[record.cid] - competitors.max()),
            "target_success": int(logits.argmax().item() == record.cid),
            "prediction": int(logits.argmax().item())}


def markdown(result):
    lines = ["# Predictive vs Causal Subspace Pilot", "",
             "Development-only diagnostic; not a confirmatory test.", "",
             "| Condition | Decode accuracy | Control success | Energy | Recovery vs full |",
             "|---|---:|---:|---:|---:|"]
    for rank, rank_summary in result["summary_by_rank"].items():
        for condition, values in rank_summary.items():
            decode = values.get("decodability") or {}
            recovery = values["probability_recovery_fraction_vs_full"]
            lines.append("| k={rank} {condition} | {decode} | {successes}/{n} | {energy:.1f}% | {recovery} |".format(
                rank=rank,
                condition=condition,
                decode="n/a" if not decode else f"{100 * decode['accuracy']:.1f}%",
                successes=values["successes"],
                n=values["n"],
                energy=100 * values["mean_energy_fraction"],
                recovery="n/a" if recovery is None else f"{100 * recovery:.1f}%"))
    lines.extend(["", "## Guardrails", ""])
    lines.extend(f"- {item}" for item in result["protocol"]["limitations"])
    lines.append("")
    return "\n".join(lines)


def run(project, protocol_path, device):
    started = time.monotonic()
    protocol = read_json(protocol_path)
    validate_protocol(protocol)
    manifest = load_manifest(project / protocol["manifest"])
    assert_matches(manifest["sha256"], protocol["manifest_sha256"], "manifest")
    output = project / protocol["output"]
    harness = Harness(str(project.parent / protocol["model"]), pick_device(device))
    harness.model.requires_grad_(False)
    train, development = pilot_records(harness, manifest, protocol)
    layers = read_json(project / "results/convergence_pilot_v1/run.json")["comparison"]["layers"]
    clean_train = state_matrices(harness, train, layers, True)
    corrupt_train = state_matrices(harness, train, layers, False)
    clean_dev = state_matrices(harness, development, layers, True)
    corrupt_dev = state_matrices(harness, development, layers, False)
    controls, predictions = control_set(harness, [])
    budgets, _ = layer_budgets(harness, train, layers, .3)
    causal_scores = causal_coordinate_scores(harness, train[:min(16, len(train))], layers)
    rows_by_rank = {}
    basis_metadata = {}
    for rank in protocol["ranks"]:
        tracking = fit_subspace_bases(harness, train, layers, rank)
        pca, pca_info = pca_bases(clean_train, corrupt_train, rank, harness.device)
        random = random_bases(harness.model.config.hidden_size, layers, rank, 76000 + rank, harness.device)
        causal, causal_indices = causal_bases(causal_scores, rank, harness.device)
        basis_sets = {"tracking": tracking, "pca": pca, "random": random, "causal": causal}
        basis_metadata[str(rank)] = {"pca": pca_info, "causal_coordinate_indices": causal_indices}
        probes = {name: centroid_probe_fit(clean_train, corrupt_train, bases) for name, bases in basis_sets.items()}
        decode = {name: centroid_probe_accuracy(probes[name], clean_dev, corrupt_dev, bases)
                  for name, bases in basis_sets.items()}
        rank_rows = []
        for record in development:
            baseline = evaluate(harness, record, layers)
            full_cfg = configuration(budgets, 32, .05, objective="cross_entropy")
            full_edit = optimize_sample(harness, record, layers, controls, predictions, None, full_cfg)
            full_vectors = full_edit.detached()
            full = evaluate(harness, record, layers, full_vectors)
            conditions = {f"full_k{rank}": {**full, "energy_fraction": 1.0}}
            for name, bases in basis_sets.items():
                rate = effective_rate(harness, bases, .05)
                cfg = configuration(budgets, 32, rate, objective="cross_entropy")
                edit = optimize_sample(harness, record, layers, controls, predictions, bases, cfg)
                vectors = edit.detached()
                measured = evaluate(harness, record, layers, vectors)
                conditions[f"{name}_k{rank}"] = {**measured, "energy_fraction": energy_fraction(vectors, full_vectors),
                                                  "decodability": decode[name]}
            projected = project_vectors(full_vectors, tracking)
            measured = evaluate(harness, record, layers, projected)
            conditions[f"full_projected_to_tracking_k{rank}"] = {
                **measured, "energy_fraction": energy_fraction(projected, full_vectors),
                "decodability": decode["tracking"]}
            rank_rows.append({"rank": rank, "sample_id": record.sample_id, "input": input_identity(record),
                              "baseline": baseline, "full": full, "conditions": conditions})
        rows_by_rank[str(rank)] = rank_rows
    summary_by_rank = {rank: summarize_decode_control(rows) for rank, rows in rows_by_rank.items()}
    n_rows = sum(len(rows) for rows in rows_by_rank.values())
    result = {"completed": True, "protocol": protocol, "protocol_sha256": fingerprint(protocol),
              "device": harness.device, "layers": layers, "n_rows": n_rows, "basis_metadata": basis_metadata,
                            "rows_by_rank": rows_by_rank, "summary_by_rank": summary_by_rank,
              "elapsed_seconds": time.monotonic() - started,
              "reviewer_gap_status": "G1 remains open until untouched confirmation and literature comparison"}
    write_new(output / "summary.json", result)
    with (output / "RESULTS.md").open("x") as stream:
        stream.write(markdown(result))
    print(f"Saved predictive-vs-causal pilot -> {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", type=Path, default=Path(__file__).resolve().parents[1] / "protocols/predictive_vs_causal_subspace_v1.json")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    run(args.project.resolve(), args.protocol.resolve(), args.device)


if __name__ == "__main__":
    main()