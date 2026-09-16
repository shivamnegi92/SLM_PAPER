"""Run a development-only causal dimensionality analysis over saved full edits.

This pilot does not optimize new interventions. It takes the successful
full-space vectors saved by the convergence pilot and asks how much of their
behavioral effect survives projection into lower-dimensional subspaces.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from audit_study import assert_matches, read_json
from experiment_metrics import paired_delta_ci, rate_ci
from intervene_pareto import fit_subspace_bases, forward_logits
from localize import Harness, pick_device
from run_convergence_pilot import input_identity, pilot_records
from study_data import fingerprint, load_manifest, write_new


def validate_protocol(protocol):
    required = {
        "version": "causal_dimensionality_pilot_v1",
        "scope": "development_only_diagnostic_not_confirmatory",
        "model": "phi-3.5-mini",
        "task": "transfer",
        "source_run": "results/convergence_pilot_v1",
        "source_condition": "full",
        "source_checkpoint": 128,
        "training_seed": 1,
        "development_seed": 1,
        "development_indices": list(range(8)),
    }
    for name, value in required.items():
        assert_matches(protocol[name], value, f"protocol.{name}")
    ranks = protocol["subspace_ranks"]
    if ranks != sorted(set(ranks)) or ranks[0] < 1:
        raise ValueError("Subspace ranks must be distinct positive ascending integers")


def load_source_cases(project, protocol, development):
    root = project / protocol["source_run"] / "cases" / protocol["source_condition"]
    cases = []
    for record in development:
        path = root / f"{record.sample_id}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        assert_matches(saved["input"], input_identity(record), "source.input")
        snapshot = saved["trajectory"]["snapshots"][str(protocol["source_checkpoint"])]
        cases.append({"record": record, "saved": saved, "vectors": snapshot["vectors"]})
    return cases


def project_vectors(vectors, bases):
    projected = []
    for vector, basis in zip(vectors, bases):
        basis = basis.to(vector.device, dtype=vector.dtype)
        projected.append((vector @ basis.T) @ basis)
    return projected


def orthogonal_vectors(vectors, bases):
    projected = project_vectors(vectors, bases)
    return [vector - projection for vector, projection in zip(vectors, projected)]


def coordinate_topk(vectors, k):
    output = []
    for vector in vectors:
        flat = vector.reshape(-1)
        keep = min(k, flat.numel())
        indices = torch.topk(flat.abs(), keep).indices
        changed = torch.zeros_like(flat)
        changed[indices] = flat[indices]
        output.append(changed.reshape_as(vector))
    return output


def random_basis(hidden, rank, seed, device):
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(hidden, rank, generator=generator, dtype=torch.float64)
    orthonormal, _ = torch.linalg.qr(matrix)
    return orthonormal.T.float().to(device)


def random_project_vectors(vectors, rank, seed, device):
    projected = []
    hidden = vectors[0].shape[-1]
    for layer_index, vector in enumerate(vectors):
        basis = random_basis(hidden, rank, seed + layer_index, device).to(vector.device, dtype=vector.dtype)
        projected.append((vector @ basis.T) @ basis)
    return projected


def energy(vectors, reference):
    numerator = sum(float((vector.float() ** 2).sum()) for vector in vectors)
    denominator = sum(float((vector.float() ** 2).sum()) for vector in reference)
    if denominator <= 0:
        raise ValueError("Reference vectors have zero energy")
    return numerator / denominator


def evaluate(harness, record, layers, vectors=None):
    if vectors is None:
        with torch.no_grad():
            logits = harness.model(record.corrupt_ids).logits[0, -1]
    else:
        vectors = [vector.to(harness.device) for vector in vectors]
        logits = forward_logits(harness, record.corrupt_ids, layers, vectors, record.dpos)
    probs = torch.softmax(logits.float(), dim=0)
    return {
        "target_probability": float(probs[record.cid]),
        "target_log_probability": float(torch.log(probs[record.cid].clamp_min(1e-45))),
        "target_margin": float(logits[record.cid] - torch.cat([logits[:record.cid], logits[record.cid + 1:]]).max()),
        "target_success": int(logits.argmax().item() == record.cid),
        "prediction": int(logits.argmax().item()),
    }


def summarize(rows):
    baseline = [row["baseline"]["target_success"] for row in rows]
    full = [row["full"]["target_success"] for row in rows]
    summary = {}
    for condition in rows[0]["conditions"]:
        flags = [row["conditions"][condition]["target_success"] for row in rows]
        probabilities = [row["conditions"][condition]["target_probability"] for row in rows]
        base_probabilities = [row["baseline"]["target_probability"] for row in rows]
        full_probabilities = [row["full"]["target_probability"] for row in rows]
        rate, interval = rate_ci(flags)
        delta, delta_ci = paired_delta_ci(baseline, flags)
        full_delta = float(np.mean(full_probabilities) - np.mean(base_probabilities))
        condition_delta = float(np.mean(probabilities) - np.mean(base_probabilities))
        summary[condition] = {
            "n": len(rows),
            "successes": int(sum(flags)),
            "success_rate": rate,
            "success_ci": interval,
            "success_delta_vs_baseline": delta,
            "success_delta_ci": delta_ci,
            "mean_target_probability": float(np.mean(probabilities)),
            "mean_probability_delta_vs_baseline": condition_delta,
            "probability_recovery_fraction_vs_full": condition_delta / full_delta if full_delta else None,
            "mean_energy_fraction": float(np.mean([row["conditions"][condition].get("energy_fraction", np.nan) for row in rows])),
        }
    summary["full_successes"] = int(sum(full))
    summary["baseline_successes"] = int(sum(baseline))
    return summary


def markdown(result):
    lines = ["# Causal Dimensionality Pilot", "", "Development-only diagnostic; not a confirmatory test.", ""]
    lines.extend(["| Condition | Success | Mean energy | Mean target-prob delta | Recovery vs full |",
                  "|---|---:|---:|---:|---:|"])
    for condition, values in result["summary"].items():
        if condition in ("full_successes", "baseline_successes"):
            continue
        recovery = values["probability_recovery_fraction_vs_full"]
        recovery_text = "n/a" if recovery is None else f"{100 * recovery:.1f}%"
        lines.append("| {condition} | {successes}/{n} | {energy:.1f}% | {delta:.4f} | {recovery} |".format(
            condition=condition,
            successes=values["successes"],
            n=values["n"],
            energy=100 * values["mean_energy_fraction"] if np.isfinite(values["mean_energy_fraction"]) else float("nan"),
            delta=values["mean_probability_delta_vs_baseline"],
            recovery=recovery_text,
        ))
    lines.extend(["", "## Interpretation Guardrails", ""])
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
    cases = load_source_cases(project, protocol, development)
    layers = read_json(project / protocol["source_run"] / "run.json")["comparison"]["layers"]
    rows = []
    rank_bases = {rank: fit_subspace_bases(harness, train, layers, rank) for rank in protocol["subspace_ranks"]}
    for case_index, item in enumerate(cases):
        record = item["record"]
        full_vectors = [vector.to(harness.device) for vector in item["vectors"]]
        baseline = evaluate(harness, record, layers)
        full = evaluate(harness, record, layers, full_vectors)
        conditions = {"full": {**full, "energy_fraction": 1.0}}
        for rank in protocol["subspace_ranks"]:
            tracking = project_vectors(full_vectors, rank_bases[rank])
            outside = orthogonal_vectors(full_vectors, rank_bases[rank])
            random = random_project_vectors(full_vectors, rank, 9000 + 100 * rank + case_index, harness.device)
            topk = coordinate_topk(full_vectors, rank)
            for name, vectors in ((f"tracking_projection_k{rank}", tracking),
                                  (f"tracking_orthogonal_k{rank}", outside),
                                  (f"random_projection_k{rank}", random),
                                  (f"coordinate_topk_k{rank}", topk)):
                conditions[name] = {**evaluate(harness, record, layers, vectors),
                                    "energy_fraction": energy(vectors, full_vectors)}
        permuted = cases[(case_index + 1) % len(cases)]["vectors"]
        conditions["permuted_full"] = {**evaluate(harness, record, layers, permuted),
                                       "energy_fraction": energy(permuted, item["vectors"])}
        rows.append({"sample_id": record.sample_id, "input": input_identity(record),
                     "baseline": baseline, "full": full, "conditions": conditions})
    result = {"completed": True, "protocol": protocol, "protocol_sha256": fingerprint(protocol),
              "device": harness.device, "n_development_pairs": len(rows), "rows": rows,
              "summary": summarize(rows), "elapsed_seconds": time.monotonic() - started,
              "reviewer_gap_status": "G1 remains open pending optimized subspace and untouched-test confirmation"}
    write_new(output / "summary.json", result)
    with (output / "RESULTS.md").open("x") as stream:
        stream.write(markdown(result))
    print(f"Saved causal dimensionality pilot -> {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", type=Path, default=Path(__file__).resolve().parents[1] / "protocols/causal_dimensionality_pilot_v1.json")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    run(args.project.resolve(), args.protocol.resolve(), args.device)


if __name__ == "__main__":
    main()