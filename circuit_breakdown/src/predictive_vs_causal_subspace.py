"""Utilities for equal-rank predictive-vs-causal subspace diagnostics."""
from __future__ import annotations

import math

import numpy as np
import torch

from audit_study import assert_matches
from experiment_metrics import paired_delta_ci, rate_ci


EXPECTED_RANKS = [1, 2, 4, 8, 16, 32, 64]
EXPECTED_SUBSPACES = {"tracking", "pca", "random", "causal"}


def validate_protocol(protocol):
    required = {
        "version": "predictive_vs_causal_subspace_v1",
        "model": "phi-3.5-mini",
        "task": "transfer",
        "training_seed": 1,
        "development_seed": 1,
        "development_indices": list(range(8)),
        "ranks": EXPECTED_RANKS,
    }
    for name, value in required.items():
        assert_matches(protocol[name], value, f"protocol.{name}")
    if set(protocol["subspaces"]) != EXPECTED_SUBSPACES:
        raise ValueError("Protocol must declare tracking, pca, random and causal subspaces")
    if "not_yet_executed" not in protocol["scope"] and "development_only" not in protocol["scope"]:
        raise ValueError("Predictive-vs-causal protocol must remain development-only")


def orthonormal_rows(matrix):
    if matrix.ndim != 2 or matrix.shape[0] < 1 or not torch.isfinite(matrix).all().item():
        raise ValueError("Finite two-dimensional matrix required")
    orthonormal, _ = torch.linalg.qr(matrix.detach().cpu().double().T)
    rows = orthonormal.T[:matrix.shape[0]].float()
    error = float((rows @ rows.T - torch.eye(len(rows))).abs().max())
    if error > 1e-5:
        raise ValueError("Could not construct an orthonormal row basis")
    return rows


def pca_basis(states, rank):
    if states.ndim != 2 or rank < 1 or not torch.isfinite(states).all().item():
        raise ValueError("Finite state matrix and positive rank required")
    centered = states.detach().cpu().double() - states.detach().cpu().double().mean(0, keepdim=True)
    _, singular, right = torch.linalg.svd(centered, full_matrices=False)
    tolerance = max(centered.shape) * torch.finfo(torch.float32).eps * singular[0].item() if len(singular) else 0.0
    actual = min(rank, int((singular > tolerance).sum().item()))
    if actual < 1:
        raise ValueError("PCA state matrix has no nonzero directions")
    return right[:actual].float(), {
        "requested_rank": rank,
        "basis_rank": actual,
        "explained_energy": float((singular[:actual] ** 2).sum() / (singular ** 2).sum()),
    }


def random_basis(hidden, rank, seed):
    if min(hidden, rank) < 1 or rank > hidden:
        raise ValueError("Random basis rank must be between 1 and hidden size")
    generator = torch.Generator().manual_seed(seed)
    return orthonormal_rows(torch.randn(rank, hidden, generator=generator, dtype=torch.float64))


def coordinate_basis_from_scores(scores, rank):
    if scores.ndim != 1 or rank < 1 or rank > scores.numel() or not torch.isfinite(scores).all().item():
        raise ValueError("Finite score vector and valid rank required")
    indices = torch.topk(scores.detach().cpu().float().abs(), rank).indices.sort().values
    basis = torch.zeros(rank, scores.numel())
    basis[torch.arange(rank), indices] = 1.0
    return basis, indices.tolist()


def project_vectors(vectors, bases):
    if len(vectors) != len(bases):
        raise ValueError("One basis is required per vector")
    output = []
    for vector, basis in zip(vectors, bases):
        if vector.ndim != 2 or vector.shape[0] != 1 or basis.ndim != 2 or basis.shape[1] != vector.shape[1]:
            raise ValueError("Expected vectors [1, hidden] and bases [rank, hidden]")
        basis = basis.to(vector.device, dtype=vector.dtype)
        output.append((vector @ basis.T) @ basis)
    return output


def energy_fraction(vectors, reference):
    if len(vectors) != len(reference):
        raise ValueError("Energy comparison requires matching vector lists")
    numerator = sum(float((vector.detach().float() ** 2).sum()) for vector in vectors)
    denominator = sum(float((vector.detach().float() ** 2).sum()) for vector in reference)
    if denominator <= 0:
        raise ValueError("Reference vector energy must be positive")
    return numerator / denominator


def centroid_probe_fit(clean_states, corrupt_states, bases):
    clean_projected = _project_states(clean_states, bases)
    corrupt_projected = _project_states(corrupt_states, bases)
    clean_center = clean_projected.mean(0)
    corrupt_center = corrupt_projected.mean(0)
    return {"clean_center": clean_center, "corrupt_center": corrupt_center}


def centroid_probe_accuracy(probe, clean_states, corrupt_states, bases):
    clean_projected = _project_states(clean_states, bases)
    corrupt_projected = _project_states(corrupt_states, bases)
    clean_flags = _nearest_center_flags(clean_projected, probe, 1)
    corrupt_flags = _nearest_center_flags(corrupt_projected, probe, 0)
    flags = clean_flags + corrupt_flags
    rate, interval = rate_ci(flags)
    return {"accuracy": rate, "accuracy_ci": interval, "n": len(flags), "correct": int(sum(flags))}


def _project_states(states, bases):
    if len(states) != len(bases):
        raise ValueError("One state matrix is required per basis")
    pieces = []
    for state, basis in zip(states, bases):
        if state.ndim != 2 or basis.ndim != 2 or state.shape[1] != basis.shape[1]:
            raise ValueError("Expected states [n, hidden] and bases [rank, hidden]")
        basis = basis.to(state.device, dtype=state.dtype)
        pieces.append(state @ basis.T)
    return torch.cat(pieces, dim=1)


def _nearest_center_flags(projected, probe, expected_clean):
    clean_distance = ((projected - probe["clean_center"].to(projected)) ** 2).sum(dim=1)
    corrupt_distance = ((projected - probe["corrupt_center"].to(projected)) ** 2).sum(dim=1)
    predicted_clean = clean_distance <= corrupt_distance
    return [int(value.item() == bool(expected_clean)) for value in predicted_clean]


def summarize_decode_control(rows):
    if not rows:
        raise ValueError("Cannot summarize empty rows")
    baseline_flags = [row["baseline"]["target_success"] for row in rows]
    baseline_probabilities = [row["baseline"]["target_probability"] for row in rows]
    full_probabilities = [row["full"]["target_probability"] for row in rows]
    full_delta = float(np.mean(full_probabilities) - np.mean(baseline_probabilities))
    output = {}
    for key in rows[0]["conditions"]:
        flags = [row["conditions"][key]["target_success"] for row in rows]
        probabilities = [row["conditions"][key]["target_probability"] for row in rows]
        rate, interval = rate_ci(flags)
        delta, delta_ci = paired_delta_ci(baseline_flags, flags)
        probability_delta = float(np.mean(probabilities) - np.mean(baseline_probabilities))
        recovery = probability_delta / full_delta if not math.isclose(full_delta, 0.0) else None
        output[key] = {
            "n": len(rows),
            "successes": int(sum(flags)),
            "success_rate": rate,
            "success_ci": interval,
            "success_delta_vs_baseline": delta,
            "success_delta_ci": delta_ci,
            "mean_target_probability": float(np.mean(probabilities)),
            "mean_probability_delta_vs_baseline": probability_delta,
            "probability_recovery_fraction_vs_full": recovery,
            "mean_energy_fraction": float(np.mean([row["conditions"][key].get("energy_fraction", np.nan) for row in rows])),
            "decodability": rows[0]["conditions"][key].get("decodability"),
        }
    return output