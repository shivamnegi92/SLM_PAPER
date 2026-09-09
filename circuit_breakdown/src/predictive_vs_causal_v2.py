"""Utilities for predictive-vs-causal subspace diagnostics, v2.

Extends `predictive_vs_causal_subspace.py` (v1) with three changes requested
after reviewing the v1 Phi pilot:

1. RENAMED "causal" -> "gradient_coordinate". Raw dimensions selected by
   gradient sensitivity test a fairly narrow hypothesis (individual
   coordinates are causally meaningful). A distributed causal direction
   u = a_1 e_1 + a_2 e_2 + ... does not require any single e_i to be
   individually sensitive, so this basis is a weak comparator for causal
   control and should not be called "causal" without inviting exactly that
   objection.

2. ADDED "learned_causal": a DAS-lite basis. Instead of selecting raw
   coordinates or an activation-variance basis, directly optimize a shared
   per-layer orthonormal rotation U (rank x hidden) so that per-example
   coefficients within it can minimize the real task loss, jointly across a
   batch of training examples:

       U* = argmin_{U, {theta_i}} sum_i CE(logits(h_i + theta_i @ U), target_i)
       subject to U^T U = I

   This is closer to Distributed Alignment Search's idea of searching over
   rotated distributed representations than raw coordinate selection is.

3. ADDED degeneracy diagnostics: cosine similarity, energy ratios, and an
   intervention-output KL divergence, to distinguish "the known successful
   full-space solution lies in this subspace" from "this subspace supports an
   independently-discoverable, functionally similar solution."
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from intervene_pareto import forward_logits
from predictive_vs_causal_subspace import orthonormal_rows


def gradient_coordinate_scores(harness, records, layers):
    """Mean |gradient| of the loss w.r.t. a zero edit at each raw dimension.

    Renamed from the earlier `causal_coordinate_scores`. This tests whether
    INDIVIDUAL raw coordinates are locally sensitive -- a narrower question
    than whether a rotated combination of coordinates is causally relevant.
    """
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


def learned_causal_basis(harness, train_records, layers, rank, cfg):
    """Fit a shared per-layer orthonormal rotation directly from the
    behavioral objective (DAS-lite), jointly over a batch of training
    examples, each with its own coefficient vector.

    Orthogonality is maintained by a QR retraction after every optimizer
    step (project back onto the Stiefel manifold), which is the standard
    practical approach for orthogonally-constrained optimization when an
    exact Riemannian optimizer is not warranted for a small pilot.

    Uses the SAME convergence-based stopping rule as
    `intervene_pareto.optimize_sample_converged`, so ranks are compared under
    equal optimization EFFORT rather than equal step count -- the fix for
    Gap G1's step-budget confound applies here too.
    """
    device = harness.device
    hidden = harness.model.config.hidden_size
    n = len(train_records)
    if n < 1:
        raise ValueError("learned_causal_basis requires at least one training example")

    generator = torch.Generator().manual_seed(cfg.get("seed", 0))
    bases = [orthonormal_rows(torch.randn(rank, hidden, generator=generator, dtype=torch.float64)).to(device)
             for _ in layers]
    for basis in bases:
        basis.requires_grad_(True)
    coefficients = [[torch.zeros(1, rank, device=device, requires_grad=True) for _ in layers]
                    for _ in train_records]

    params = list(bases) + [theta for row in coefficients for theta in row]
    optimizer = torch.optim.Adam(params, lr=cfg["lr"])

    eps = cfg["convergence_eps"]
    patience = int(cfg["convergence_patience"])
    max_steps = int(cfg["max_steps"])
    if eps <= 0 or patience < 1 or max_steps < 1:
        raise ValueError("convergence_eps, convergence_patience, max_steps must be positive")

    previous_loss = None
    stable_steps = 0
    converged = False
    steps_taken = 0

    for step in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for example_index, record in enumerate(train_records):
            vectors = [coefficients[example_index][layer_index] @ bases[layer_index]
                       for layer_index in range(len(layers))]
            logits = forward_logits(harness, record.corrupt_ids, layers, vectors, record.dpos)
            total_loss = total_loss + (-F.log_softmax(logits.float(), dim=0)[record.cid])
        total_loss = total_loss / n
        if not torch.isfinite(total_loss).item():
            raise ValueError("Non-finite learned-causal joint loss")
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            for layer_index in range(len(layers)):
                orthonormal, _ = torch.linalg.qr(bases[layer_index].detach().cpu().double().T)
                bases[layer_index].copy_(orthonormal.T[:rank].float().to(device))

        loss_value = float(total_loss.detach().item())
        steps_taken = step + 1
        if previous_loss is not None and abs(previous_loss - loss_value) < eps:
            stable_steps += 1
            if stable_steps >= patience:
                converged = True
                break
        else:
            stable_steps = 0
        previous_loss = loss_value

    diagnostics = {"converged": converged, "steps_taken": steps_taken,
                   "max_steps": max_steps, "final_joint_loss": previous_loss,
                   "n_training_examples": n, "rank": rank}
    return [basis.detach() for basis in bases], diagnostics


def cosine_similarity_across_layers(vectors_a, vectors_b):
    """Flatten and concatenate edit vectors across layers, then cosine similarity."""
    if len(vectors_a) != len(vectors_b):
        raise ValueError("Vector lists must have matching layer counts")
    flat_a = torch.cat([v.detach().reshape(-1) for v in vectors_a])
    flat_b = torch.cat([v.detach().reshape(-1) for v in vectors_b])
    denominator = float(flat_a.norm() * flat_b.norm())
    if denominator <= 0:
        return None
    return float((flat_a @ flat_b) / denominator)


def norm_ratio(vectors_a, vectors_b):
    """||vectors_a|| / ||vectors_b||, flattened and concatenated across layers."""
    flat_a = torch.cat([v.detach().reshape(-1) for v in vectors_a])
    flat_b = torch.cat([v.detach().reshape(-1) for v in vectors_b])
    denominator = float(flat_b.norm())
    if denominator <= 0:
        return None
    return float(flat_a.norm() / denominator)


def intervention_kl(harness, layers, vectors_a, vectors_b, record):
    """KL(p(y | h + vectors_a) || p(y | h + vectors_b)) over the full
    vocabulary at the query position, both interventions applied to the SAME
    corrupt prompt at the same edited position.

    If v_full and v_basis are geometrically dissimilar (low cosine
    similarity) yet this KL is small, that is direct evidence of
    intervention degeneracy: multiple geometrically distinct edits producing
    functionally similar output behavior.
    """
    with torch.no_grad():
        logits_a = forward_logits(harness, record.corrupt_ids, layers, vectors_a, record.dpos)
        logits_b = forward_logits(harness, record.corrupt_ids, layers, vectors_b, record.dpos)
        log_p_a = F.log_softmax(logits_a.float(), dim=-1)
        log_p_b = F.log_softmax(logits_b.float(), dim=-1)
        return float((log_p_a.exp() * (log_p_a - log_p_b)).sum())


def degeneracy_diagnostics(harness, layers, vectors_full, vectors_basis, record):
    """Bundle the four diagnostics requested for double-success cases."""
    return {
        "cosine_full_vs_basis": cosine_similarity_across_layers(vectors_full, vectors_basis),
        "basis_norm_over_full_norm": norm_ratio(vectors_basis, vectors_full),
        "kl_full_intervention_vs_basis_intervention": intervention_kl(
            harness, layers, vectors_full, vectors_basis, record),
    }
