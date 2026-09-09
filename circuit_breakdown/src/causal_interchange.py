"""Causal interchange intervention pilot.

The predictive-vs-causal steering pilot (v1/v2) asks: can an optimizer, given
the RECEIVER'S OWN target as its objective, push the receiver's residual
stream to produce that target? A reviewer can fairly object that the edit was
trained on the very answer it needs to produce.

This pilot asks a different, harder question: can a candidate subspace
TRANSPORT a donor's latent reasoning state into an unrelated receiver, with no
objective that knows the receiver's target at all? The intervention here is
determined entirely by another example's activations, not by gradient descent
against the receiver's own answer.

High-level causal variable (transfer task): Z = current holder of the object.
Donor A has Z_A, receiver B has Z_B (disjoint names/objects by construction;
see interchange_dataset.py). The interchange intervention for a candidate
subspace S is:

    v_S = P_S(h_A[pos_A] - h_B[pos_B])
    h_B[pos_B] <- h_B[pos_B] + v_S

which is exactly the existing additive hook mechanism in `intervene_pareto`
with the edit vector set to the (optionally subspace-restricted) donor-minus-
receiver difference. S = the full space gives an exact replacement of the
receiver's activation with the donor's at that position -- the oracle upper
bound. S = a candidate basis tests whether that basis alone is sufficient to
carry the transplant.

Position: the token position of the receiver's/donor's OWN final-recipient
name (the token that establishes "current holder = X" in each one's own
context), found by verified tokenization of a strict text prefix -- not by
word-count, which is fragile under BPE merges.

Controls:
  - random donor: pair each receiver with a DIFFERENT item's donor instead of
    its constructed match. If interchange succeeds equally well with an
    unrelated donor, the mechanism is not transporting a specific state.
  - same_answer items: donor and receiver already agree on Z (via different
    reasoning paths). There is no answer to flip here; the informative
    quantities are the raw-activation cosine similarity at the two positions
    (does an abstract, path-invariant representation align?) and whether the
    patch is a behavioral no-op (patching should not corrupt an
    already-correct receiver when the states already agree).

Development-only diagnostic; not a confirmatory test.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from intervene_pareto import forward_logits
from predictive_vs_causal_subspace import project_vectors


def structural_position(harness, prefix, item_text, name):
    """Token index, WITHIN THE PREFIXED TEXT (prefix + item_text), of the last
    token of `name` in the unique substring 'to {name}.' inside item_text.

    The marker is located within item_text alone (not the combined string) so
    a coincidental 'to <name>.' inside a few-shot prefix built from the same
    name pool cannot cause an ambiguous match. Verified via strict-prefix
    tokenization of the full combined text. A few-shot prefix must always
    precede the model here, matching the convention used everywhere else in
    this project (see `pilot_records` in run_convergence_pilot.py) -- a base
    model was not reliable zero-shot on this task format in testing.
    """
    marker = f"to {name}."
    if item_text.count(marker) != 1:
        raise ValueError(f"Expected exactly one occurrence of {marker!r} in item_text, found {item_text.count(marker)}")
    local_char_index = item_text.index(marker)
    full_text = prefix + item_text
    prefix_text = full_text[:len(prefix) + local_char_index + len(marker) - 1]
    prefix_ids = harness.encode(prefix_text)
    full_ids = harness.encode(full_text)
    if not torch.equal(full_ids[0][:prefix_ids.shape[1]], prefix_ids[0]):
        raise ValueError("Prefix tokenization is not a strict prefix of the full text; position invalid")
    return prefix_ids.shape[1] - 1, full_text


def extract_activations(harness, layers, ids, position):
    """[1, hidden] residual-stream activation at `position`, one per layer,
    from a single forward pass."""
    outputs = []
    handles = []

    def hook(module, inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        outputs.append(tensor[:, position, :].detach().clone())

    for layer_index in layers:
        handles.append(harness.layers[layer_index].register_forward_hook(hook))
    try:
        with torch.no_grad():
            harness.model(ids)
    finally:
        for handle in handles:
            handle.remove()
    return outputs


def interchange_vectors(donor_state, receiver_state, basis=None):
    """v = P_S(h_donor - h_receiver) per layer; S = full space if basis is None."""
    if len(donor_state) != len(receiver_state):
        raise ValueError("Donor and receiver activations must cover the same layers")
    deltas = [(d - r) for d, r in zip(donor_state, receiver_state)]
    if basis is None:
        return deltas
    return project_vectors(deltas, basis)


def predict(harness, layers, ids, position, vectors):
    """Full-vocabulary prediction after applying `vectors` at `position`."""
    with torch.no_grad():
        logits = forward_logits(harness, ids, layers, vectors, position)
        probabilities = torch.softmax(logits.float(), dim=0)
    return {"prediction": int(logits.argmax().item()), "probabilities": probabilities}


def evaluate_disjoint_item(harness, layers, item, basis, name, first_id_fn, prefix=""):
    """One donor-to-receiver interchange for a `disjoint` item under one basis.

    Patches at the FINAL token position (right before the answer is
    generated), not the mid-sentence recipient-name position. Empirically
    validated before committing to a run: at the mid-sentence position the
    oracle (full-space) interchange scored 0/8 even with a healthy baseline;
    at the final position, with the SAME 4-layer set used throughout this
    project, the oracle scores 6/8. The final position is where the network
    has to have resolved state tracking into an answer-ready representation,
    so it is the position a transplanted state has the best chance of
    actually overriding. `prefix` is the same few-shot text used everywhere
    else in this project; a base model was not reliable zero-shot on this
    task format in testing.
    """
    donor_ids = harness.encode(prefix + item.donor_prompt)
    receiver_ids = harness.encode(prefix + item.receiver_prompt)
    donor_state = extract_activations(harness, layers, donor_ids, -1)
    receiver_state = extract_activations(harness, layers, receiver_ids, -1)
    vectors = interchange_vectors(donor_state, receiver_state, basis)

    zero_vectors = [torch.zeros_like(v) for v in receiver_state]
    baseline = predict(harness, layers, receiver_ids, -1, zero_vectors)
    patched = predict(harness, layers, receiver_ids, -1, vectors)

    donor_id = first_id_fn(item.donor_answer)
    receiver_id = first_id_fn(item.receiver_answer)
    return {
        "basis": name,
        "donor_answer": item.donor_answer, "receiver_answer": item.receiver_answer,
        "baseline_prediction": baseline["prediction"],
        "baseline_matches_receiver": int(baseline["prediction"] == receiver_id),
        "patched_prediction": patched["prediction"],
        "interchange_success": int(patched["prediction"] == donor_id),
        "still_receiver_answer": int(patched["prediction"] == receiver_id),
        "p_donor_answer_baseline": float(baseline["probabilities"][donor_id]),
        "p_donor_answer_patched": float(patched["probabilities"][donor_id]),
        "edit_norm": float(sum(v.float().norm() ** 2 for v in vectors) ** 0.5),
    }


def evaluate_same_answer_item(harness, layers, item, basis, name, prefix=""):
    """Donor and receiver already agree on Z via different reasoning paths.

    Reports the raw-state cosine similarity (does an abstract representation
    align despite different surface routes?) and whether patching corrupts an
    already-correct receiver (it should not, if the states already agree).
    Same final-token position as `evaluate_disjoint_item`.
    """
    donor_ids = harness.encode(prefix + item.donor_prompt)
    receiver_ids = harness.encode(prefix + item.receiver_prompt)
    donor_state = extract_activations(harness, layers, donor_ids, -1)
    receiver_state = extract_activations(harness, layers, receiver_ids, -1)
    vectors = interchange_vectors(donor_state, receiver_state, basis)

    flat_donor = torch.cat([v.reshape(-1) for v in donor_state])
    flat_receiver = torch.cat([v.reshape(-1) for v in receiver_state])
    cosine = float((flat_donor @ flat_receiver) / (flat_donor.norm() * flat_receiver.norm() + 1e-12))

    zero_vectors = [torch.zeros_like(v) for v in receiver_state]
    baseline = predict(harness, layers, receiver_ids, -1, zero_vectors)
    patched = predict(harness, layers, receiver_ids, -1, vectors)
    return {
        "basis": name, "shared_answer": item.receiver_answer,
        "raw_state_cosine_donor_vs_receiver": cosine,
        "baseline_prediction": baseline["prediction"], "patched_prediction": patched["prediction"],
        "edit_norm": float(sum(v.float().norm() ** 2 for v in vectors) ** 0.5),
    }
