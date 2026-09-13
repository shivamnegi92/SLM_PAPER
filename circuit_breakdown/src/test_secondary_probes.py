"""Secondary probes: is the transported thing a REASONING STATE or a
compressed ANSWER TOKEN?

Validated so far: patching the donor's final-token residual installs the
donor's answer (DS_cross = +0.718), suppresses the receiver's own answer,
and requires a real model-state direction (norm-matched random is inert).
But the position sweep found ZERO effect at any position earlier than -1,
which is consistent with the final-token residual having already collapsed
into a near-linearly-decodable answer encoding. Specificity does not settle
WHAT is being transported.

The transfer task's causal program exposes more than one variable over the
same chain:

    queried_chain = [p0, p1, ..., pk]
    current holder  Z_curr = p_k     <- what we have tested so far
    original holder Z_orig = p_0     <- a DIFFERENT variable, same body text

TEST A (matched question, necessary precondition). Ask BOTH donor and
receiver "Who originally had the X?" and interchange. If the donor's
original-holder does not transport at all, the mechanism is specific to the
one question form and there is nothing further to test.

TEST B (cross-question, DECISIVE). Ask the DONOR "Who originally had the X?"
and the RECEIVER "Who has the X?", then patch donor -> receiver. The donor's
two answers differ (p0 vs pk), so the predictions come apart cleanly:

    output == donor_ORIGINAL holder  => the patch carried a literal ANSWER
                                        TOKEN for the donor's own question,
                                        ignoring the receiver's question.
                                        => answer compression.
    output == donor_CURRENT holder   => the patch carried a STATE, which the
                                        receiver's own question then queried
                                        correctly. => genuine reasoning state.
    output == receiver's own answer  => no effect.

A mixed few-shot prefix (both question forms) is used for every condition so
donor and receiver always see identical priming and the only difference is
the question actually asked.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "src")

import random as pyrandom

import numpy as np
import torch

from localize import Harness, pick_device
import dataset
import causal_interchange as ci
from interchange_dataset import _draw_problem

N_ITEMS = 24
ROUNDS = 2
LAYERS = [18, 20, 22, 24]
POSITION = -1
ITEM_SEED = 21
PREFIX_SEED = 99
N_FEWSHOT_PER_FORM = 3

CURRENT_QUESTION = "Who has the {object}?"
ORIGINAL_QUESTION = "Who originally had the {object}?"


def render_body(problem):
    """The transfer chain text WITHOUT any question, so the same body can be
    queried by either probe."""
    queried, distractor = problem["queried_chain"], problem["distractor_chain"]
    queried_object, distractor_object = problem["object"], problem["distractor_object"]
    lines = [f"{queried[0]} has the {queried_object}.",
             f"{distractor[0]} has the {distractor_object}."]
    for index in range(len(queried) - 1):
        lines.append(f"{queried[index]} gives the {queried_object} to {queried[index + 1]}.")
        lines.append(f"{distractor[index]} gives the {distractor_object} to {distractor[index + 1]}.")
    return " ".join(lines)


def render(problem, question_template):
    return render_body(problem) + " " + question_template.format(object=problem["object"])


def draw_disjoint_pair(rng, rounds):
    """Donor and receiver with fully disjoint names/objects, and with BOTH
    variables distinct across the pair so no probe is trivially satisfied."""
    need = 2 + 2 * rounds
    people = rng.sample(dataset.PEOPLE, 2 * need)
    objects = rng.sample(dataset.OBJECTS, 4)
    donor = _draw_problem(rng, people[:need], objects[:2], rounds)
    receiver = _draw_problem(rng, people[need:], objects[2:], rounds)
    donor_current, donor_original = donor["queried_chain"][-1], donor["queried_chain"][0]
    receiver_current = receiver["queried_chain"][-1]
    if len({donor_current, donor_original, receiver_current}) != 3:
        raise ValueError("Need donor-current, donor-original and receiver-current all distinct")
    return donor, receiver


def build_items(harness, n, seed, rounds):
    """Generate pairs, keeping only those whose three target names are all
    single tokens (so top-1 comparisons are exact)."""
    rng = pyrandom.Random(seed)
    items = []
    guard = 0
    while len(items) < n and guard < n * 400:
        guard += 1
        try:
            donor, receiver = draw_disjoint_pair(rng, rounds)
        except ValueError:
            continue
        targets = [donor["queried_chain"][-1], donor["queried_chain"][0],
                   receiver["queried_chain"][-1], receiver["queried_chain"][0]]
        if any(len(harness.tok(" " + name, add_special_tokens=False)["input_ids"]) != 1
               for name in targets):
            continue
        items.append((donor, receiver))
    if len(items) < n:
        raise RuntimeError(f"Only produced {len(items)}/{n} probe items")
    return items


def build_mixed_prefix(harness, seed, rounds, per_form):
    """Few-shot prefix containing BOTH question forms, so every condition sees
    identical priming and the question asked is the only difference."""
    rng = pyrandom.Random(seed)
    lines = []
    made = 0
    guard = 0
    while made < per_form * 2 and guard < 400:
        guard += 1
        need = 2 + 2 * rounds
        try:
            problem = _draw_problem(rng, rng.sample(dataset.PEOPLE, need),
                                    rng.sample(dataset.OBJECTS, 2), rounds)
        except ValueError:
            continue
        use_original = made % 2 == 1
        question = ORIGINAL_QUESTION if use_original else CURRENT_QUESTION
        answer = problem["queried_chain"][0] if use_original else problem["queried_chain"][-1]
        if len(harness.tok(" " + answer, add_special_tokens=False)["input_ids"]) != 1:
            continue
        lines.append(render(problem, question) + " " + answer + ".")
        made += 1
    return "\n".join(lines) + "\n"


def patched_prediction(harness, donor_text, receiver_text):
    """Interchange donor -> receiver at the final token, full space."""
    donor_ids = harness.encode(donor_text)
    receiver_ids = harness.encode(receiver_text)
    donor_state = ci.extract_activations(harness, LAYERS, donor_ids, POSITION)
    receiver_state = ci.extract_activations(harness, LAYERS, receiver_ids, POSITION)
    vectors = ci.interchange_vectors(donor_state, receiver_state, None)
    zero_vectors = [torch.zeros_like(v) for v in receiver_state]
    baseline = ci.predict(harness, LAYERS, receiver_ids, POSITION, zero_vectors)
    patched = ci.predict(harness, LAYERS, receiver_ids, POSITION, vectors)
    return baseline, patched


def main():
    harness = Harness("../llama-3.2-3b", pick_device("auto"))
    harness.model.requires_grad_(False)
    dataset.restrict_to_single_token(harness.tok)

    prefix = build_mixed_prefix(harness, PREFIX_SEED, ROUNDS, N_FEWSHOT_PER_FORM)
    print("mixed few-shot prefix:")
    for line in prefix.strip().split("\n"):
        print("   ", line[:110] + ("..." if len(line) > 110 else ""))
    print()

    items = build_items(harness, N_ITEMS, ITEM_SEED, ROUNDS)

    # ---------- sanity: can the model answer each probe unpatched? ----------
    current_ok = original_ok = 0
    for donor, receiver in items:
        for problem, counter in ((receiver, "current"), (receiver, "original")):
            question = CURRENT_QUESTION if counter == "current" else ORIGINAL_QUESTION
            expected = (problem["queried_chain"][-1] if counter == "current"
                        else problem["queried_chain"][0])
            ids = harness.encode(prefix + render(problem, question))
            state = ci.extract_activations(harness, LAYERS, ids, POSITION)
            prediction = ci.predict(harness, LAYERS, ids, POSITION,
                                    [torch.zeros_like(v) for v in state])["prediction"]
            if prediction == harness.first_id(expected):
                if counter == "current":
                    current_ok += 1
                else:
                    original_ok += 1
    print(f"=== unpatched baseline accuracy (n={len(items)}) ===")
    print(f"  'Who has the X?'            : {current_ok}/{len(items)}")
    print(f"  'Who originally had the X?' : {original_ok}/{len(items)}")
    if original_ok < len(items) * 0.5:
        print("  WARNING: the original-holder probe is weak unpatched; Test A/B are"
              " uninterpretable if the model cannot answer the question at all.")
    print()

    # ---------- TEST A: matched question, both asked ORIGINAL ----------
    transported = stayed = 0
    for donor, receiver in items:
        donor_text = prefix + render(donor, ORIGINAL_QUESTION)
        receiver_text = prefix + render(receiver, ORIGINAL_QUESTION)
        _, patched = patched_prediction(harness, donor_text, receiver_text)
        transported += int(patched["prediction"] == harness.first_id(donor["queried_chain"][0]))
        stayed += int(patched["prediction"] == harness.first_id(receiver["queried_chain"][0]))
    print(f"=== TEST A: matched question (both asked 'originally'), n={len(items)} ===")
    print(f"  output == donor's original holder (transported): {transported}/{len(items)}")
    print(f"  output == receiver's own original holder (unmoved): {stayed}/{len(items)}")
    print()

    # ---------- TEST B: cross-question, DECISIVE ----------
    as_token = as_state = unmoved = 0
    print(f"=== TEST B: cross-question (donor asked 'originally', receiver asked 'has'), n={len(items)} ===")
    print(f"  {'i':>2} {'d_orig':>8} {'d_curr':>8} {'r_curr':>8} {'output':>10} {'verdict':>18}")
    for index, (donor, receiver) in enumerate(items):
        donor_text = prefix + render(donor, ORIGINAL_QUESTION)
        receiver_text = prefix + render(receiver, CURRENT_QUESTION)
        _, patched = patched_prediction(harness, donor_text, receiver_text)

        donor_original = donor["queried_chain"][0]
        donor_current = donor["queried_chain"][-1]
        receiver_current = receiver["queried_chain"][-1]
        prediction = patched["prediction"]
        token = harness.tok.decode([prediction]).strip()

        if prediction == harness.first_id(donor_original):
            verdict, _ = "ANSWER TOKEN", as_token
            as_token += 1
        elif prediction == harness.first_id(donor_current):
            verdict = "REASONING STATE"
            as_state += 1
        elif prediction == harness.first_id(receiver_current):
            verdict = "unmoved"
            unmoved += 1
        else:
            verdict = "other"
        print(f"  {index:>2} {donor_original:>8} {donor_current:>8} {receiver_current:>8} "
              f"{token:>10} {verdict:>18}")

    n = len(items)
    print(f"\n  donor's ORIGINAL holder (answer-token transport): {as_token}/{n}")
    print(f"  donor's CURRENT holder  (reasoning-state transport): {as_state}/{n}")
    print(f"  receiver's own answer   (no effect): {unmoved}/{n}")
    print(f"  other: {n - as_token - as_state - unmoved}/{n}")


if __name__ == "__main__":
    main()
