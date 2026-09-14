"""Cross-question causal validation, run on either architecture.

Established on Llama: the final-token interchange is behaviorally effective,
donor-specific (DS_cross = +0.718), inert under a norm-matched random
direction, and graded in magnitude -- yet a cross-question probe showed it
transports the donor's ANSWER, not the donor's reasoning state (24/24 vs
0/24).

This script generalizes that to (a) any model and (b) a panel of probes that
separates COMPLETENESS from SELECTIVITY, which single-answer IIA cannot.

INTERVENTION. The donor is always asked the canonical question
"Who has the {object}?", so its final-token residual corresponds to
do(Z_curr := z'_D) where z'_D is the donor's current holder. That single
donor state is patched into receivers that were asked DIFFERENT questions.

CAUSAL PREDICTIONS under do(Z_curr := z'_D) on the receiver's own problem:

  probe              question                                should change?
  current_holder     Who has the {object}?                   YES  -> z'_D
  last_recipient     Who received the {object} last?         YES  -> z'_D
  original_holder    Who originally had the {object}?        NO   -> receiver's r0
  object_identity    What object did {r0} have at the start? NO   -> receiver's object
  transfer_count     How many times was the {object} given   NO   -> receiver's count
                     away?

Completeness = the YES probes move to the donor's value.
Selectivity   = the NO probes stay at the receiver's own value.

An intervention that merely installs an answer token will violate
SELECTIVITY spectacularly: asked "what object?", it answers a person's name.
Note that a selectivity failure is only interpretable on probes the model can
answer UNPATCHED, so per-probe baseline competence is measured and reported
first, and probes below a competence floor are excluded from the verdict.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import random as pyrandom

import torch

from localize import Harness, pick_device
import dataset
import causal_interchange as ci
from interchange_dataset import _draw_problem

LAYERS = [18, 20, 22, 24]
POSITION = -1
ROUNDS = 2
COMPETENCE_FLOOR = 0.5

# Count answers are rendered as words: " 2" is TWO tokens in Llama, which
# silently rejected every generated pair until caught.
COUNT_WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

CANONICAL_QUESTION = "Who has the {object}?"

# probe -> (question template, needs_first_holder_name, should_change)
PROBES = {
    "current_holder": ("Who has the {object}?", False, True),
    "last_recipient": ("Who received the {object} last?", False, True),
    "original_holder": ("Who originally had the {object}?", False, False),
    "object_identity": ("What object did {first} have at the start?", True, False),
    "transfer_count": ("How many times was the {object} given away?", False, False),
}


def render_body(problem):
    """Chain text with no question, so one body can serve every probe."""
    queried, distractor = problem["queried_chain"], problem["distractor_chain"]
    queried_object, distractor_object = problem["object"], problem["distractor_object"]
    lines = [f"{queried[0]} has the {queried_object}.",
             f"{distractor[0]} has the {distractor_object}."]
    for index in range(len(queried) - 1):
        lines.append(f"{queried[index]} gives the {queried_object} to {queried[index + 1]}.")
        lines.append(f"{distractor[index]} gives the {distractor_object} to {distractor[index + 1]}.")
    return " ".join(lines)


def render(problem, template):
    return (render_body(problem) + " "
            + template.format(object=problem["object"], first=problem["queried_chain"][0]))


def unpatched_answer(problem, probe, rounds):
    """What the causal program says, with no intervention."""
    if probe in ("current_holder", "last_recipient"):
        return problem["queried_chain"][-1]
    if probe == "original_holder":
        return problem["queried_chain"][0]
    if probe == "object_identity":
        return problem["object"]
    if probe == "transfer_count":
        return COUNT_WORDS[rounds]
    raise ValueError(probe)


def single_token(harness, text):
    """Does `text` contribute exactly one token IN CONTEXT?

    Must mirror Harness.first_id's in-context convention rather than testing
    " " + text: SentencePiece models (Phi) emit a separate space token for the
    naive form, so every candidate looks like 2 tokens and all generation is
    silently rejected.
    """
    base = harness.tok("the", add_special_tokens=False)["input_ids"]
    extended = harness.tok("the " + text, add_special_tokens=False)["input_ids"]
    return len(extended) == len(base) + 1


def build_pairs(harness, n, seed, rounds):
    """Disjoint donor/receiver pairs whose every probe target is single-token."""
    rng = pyrandom.Random(seed)
    need = 2 + 2 * rounds
    pairs = []
    guard = 0
    while len(pairs) < n and guard < n * 500:
        guard += 1
        try:
            people = rng.sample(dataset.PEOPLE, 2 * need)
            objects = rng.sample(dataset.OBJECTS, 4)
            donor = _draw_problem(rng, people[:need], objects[:2], rounds)
            receiver = _draw_problem(rng, people[need:], objects[2:], rounds)
        except ValueError:
            continue
        donor_current = donor["queried_chain"][-1]
        targets = [donor_current, receiver["queried_chain"][-1],
                   receiver["queried_chain"][0], receiver["object"], COUNT_WORDS[rounds]]
        if len({donor_current, receiver["queried_chain"][-1], receiver["queried_chain"][0]}) != 3:
            continue
        if any(not single_token(harness, value) for value in targets):
            continue
        pairs.append((donor, receiver))
    if len(pairs) < n:
        raise RuntimeError(f"Only produced {len(pairs)}/{n} pairs")
    return pairs


def build_prefix(harness, seed, rounds, per_probe):
    """Few-shot prefix covering EVERY probe form, so no probe is disadvantaged
    by unfamiliar formatting and priming is identical across conditions."""
    rng = pyrandom.Random(seed)
    need = 2 + 2 * rounds
    lines = []
    for probe, (template, _, _) in PROBES.items():
        made = guard = 0
        while made < per_probe and guard < 500:
            guard += 1
            try:
                problem = _draw_problem(rng, rng.sample(dataset.PEOPLE, need),
                                        rng.sample(dataset.OBJECTS, 2), rounds)
            except ValueError:
                continue
            answer = unpatched_answer(problem, probe, rounds)
            if not single_token(harness, answer):
                continue
            lines.append(render(problem, template) + " " + answer + ".")
            made += 1
    rng.shuffle(lines)
    return "\n".join(lines) + "\n"


def predict_pair(harness, donor_text, receiver_text):
    donor_ids = harness.encode(donor_text)
    receiver_ids = harness.encode(receiver_text)
    donor_state = ci.extract_activations(harness, LAYERS, donor_ids, POSITION)
    receiver_state = ci.extract_activations(harness, LAYERS, receiver_ids, POSITION)
    vectors = ci.interchange_vectors(donor_state, receiver_state, None)
    zeros = [torch.zeros_like(v) for v in receiver_state]
    baseline = ci.predict(harness, LAYERS, receiver_ids, POSITION, zeros)
    patched = ci.predict(harness, LAYERS, receiver_ids, POSITION, vectors)
    return baseline, patched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llama-3.2-3b")
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--prefix-seed", type=int, default=99)
    parser.add_argument("--fewshot-per-probe", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    harness = Harness(str(project.parent / args.model), pick_device(args.device))
    harness.model.requires_grad_(False)
    dataset.restrict_to_single_token(harness.tok)
    print(f"model={args.model}  layers={LAYERS}  position={POSITION}  n={args.n}\n")

    prefix = build_prefix(harness, args.prefix_seed, ROUNDS, args.fewshot_per_probe)
    pairs = build_pairs(harness, args.n, args.seed, ROUNDS)

    # ---------- per-probe unpatched competence ----------
    print("=== unpatched baseline competence (gates interpretation) ===")
    competence = {}
    for probe, (template, _, _) in PROBES.items():
        correct = 0
        for _, receiver in pairs:
            expected = unpatched_answer(receiver, probe, ROUNDS)
            ids = harness.encode(prefix + render(receiver, template))
            state = ci.extract_activations(harness, LAYERS, ids, POSITION)
            prediction = ci.predict(harness, LAYERS, ids, POSITION,
                                    [torch.zeros_like(v) for v in state])["prediction"]
            correct += int(prediction == harness.first_id(expected))
        competence[probe] = correct / len(pairs)
        flag = "" if competence[probe] >= COMPETENCE_FLOOR else "   <- BELOW FLOOR, excluded"
        print(f"  {probe:>16}: {correct}/{len(pairs)} ({competence[probe]:.0%}){flag}")

    # ---------- cross-question panel ----------
    print(f"\n=== cross-question panel: donor asked '{CANONICAL_QUESTION}', "
          f"patched into receivers asked each probe ===")
    print(f"  {'probe':>16} {'should change?':>14} {'-> donor value':>15} "
          f"{'-> receiver value':>18} {'-> other':>9}  verdict")

    results = {}
    for probe, (template, _, should_change) in PROBES.items():
        to_donor = to_receiver = to_other = 0
        for donor, receiver in pairs:
            donor_text = prefix + render(donor, CANONICAL_QUESTION)
            receiver_text = prefix + render(receiver, template)
            _, patched = predict_pair(harness, donor_text, receiver_text)
            prediction = patched["prediction"]

            donor_value = donor["queried_chain"][-1]
            receiver_value = unpatched_answer(receiver, probe, ROUNDS)
            if prediction == harness.first_id(donor_value):
                to_donor += 1
            elif prediction == harness.first_id(receiver_value):
                to_receiver += 1
            else:
                to_other += 1

        n = len(pairs)
        results[probe] = {"to_donor": to_donor, "to_receiver": to_receiver, "to_other": to_other}
        if competence[probe] < COMPETENCE_FLOOR:
            verdict = "(excluded)"
        elif should_change:
            verdict = "COMPLETE" if to_donor > n / 2 else "incomplete"
        else:
            verdict = "SELECTIVE" if to_receiver > n / 2 else "SELECTIVITY FAILURE"
        print(f"  {probe:>16} {str(should_change):>14} {to_donor:>10}/{n:<4} "
              f"{to_receiver:>13}/{n:<4} {to_other:>4}/{n:<4}  {verdict}")

    # ---------- headline ----------
    graded = [p for p in PROBES if competence[p] >= COMPETENCE_FLOOR]
    changers = [p for p in graded if PROBES[p][2]]
    keepers = [p for p in graded if not PROBES[p][2]]
    n = len(pairs)
    complete = sum(results[p]["to_donor"] for p in changers)
    selective = sum(results[p]["to_receiver"] for p in keepers)
    print(f"\n  COMPLETENESS (should-change probes moved to donor): "
          f"{complete}/{len(changers) * n}" if changers else "  COMPLETENESS: no gradeable probes")
    print(f"  SELECTIVITY (should-NOT-change probes held receiver value): "
          f"{selective}/{len(keepers) * n}" if keepers else "  SELECTIVITY: no gradeable probes")
    leaked = sum(results[p]["to_donor"] for p in keepers)
    print(f"  leakage (should-NOT-change probes overwritten by donor value): "
          f"{leaked}/{len(keepers) * n}")


if __name__ == "__main__":
    main()
