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
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import random as pyrandom

import torch

from localize import Harness, pick_device
from experiment_metrics import rate_ci
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


def run_one_seed(harness, n, item_seed, prefix_seed, fewshot_per_probe, verbose=True):
    """One independent replicate: its own sampled pairs AND its own sampled
    few-shot prefix. The intervention itself is deterministic (no optimizer,
    eval mode, patch = h_donor - h_receiver), so pairs and prefix are the ONLY
    stochastic components -- and the prefix is a real confound worth varying.
    Returns per-pair binary outcomes so Wilson intervals can be computed over
    INDEPENDENT PAIRS rather than over probe-outcomes.
    """
    prefix = build_prefix(harness, prefix_seed, ROUNDS, fewshot_per_probe)
    pairs = build_pairs(harness, n, item_seed, ROUNDS)

    competence = {}
    for probe, (template, _, _) in PROBES.items():
        flags = []
        for _, receiver in pairs:
            expected = unpatched_answer(receiver, probe, ROUNDS)
            ids = harness.encode(prefix + render(receiver, template))
            state = ci.extract_activations(harness, LAYERS, ids, POSITION)
            prediction = ci.predict(harness, LAYERS, ids, POSITION,
                                    [torch.zeros_like(v) for v in state])["prediction"]
            flags.append(int(prediction == harness.first_id(expected)))
        competence[probe] = flags

    outcomes = {}
    for probe, (template, _, _) in PROBES.items():
        to_donor, to_receiver = [], []
        for donor, receiver in pairs:
            donor_text = prefix + render(donor, CANONICAL_QUESTION)
            receiver_text = prefix + render(receiver, template)
            _, patched = predict_pair(harness, donor_text, receiver_text)
            prediction = patched["prediction"]
            donor_value = donor["queried_chain"][-1]
            receiver_value = unpatched_answer(receiver, probe, ROUNDS)
            to_donor.append(int(prediction == harness.first_id(donor_value)))
            to_receiver.append(int(prediction == harness.first_id(receiver_value)))
        outcomes[probe] = {"to_donor": to_donor, "to_receiver": to_receiver}
        if verbose:
            print(f"    {probe:>16}: ->donor {sum(to_donor):>3}/{len(pairs)}  "
                  f"->receiver {sum(to_receiver):>3}/{len(pairs)}  "
                  f"(baseline {sum(competence[probe])}/{len(pairs)})", flush=True)
    return {"n_pairs": len(pairs), "competence": competence, "outcomes": outcomes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llama-3.2-3b")
    parser.add_argument("--n", type=int, default=40, help="independent pairs PER SEED")
    parser.add_argument("--seeds", default="21,22,23",
                        help="comma-separated; each varies BOTH pairs and prefix")
    parser.add_argument("--fewshot-per-probe", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None,
                        help="optional JSON path for the headline figure")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    harness = Harness(str(project.parent / args.model), pick_device(args.device))
    harness.model.requires_grad_(False)
    dataset.restrict_to_single_token(harness.tok)
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"model={args.model}  layers={LAYERS}  position={POSITION}")
    print(f"seeds={seeds}  pairs_per_seed={args.n}  "
          f"total_independent_pairs={len(seeds) * args.n}\n", flush=True)

    replicates = []
    for seed in seeds:
        print(f"  --- seed {seed} (pairs and prefix both resampled) ---", flush=True)
        replicates.append(run_one_seed(harness, args.n, seed, seed * 7 + 1,
                                       args.fewshot_per_probe))

    # ---------- pooled across seeds, per MODEL only (never across models) ----------
    total_pairs = sum(r["n_pairs"] for r in replicates)
    print(f"\n=== {args.model}: pooled over {len(seeds)} seeds, "
          f"{total_pairs} INDEPENDENT PAIRS ===")
    print("(Wilson 95% intervals over independent pairs, not over probe-outcomes)\n")

    competence_rates = {}
    print(f"  {'probe':>16} {'baseline competence':>28}")
    for probe in PROBES:
        flags = [f for r in replicates for f in r["competence"][probe]]
        rate, (low, high) = rate_ci(flags)
        competence_rates[probe] = rate
        flag = "" if rate >= COMPETENCE_FLOOR else "  <- EXCLUDED"
        print(f"  {probe:>16} {sum(flags):>4}/{len(flags):<4} "
              f"{rate:>6.1%} [{low:.1%}, {high:.1%}]{flag}")

    print(f"\n  {'probe':>16} {'change?':>8} {'-> donor value':>26} {'-> receiver value':>26}")
    graded = []
    for probe, (_, _, should_change) in PROBES.items():
        donor_flags = [f for r in replicates for f in r["outcomes"][probe]["to_donor"]]
        receiver_flags = [f for r in replicates for f in r["outcomes"][probe]["to_receiver"]]
        d_rate, (d_low, d_high) = rate_ci(donor_flags)
        r_rate, (r_low, r_high) = rate_ci(receiver_flags)
        excluded = competence_rates[probe] < COMPETENCE_FLOOR
        if not excluded:
            graded.append((probe, should_change, donor_flags, receiver_flags))
        mark = " (excl.)" if excluded else ""
        print(f"  {probe:>16} {str(should_change):>8} "
              f"{sum(donor_flags):>4}/{len(donor_flags):<4} {d_rate:>5.1%} [{d_low:.1%},{d_high:.1%}] "
              f"{sum(receiver_flags):>4}/{len(receiver_flags):<4} {r_rate:>5.1%} [{r_low:.1%},{r_high:.1%}]{mark}")

    print(f"\n  --- per-seed stability (selectivity-relevant probes) ---")
    for probe, should_change, _, _ in graded:
        if should_change:
            continue
        per_seed = [f"{sum(r['outcomes'][probe]['to_receiver'])}/{r['n_pairs']}"
                    for r in replicates]
        print(f"    {probe:>16} -> receiver value by seed: {', '.join(per_seed)}")

    changers = [g for g in graded if g[1]]
    keepers = [g for g in graded if not g[1]]
    if keepers:
        selectivity_flags = [f for _, _, _, rf in keepers for f in rf]
        rate, (low, high) = rate_ci(selectivity_flags)
        print(f"\n  SELECTIVITY (should-NOT-change probes holding receiver value): "
              f"{sum(selectivity_flags)}/{len(selectivity_flags)} = {rate:.1%} [{low:.1%}, {high:.1%}]")
    if changers:
        completeness_flags = [f for _, _, df, _ in changers for f in df]
        rate, (low, high) = rate_ci(completeness_flags)
        print(f"  COMPLETENESS (should-change probes moving to donor): "
              f"{sum(completeness_flags)}/{len(completeness_flags)} = {rate:.1%} [{low:.1%}, {high:.1%}]")
    print(f"\n  REPORTING NOTE: independent reasoning pairs = {total_pairs}; "
          f"graded probes = {len(graded)}; "
          f"probe-outcomes = {total_pairs * len(graded)}. "
          f"Intervals above are over PAIRS within each probe.")

    if args.out:
        summary = {"models": {}}
        if args.out.exists():  # merge, so both models land in one file
            with args.out.open() as stream:
                summary = json.load(stream)
        entry = {"n_independent_pairs": total_pairs, "seeds": seeds,
                 "graded_probes": [g[0] for g in graded]}
        if keepers:
            flags = [f for _, _, _, rf in keepers for f in rf]
            rate, interval = rate_ci(flags)
            entry["selectivity"] = {"successes": int(sum(flags)), "n": len(flags),
                                    "rate": rate, "ci": list(interval)}
        if changers:
            flags = [f for _, _, df, _ in changers for f in df]
            rate, interval = rate_ci(flags)
            entry["completeness"] = {"successes": int(sum(flags)), "n": len(flags),
                                     "rate": rate, "ci": list(interval)}
        summary["models"][args.model] = entry
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as stream:
            json.dump(summary, stream, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
