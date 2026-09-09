"""Paired-example generator for causal interchange (interchange-intervention) tests.

The steering experiments ask "can an edit make the model say X". That objective
is chosen by the optimizer, so a reviewer can fairly say the edit was trained to
produce the answer token. An interchange intervention is a stronger test: the
intervention is DETERMINED BY ANOTHER EXAMPLE, not by an objective that knows
the receiver's target.

High-level causal variable for the transfer task:

    Z = current holder of the queried object

Given a donor A with Z_A and a receiver B with Z_B, transplanting A's latent
state into B should, if the representation is a faithful causal abstraction,
make B behave as though its current holder were Z_A.

Design decisions that make this measurable:

  - DISJOINT NAMES. A's names never appear in B's prompt. So if patched-B
    predicts Z_A, it emits a token absent from its own context; that cannot be
    explained by B's surface content. This makes interchange accuracy
    unambiguous rather than a shift among names B already mentions.
  - DISJOINT OBJECTS, for the same reason at the object level.
  - IDENTICAL STRUCTURE. A and B share round count and rendering, so the two
    prompts are token-length aligned and residual positions correspond.
  - SAME-ANSWER CONTROL. Pairs where Z_A == Z_B via different chains. If an
    intervention only manipulates answer identity there is nothing to transfer
    here, so a large behavioral change on these items is evidence the edit
    moves more than the output token.
  - RANDOM DONOR CONTROL is constructed at run time by re-pairing donors.

Public data only: generic names and objects, reusing `dataset.py` pools.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random

import dataset


@dataclass
class InterchangeItem:
    """A donor/receiver pair sharing structure but not vocabulary."""

    kind: str
    rounds: int
    donor_prompt: str
    receiver_prompt: str
    donor_answer: str
    receiver_answer: str
    donor_object: str
    receiver_object: str
    donor_names: list
    receiver_names: list
    metadata: dict = field(default_factory=dict)


def render_transfer(queried_chain, distractor_chain, queried_object, distractor_object):
    """Render one transfer problem. Mirrors `dataset._make_transfer` rendering:
    the distractor transfer comes last in each round, so the most recently
    named person does NOT hold the queried object and a recency heuristic fails.
    """
    lines = [f"{queried_chain[0]} has the {queried_object}.",
             f"{distractor_chain[0]} has the {distractor_object}."]
    for index in range(len(queried_chain) - 1):
        lines.append(f"{queried_chain[index]} gives the {queried_object} to {queried_chain[index + 1]}.")
        lines.append(f"{distractor_chain[index]} gives the {distractor_object} to {distractor_chain[index + 1]}.")
    lines.append(f"Who has the {queried_object}?")
    return " ".join(lines)


def _draw_problem(rng, people_pool, objects_pool, rounds):
    """Sample one transfer problem, consuming names/objects from the pools."""
    need = 2 + 2 * rounds
    if len(people_pool) < need or len(objects_pool) < 2:
        raise ValueError("Insufficient disjoint vocabulary for the requested rounds")
    people = rng.sample(people_pool, need)
    queried_object, distractor_object = rng.sample(objects_pool, 2)
    queried_chain = [people[0]] + people[2:2 + rounds]
    distractor_chain = [people[1]] + people[2 + rounds:2 + 2 * rounds]
    prompt = render_transfer(queried_chain, distractor_chain, queried_object, distractor_object)
    return {"prompt": prompt, "answer": queried_chain[-1], "object": queried_object,
            "names": people, "queried_chain": queried_chain,
            "distractor_chain": distractor_chain, "distractor_object": distractor_object}


def make_disjoint_item(rng, rounds):
    """Donor and receiver with fully disjoint names and objects, Z_A != Z_B."""
    people = rng.sample(dataset.PEOPLE, 2 * (2 + 2 * rounds))
    objects = rng.sample(dataset.OBJECTS, 4)
    half = len(people) // 2
    donor = _draw_problem(rng, people[:half], objects[:2], rounds)
    receiver = _draw_problem(rng, people[half:], objects[2:], rounds)
    if donor["answer"] == receiver["answer"]:
        raise ValueError("Disjoint pools must not produce a shared answer")
    return InterchangeItem(
        kind="disjoint", rounds=rounds,
        donor_prompt=donor["prompt"], receiver_prompt=receiver["prompt"],
        donor_answer=donor["answer"], receiver_answer=receiver["answer"],
        donor_object=donor["object"], receiver_object=receiver["object"],
        donor_names=donor["names"], receiver_names=receiver["names"],
        metadata={"donor_chain": donor["queried_chain"], "receiver_chain": receiver["queried_chain"]})


def make_same_answer_item(rng, rounds):
    """Donor and receiver reaching the SAME answer by different chains.

    Answer identity is held constant, so an intervention that only manipulates
    the output token has nothing to transfer. Every other name is disjoint.
    """
    people = rng.sample(dataset.PEOPLE, 2 * (2 + 2 * rounds) + 1)
    objects = rng.sample(dataset.OBJECTS, 4)
    shared = people[-1]
    remaining = people[:-1]
    half = len(remaining) // 2
    donor = _draw_problem(rng, remaining[:half], objects[:2], rounds)
    receiver = _draw_problem(rng, remaining[half:], objects[2:], rounds)

    donor_chain = list(donor["queried_chain"])
    receiver_chain = list(receiver["queried_chain"])
    donor_chain[-1] = shared
    receiver_chain[-1] = shared
    donor_prompt = render_transfer(donor_chain, donor["distractor_chain"],
                                   donor["object"], donor["distractor_object"])
    receiver_prompt = render_transfer(receiver_chain, receiver["distractor_chain"],
                                      receiver["object"], receiver["distractor_object"])
    return InterchangeItem(
        kind="same_answer", rounds=rounds,
        donor_prompt=donor_prompt, receiver_prompt=receiver_prompt,
        donor_answer=shared, receiver_answer=shared,
        donor_object=donor["object"], receiver_object=receiver["object"],
        donor_names=donor["names"] + [shared], receiver_names=receiver["names"] + [shared],
        metadata={"donor_chain": donor_chain, "receiver_chain": receiver_chain,
                  "shared_answer": shared})


_MAKERS = {"disjoint": make_disjoint_item, "same_answer": make_same_answer_item}


def _tokenizer_aligned(item, tokenizer):
    """Validate token-length alignment and single-token answers against the
    ACTUAL rendered prompt (not a generic 'the <name>' proxy), since
    sentence-initial position and surrounding punctuation can tokenize a name
    differently than mid-sentence occurrences do."""
    donor_ids = tokenizer(item.donor_prompt, add_special_tokens=False)["input_ids"]
    receiver_ids = tokenizer(item.receiver_prompt, add_special_tokens=False)["input_ids"]
    donor_answer_ids = tokenizer(" " + item.donor_answer, add_special_tokens=False)["input_ids"]
    receiver_answer_ids = tokenizer(" " + item.receiver_answer, add_special_tokens=False)["input_ids"]
    return (len(donor_ids) == len(receiver_ids)
            and len(donor_answer_ids) == 1 and len(receiver_answer_ids) == 1)


def generate(kind, n, seed=0, rounds=2, attempts_per_item=200, tokenizer=None):
    if kind not in _MAKERS:
        raise ValueError(f"Unknown interchange item kind {kind!r}")
    rng = random.Random(seed)
    maker = _MAKERS[kind]
    items, seen = [], set()
    guard = 0
    while len(items) < n and guard < n * attempts_per_item:
        guard += 1
        try:
            item = maker(rng, rounds)
        except ValueError:
            continue
        key = (item.donor_prompt, item.receiver_prompt)
        if key in seen:
            continue
        if tokenizer is not None and not _tokenizer_aligned(item, tokenizer):
            continue
        seen.add(key)
        items.append(item)
    if len(items) < n:
        raise RuntimeError(f"Only produced {len(items)}/{n} interchange items")
    return items


def self_check(items, tokenizer=None):
    """Assert the invariants the interchange test depends on."""
    for index, item in enumerate(items):
        donor_names = set(item.metadata["donor_chain"])
        receiver_names = set(item.metadata["receiver_chain"])
        if item.kind == "disjoint":
            assert item.donor_answer != item.receiver_answer, \
                f"[{index}] disjoint item must have different answers"
            assert not (donor_names & receiver_names), \
                f"[{index}] donor and receiver chains share a name"
            assert item.donor_answer not in item.receiver_prompt, \
                f"[{index}] donor answer leaks into the receiver prompt"
            assert item.donor_object != item.receiver_object, \
                f"[{index}] donor and receiver share the queried object"
        else:
            assert item.donor_answer == item.receiver_answer, \
                f"[{index}] same-answer item must share the answer"
            assert item.metadata["donor_chain"][:-1] != item.metadata["receiver_chain"][:-1], \
                f"[{index}] same-answer item must reach the answer by different chains"
        assert item.donor_answer == item.metadata["donor_chain"][-1]
        assert item.receiver_answer == item.metadata["receiver_chain"][-1]
        assert len(item.donor_prompt.split()) == len(item.receiver_prompt.split()), \
            f"[{index}] donor/receiver word-length mismatch breaks position alignment"

    if tokenizer is not None:
        bad = 0
        for item in items:
            donor_ids = tokenizer(item.donor_prompt, add_special_tokens=False)["input_ids"]
            receiver_ids = tokenizer(item.receiver_prompt, add_special_tokens=False)["input_ids"]
            donor_answer = tokenizer(" " + item.donor_answer, add_special_tokens=False)["input_ids"]
            receiver_answer = tokenizer(" " + item.receiver_answer, add_special_tokens=False)["input_ids"]
            if (len(donor_ids) != len(receiver_ids) or len(donor_answer) != 1
                    or len(receiver_answer) != 1):
                bad += 1
        fraction = bad / len(items)
        assert fraction < 0.05, (
            f"{fraction:.0%} of interchange items are not token-length aligned or "
            "not single-token answers; trim the vocabulary")
    print(f"interchange self-check PASSED on {len(items)} items"
          + ("" if tokenizer is None else " (incl. tokenizer alignment)"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--kind", default="disjoint", choices=list(_MAKERS))
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    items = generate(args.kind, args.n, args.seed, args.rounds, tokenizer=tokenizer)
    if args.self_check or tokenizer is not None:
        self_check(items, tokenizer=tokenizer)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as stream:
        for item in items:
            stream.write(json.dumps(asdict(item)) + "\n")
    print(f"wrote {len(items)} interchange items -> {args.out}")
    example = items[0]
    print("\nexample:")
    print("  donor   :", example.donor_prompt, "=>", example.donor_answer)
    print("  receiver:", example.receiver_prompt, "=>", example.receiver_answer)


if __name__ == "__main__":
    main()
