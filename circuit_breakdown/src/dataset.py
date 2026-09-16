"""Minimal-pair dataset generator for state-tracking interpretability.

Fixes the fatal bug in the original blueprint: the clean/corrupt pair MUST have
different targets, otherwise activation patching measures pure noise.

Two task families, both with clean != corrupt targets and (with single-token
names) token-position alignment suitable for activation patching:

  - "intermediate": query a NON-final state so corrupting a middle step changes
    the answer. e.g. "...London to Paris to Berlin. The location before Berlin
    was ___" (Paris) vs "...London to Tokyo to Berlin..." (Tokyo).

  - "transfer": object-passing chains that require composing steps (not a
    recency heuristic). "Alice has the key. Alice gives it to Bob. Bob gives it
    to Carol. Who has the key? ___" (Carol); corrupt one hop -> different holder.

Run with --self-check to assert the minimal-pair invariants. Optionally pass
--tokenizer <path> to also assert clean/corrupt token-length alignment.

Public data only: names/cities are generic. No proprietary content.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path

# Short, common, usually single-token vocab (verify with --tokenizer).
# Pools are intentionally large; restrict_to_single_token() trims them per model.
CITIES = ["London", "Paris", "Berlin", "Tokyo", "Rome", "Madrid", "Cairo",
          "Oslo", "Lima", "Delhi", "Seoul", "Dublin", "Athens", "Vienna",
          "Prague", "Boston", "Chicago", "Miami", "Denver", "Austin",
          "Dallas", "Houston", "Seattle", "Moscow", "Sydney", "Toronto"]
PEOPLE = ["Alice", "Bob", "Carol", "Dave", "Erin", "Frank", "Grace", "Henry",
          "Ivy", "Jack", "Kate", "Leo", "Mia", "Noah", "Emma", "Ava",
          "Lucas", "Mason", "Ethan", "Zoe", "Owen", "Nora", "Ruby", "Liam"]
OBJECTS = ["key", "book", "ball", "coin", "ring", "map"]
BOXES = list("ABCDEFGH")


def restrict_to_single_token(tokenizer, min_cities=8, min_people=10):
    """Trim CITIES/PEOPLE to items that are a SINGLE leading-space token under
    this tokenizer, so clean/corrupt pairs stay length-aligned for patching.
    Mutates the module globals. Raises if too few survive."""
    global CITIES, PEOPLE

    def ok(w):
        # count tokens the word contributes IN CONTEXT (after a dummy word).
        # Robust across BPE (leading-space token) and SentencePiece (u2581 mark).
        base = tokenizer("the", add_special_tokens=False)["input_ids"]
        ext = tokenizer("the " + w, add_special_tokens=False)["input_ids"]
        return len(ext) - len(base) == 1
    CITIES = [c for c in CITIES if ok(c)]
    PEOPLE = [p for p in PEOPLE if ok(p)]
    if len(CITIES) < min_cities or len(PEOPLE) < min_people:
        raise RuntimeError(
            f"after single-token filter: {len(CITIES)} cities, {len(PEOPLE)} "
            f"people (need >= {min_cities}/{min_people}); widen the pools")
    return len(CITIES), len(PEOPLE)


def _strip(word: str) -> str:
    """Punctuation-stripped core of a whitespace token (e.g. 'Henry.' -> 'Henry')."""
    return word.strip(".,!?;:")


def _diff_index(clean: str, corrupt: str) -> int:
    """Index of the single differing whitespace token between two prompts."""
    cw, kw = clean.split(), corrupt.split()
    assert len(cw) == len(kw), "prompts differ in word length"
    diffs = [j for j, (a, b) in enumerate(zip(cw, kw)) if a != b]
    assert len(diffs) == 1, f"expected exactly one differing word, got {diffs}"
    return diffs[0]


@dataclass
class Pair:
    task: str
    clean_prompt: str
    corrupt_prompt: str
    clean_target: str
    corrupt_target: str
    # word index (whitespace-split) where the single-token corruption lives;
    # used later to align activation-patching positions.
    corrupt_word_index: int
    chain_len: int
    metadata: dict = field(default_factory=dict)


def _make_intermediate(rng: random.Random, chain_len: int) -> Pair:
    """Query the state BEFORE the final one so a middle swap changes the answer.

    The corruption swaps the second-to-last city (the queried answer) for a
    different city, keeping every other token identical -> single-token,
    length-matched minimal pair, and clean_target != corrupt_target.
    """
    cities = rng.sample(CITIES, chain_len)
    clean_target = cities[-2]                      # "the location before <last>"
    # pick a replacement distinct from all cities so targets differ cleanly
    pool = [c for c in CITIES if c not in cities]
    corrupt_city = rng.choice(pool)

    corrupt_cities = list(cities)
    corrupt_cities[-2] = corrupt_city
    corrupt_target = corrupt_city

    route = " to ".join(cities)
    corrupt_route = " to ".join(corrupt_cities)
    last = cities[-1]
    clean = f"The package moved from {route}. The location before {last} was"
    corrupt = f"The package moved from {corrupt_route}. The location before {last} was"

    idx = _diff_index(clean, corrupt)
    return Pair("intermediate", clean, corrupt, clean_target, corrupt_target,
                idx, chain_len)


def _make_transfer(rng: random.Random, chain_len: int) -> Pair:
    """Object-passing chain; corrupt the LAST hop's recipient so the final
    holder (the answer) changes. Composition-required, not recency-solvable
    because the query asks for the current holder after a sequence of transfers.

    IMPORTANT anti-recency design: two objects are tracked in parallel and the
    LAST sentence transfers the *distractor* object. So the most-recently-named
    person holds the distractor, NOT the queried object -- a "copy last name"
    induction head gets it wrong. Answering correctly requires object->holder
    binding, i.e. genuine tracking (not a recency heuristic).

    chain_len = number of transfer rounds per object (each round moves both
    objects once). Needs 2 + 2*chain_len distinct people, plus one for corrupt.
    """
    max_rounds = (len(PEOPLE) - 3) // 2  # keep 2+2r+1 distinct people available
    rounds = max(1, min(chain_len, max_rounds))
    need = 2 + 2 * rounds + 1  # +1 spare for the corrupt recipient
    people = rng.sample(PEOPLE, need)
    obj_q, obj_d = rng.sample(OBJECTS, 2)  # queried vs distractor object

    # build the two holder chains from disjoint people
    q_chain = [people[0]] + people[2:2 + rounds]
    d_chain = [people[1]] + people[2 + rounds:2 + 2 * rounds]
    spare = people[2 + 2 * rounds]
    clean_target = q_chain[-1]

    def render(qc: list[str], dc: list[str]) -> str:
        lines = [f"{qc[0]} has the {obj_q}.", f"{dc[0]} has the {obj_d}."]
        for r in range(rounds):
            # queried transfer first, distractor transfer LAST each round
            lines.append(f"{qc[r]} gives the {obj_q} to {qc[r + 1]}.")
            lines.append(f"{dc[r]} gives the {obj_d} to {dc[r + 1]}.")
        lines.append(f"Who has the {obj_q}?")
        return " ".join(lines)

    corrupt_q = list(q_chain)
    corrupt_q[-1] = spare  # change only the queried object's final recipient
    corrupt_target = spare

    clean = render(q_chain, d_chain)
    corrupt = render(corrupt_q, d_chain)
    idx = _diff_index(clean, corrupt)
    return Pair("transfer", clean, corrupt, clean_target, corrupt_target,
                idx, chain_len)


def simulate_swaps(initial, swaps):
    location, moves = initial, 0
    for first, second in swaps:
        if first == second:
            raise ValueError("A swap requires distinct boxes")
        if location in (first, second):
            location = second if location == first else first
            moves += 1
    return location, moves


def render_swap(initial, swaps, obj, template):
    if template == 0:
        lines = [f"The {obj} is in box {initial}."]
        lines.extend(f"Swap the contents of box {first} and box {second}." for first, second in swaps)
        lines.append(f"The {obj} is now in box")
    elif template == 1:
        lines = [f"Initially, box {initial} contains the {obj}."]
        lines.extend(f"Exchange the contents of boxes {first} and {second}." for first, second in swaps)
        lines.append(f"The final box containing the {obj} is")
    else:
        raise ValueError("Unknown swap template")
    return " ".join(lines)


def _make_container_swap(rng, chain_len):
    if chain_len < 3:
        raise ValueError("Container swaps need at least three operations")
    boxes = rng.sample(BOXES, 6)
    clean_start, corrupt_start = rng.sample(boxes[:4], 2)
    for attempt in range(1000):
        swaps = [rng.sample(boxes[:4], 2) for _ in range(chain_len)]
        clean_target, clean_moves = simulate_swaps(clean_start, swaps)
        corrupt_target, corrupt_moves = simulate_swaps(corrupt_start, swaps)
        if (min(clean_moves, corrupt_moves) >= 2 and clean_target != clean_start
                and corrupt_target != corrupt_start):
            break
    else:
        raise RuntimeError("Could not sample a compositional swap chain")
    swaps.append(boxes[4:])
    obj = rng.choice(OBJECTS)
    template = rng.randrange(2)
    clean = render_swap(clean_start, swaps, obj, template)
    corrupt = render_swap(corrupt_start, swaps, obj, template)
    return Pair("container_swap", clean, corrupt, clean_target, corrupt_target,
                _diff_index(clean, corrupt), chain_len,
                {"initial": clean_start, "counterfactual_initial": corrupt_start,
                 "swaps": swaps, "object": obj, "template": template,
                 "clean_moves": clean_moves, "corrupt_moves": corrupt_moves})


def shortcut_scores(pairs):
    scores = {"copy_changed_word": [], "last_mentioned_name": [], "query_aware_extraction": []}
    for pair in pairs:
        for prompt, target in ((pair.clean_prompt, pair.clean_target),
                               (pair.corrupt_prompt, pair.corrupt_target)):
            changed = _strip(prompt.split()[pair.corrupt_word_index])
            if pair.task == "intermediate":
                route = prompt.split("moved from ", 1)[1].split(".", 1)[0].split(" to ")
                extraction, last = route[-2], route[-1]
            elif pair.task == "transfer":
                obj = re.search(r"Who has the (\w+)\?", prompt).group(1)
                extraction = re.findall(rf"gives the {re.escape(obj)} to (\w+)", prompt)[-1]
                last = re.findall(r" to (\w+)", prompt)[-1]
            else:
                extraction = changed
                last = pair.metadata["swaps"][-1][-1]
            for name, answer in (("copy_changed_word", changed),
                                 ("last_mentioned_name", last),
                                 ("query_aware_extraction", extraction)):
                scores[name].append(int(answer == target))
    return {name: sum(values) / len(values) for name, values in scores.items()}


_MAKERS = {"intermediate": _make_intermediate, "transfer": _make_transfer,
           "container_swap": _make_container_swap}


def generate(task: str, n: int, seed: int = 0,
             min_len: int = 3, max_len: int = 5) -> list[Pair]:
    if task not in _MAKERS:
        raise ValueError(f"unknown task {task!r}; choose from {list(_MAKERS)}")
    rng = random.Random(seed)
    maker = _MAKERS[task]
    out: list[Pair] = []
    seen: set[tuple[str, str]] = set()
    guard = 0
    while len(out) < n and guard < n * 50:
        guard += 1
        chain_len = rng.randint(min_len, max_len)
        p = maker(rng, chain_len)
        key = (p.clean_prompt, p.corrupt_prompt)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    if len(out) < n:
        raise RuntimeError(
            f"only produced {len(out)}/{n} unique pairs; widen vocab or lengths")
    return out


def self_check(pairs: list[Pair], tokenizer=None) -> None:
    """Assert the minimal-pair invariants. Raises AssertionError on violation."""
    for i, p in enumerate(pairs):
        assert p.clean_target != p.corrupt_target, \
            f"[{i}] SAME TARGET both sides -> zero patching signal: {p}"
        cw, kw = p.clean_prompt.split(), p.corrupt_prompt.split()
        assert len(cw) == len(kw), \
            f"[{i}] word-length mismatch breaks position alignment: {p}"
        diffs = [j for j, (a, b) in enumerate(zip(cw, kw)) if a != b]
        assert diffs == [p.corrupt_word_index], \
            f"[{i}] expected single diff at {p.corrupt_word_index}, got {diffs}"
        if p.task == "container_swap":
            meta = p.metadata
            clean_target, clean_moves = simulate_swaps(meta["initial"], meta["swaps"])
            corrupt_target, corrupt_moves = simulate_swaps(meta["counterfactual_initial"], meta["swaps"])
            assert (clean_target, corrupt_target) == (p.clean_target, p.corrupt_target)
            assert min(clean_moves, corrupt_moves) >= 2
            assert p.clean_target != _strip(cw[p.corrupt_word_index])
            assert p.corrupt_target != _strip(kw[p.corrupt_word_index])
            assert p.clean_prompt == render_swap(meta["initial"], meta["swaps"], meta["object"], meta["template"])
            assert p.corrupt_prompt == render_swap(meta["counterfactual_initial"], meta["swaps"], meta["object"], meta["template"])
        else:
            assert _strip(cw[p.corrupt_word_index]) == p.clean_target, \
                f"[{i}] clean diff word != clean_target: {p}"
            assert _strip(kw[p.corrupt_word_index]) == p.corrupt_target, \
                f"[{i}] corrupt diff word != corrupt_target: {p}"

    if tokenizer is not None:
        bad = 0
        for p in pairs:
            ct = tokenizer(" " + p.clean_target, add_special_tokens=False)["input_ids"]
            kt = tokenizer(" " + p.corrupt_target, add_special_tokens=False)["input_ids"]
            lc = len(tokenizer(p.clean_prompt, add_special_tokens=False)["input_ids"])
            lk = len(tokenizer(p.corrupt_prompt, add_special_tokens=False)["input_ids"])
            if len(ct) != 1 or len(kt) != 1 or lc != lk:
                bad += 1
        frac = bad / len(pairs)
        assert frac < 0.05, (
            f"{frac:.0%} of pairs are not single-token / token-length aligned; "
            "trim the vocab to single-token names for clean patching")
    print(f"self-check PASSED on {len(pairs)} pairs"
          + ("" if tokenizer is None else " (incl. tokenizer alignment)"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--task", default="transfer", choices=list(_MAKERS))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=5)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--tokenizer", default=None,
                    help="optional HF tokenizer path for alignment check")
    args = ap.parse_args()

    pairs = generate(args.task, args.n, args.seed, args.min_len, args.max_len)

    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer  # lazy import
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.self_check or tok is not None:
        self_check(pairs, tokenizer=tok)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for p in pairs:
            f.write(json.dumps(asdict(p)) + "\n")
    print(f"wrote {len(pairs)} pairs -> {args.out}")
    # show one example so a human can eyeball it
    ex = pairs[0]
    print("\nexample:")
    print("  clean  :", ex.clean_prompt, "=>", ex.clean_target)
    print("  corrupt:", ex.corrupt_prompt, "=>", ex.corrupt_target)


if __name__ == "__main__":
    main()
