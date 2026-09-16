"""Paired fresh questions with fixed wording/cue and independently chosen contexts."""
from __future__ import annotations

import random

from dataset import BOXES, simulate_swaps
from study_data import fingerprint
from swap_format_data import render, sample_semantics


def prefix_from_examples(examples):
    return "".join(f"{render(example, 'swap', 'in_box', 'clean')} {example['clean_target']}.\n" for example in examples)


def fresh_pairs(seed, depth, blocked, objects):
    originals = sample_semantics(seed, (depth,))
    for offset in range(1, len(BOXES)):
        generator = random.Random(seed + depth * 100 + offset)
        candidates = []
        for index, template in enumerate(originals):
            clean, changed = BOXES[index], BOXES[(index + offset) % len(BOXES)]
            swaps = [] if depth == 0 else [[clean, changed]]
            clean_initial, _ = simulate_swaps(clean, reversed(swaps))
            changed_initial, _ = simulate_swaps(changed, reversed(swaps))
            options = []
            for obj in generator.sample(objects, len(objects)):
                example = {**template, "object": obj, "swaps": swaps, "clean_initial": clean_initial,
                           "counterfactual_initial": changed_initial, "clean_target": clean,
                           "counterfactual_target": changed, "clean_moves": depth, "counterfactual_moves": depth}
                example["sample_id"] = fingerprint({key: value for key, value in example.items() if key != "sample_id"})
                prompts = {render(example, "swap", "in_box", side) for side in ("clean", "counterfactual")}
                if not prompts & blocked:
                    options.append((example, prompts))
            candidates.append(options)

        def assign(index, used):
            if index == len(candidates):
                return []
            for example, prompts in candidates[index]:
                if not prompts & used:
                    following = assign(index + 1, used | prompts)
                    if following is not None:
                        return [example] + following
            return None

        selected = assign(0, set())
        if selected is not None:
            return selected, offset
    raise ValueError("Cannot fill fresh context queries without overlap; do not silently reuse examples")


def build_design(protocol, previous, excluded):
    prefixes = {"none": "", "prior": previous["prefix"]}
    demonstrations = {"none": [], "prior": previous["demonstrations"]}
    for name, seed in protocol["demonstration_seeds"].items():
        generated = sample_semantics(seed, (0, 1, 3))
        selected = [generated[0], generated[11], generated[22]]
        demonstrations[name] = selected
        prefixes[name] = prefix_from_examples(selected)
    if len(set(prefixes.values())) != 4:
        raise ValueError("Context conditions must have distinct prefixes")
    blocked = set(excluded)
    blocked.update(render(case["semantic"], case["wording"], case["answer_cue"], case["side"]) for case in previous["cases"])
    blocked.update(render(example, "swap", "in_box", "clean") for group in demonstrations.values() for example in group)
    samples, offsets, unique_queries = [], {}, set()
    for depth in protocol["chain_lengths"]:
        group, offset = fresh_pairs(protocol["generator_seed"], depth, blocked | unique_queries, protocol["query_objects"])
        samples.extend(group)
        offsets[str(depth)] = offset
        unique_queries.update(render(example, "swap", "in_box", side) for example in group for side in protocol["sides"])
    cases = []
    for example in samples:
        for context in protocol["contexts"]:
            for side in protocol["sides"]:
                query = render(example, "swap", "in_box", side)
                case = {"sample_id": example["sample_id"], "context": context, "depth": example["depth"],
                        "side": side, "semantic": example, "target": example[f"{side}_target"],
                        "query": query, "prompt": prefixes[context] + query}
                case["case_id"] = fingerprint(case)
                cases.append(case)
    if len(cases) != 128 or len(unique_queries) != 32:
        raise ValueError("Expected 128 responses from 32 distinct side-specific questions")
    return {"version": protocol["version"], "prefixes": prefixes, "demonstrations": demonstrations, "counterfactual_offsets": offsets,
            "cases": cases, "n_semantic_pairs": 16, "n_responses": 128}