"""Fresh, balanced swap chains for a crossed wording/answer-format diagnostic."""
from __future__ import annotations

import random
import re

from dataset import BOXES, OBJECTS, simulate_swaps
from study_data import fingerprint


def sample_semantics(seed, lengths=(0, 1, 3, 4)):
    generator = random.Random(seed)
    result = []
    for depth in lengths:
        if depth not in (0, 1, 3, 4):
            raise ValueError("Unsupported diagnostic chain length")
        for index, clean_target in enumerate(BOXES):
            counterfactual_target = BOXES[(index + 1) % len(BOXES)]
            for attempt in range(10000):
                swaps = ([generator.sample(BOXES, 2) for step in range(depth)] if depth > 1
                         else [[clean_target, counterfactual_target]] if depth == 1 else [])
                clean_initial, _ = simulate_swaps(clean_target, reversed(swaps))
                counterfactual_initial, _ = simulate_swaps(counterfactual_target, reversed(swaps))
                clean_final, clean_moves = simulate_swaps(clean_initial, swaps)
                counterfactual_final, counterfactual_moves = simulate_swaps(counterfactual_initial, swaps)
                if depth >= 3 and (min(clean_moves, counterfactual_moves) < 2
                                   or clean_final == clean_initial or counterfactual_final == counterfactual_initial):
                    continue
                break
            else:
                raise ValueError("Could not sample the declared balanced swap chain")
            example = {"depth": depth, "object": generator.choice(OBJECTS), "swaps": swaps,
                       "clean_initial": clean_initial, "counterfactual_initial": counterfactual_initial,
                       "clean_target": clean_final, "counterfactual_target": counterfactual_final,
                       "clean_moves": clean_moves, "counterfactual_moves": counterfactual_moves}
            example["sample_id"] = fingerprint(example)
            result.append(example)
    return result


def render(example, wording, cue, side):
    if side not in ("clean", "counterfactual"):
        raise ValueError("Unknown counterfactual side")
    initial, obj = example[f"{side}_initial"], example["object"]
    if wording == "swap":
        parts = [f"The {obj} is in box {initial}."]
        parts.extend(f"Swap the contents of box {first} and box {second}." for first, second in example["swaps"])
    elif wording == "exchange":
        parts = [f"Initially, box {initial} contains the {obj}."]
        parts.extend(f"Exchange the contents of boxes {first} and {second}." for first, second in example["swaps"])
    else:
        raise ValueError("Unknown diagnostic wording")
    if cue == "in_box":
        parts.append(f"The {obj} is now in box")
    elif cue == "is":
        parts.append(f"The final box containing the {obj} is")
    else:
        raise ValueError("Unknown diagnostic answer cue")
    return " ".join(parts)


def parse_first_answer(response):
    matched = re.match(r"\A\s*(?:[Bb]ox\s+)?([A-H])(?=\s|[.!?,;:]|$)", response)
    return matched.group(1) if matched else None


def build_cases(protocol, excluded_prompts):
    samples = sample_semantics(protocol["generator_seed"], protocol["chain_lengths"])
    demos = sample_semantics(protocol["demonstration_seed"], (0, 1, 3))
    selected = [demos[0], demos[8 + 3], demos[16 + 6]]
    prefix = "".join(f"{render(example, 'swap', 'in_box', 'clean')} {example['clean_target']}.\n"
                     for example in selected)
    demonstration_prompts = {render(example, "swap", "in_box", "clean") for example in selected}
    seen, cases = set(), []
    for example in samples:
        for wording in protocol["wordings"]:
            for cue in protocol["answer_cues"]:
                for side in protocol["sides"]:
                    prompt = render(example, wording, cue, side)
                    if prompt in excluded_prompts or prompt in demonstration_prompts or prompt in seen:
                        raise ValueError("Diagnostic prompt overlaps old data, demonstrations or another case")
                    seen.add(prompt)
                    case = {"sample_id": example["sample_id"], "depth": example["depth"], "wording": wording,
                            "answer_cue": cue, "side": side, "target": example[f"{side}_target"],
                            "prompt": prefix + prompt, "semantic": example}
                    case["case_id"] = fingerprint(case)
                    cases.append(case)
    if len(cases) != 256:
        raise ValueError("Diagnostic matrix requires exactly 256 paired responses")
    return {"demonstrations": selected, "cases": cases, "prefix": prefix,
            "n_semantic_pairs": len(samples), "n_responses": len(cases)}