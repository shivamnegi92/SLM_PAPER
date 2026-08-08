"""Label vocabularies for intent classification + BIO slot tagging."""
from __future__ import annotations

from .datasets import Example


def build_intent_vocab(examples: list[Example]) -> dict[str, int]:
    """Sorted, deterministic intent -> id mapping."""
    intents = sorted({ex.intent for ex in examples})
    return {intent: i for i, intent in enumerate(intents)}


def build_tag_vocab(examples: list[Example]) -> dict[str, int]:
    """BIO tag -> id mapping. "O" is always id 0 (safe default/pad label).

    Both B-<type> and I-<type> are included for every observed slot type, even
    if only one variant appears in the given examples -- this prevents KeyErrors
    when an unseen I-/B- variant shows up at eval time.
    """
    types = set()
    for ex in examples:
        for tag in ex.bio_tags:
            if tag != "O":
                types.add(tag[2:])  # strip "B-"/"I-" prefix
    tag2id = {"O": 0}
    next_id = 1
    for t in sorted(types):
        tag2id[f"B-{t}"] = next_id
        tag2id[f"I-{t}"] = next_id + 1
        next_id += 2
    return tag2id
