"""Label vocabularies for intent classification + BIO slot tagging."""
from __future__ import annotations

from .datasets import Example


def build_intent_vocab(examples: list[Example]) -> dict[str, int]:
    """Sorted, deterministic intent -> id mapping."""
    intents = sorted({ex.intent for ex in examples})
    return {intent: i for i, intent in enumerate(intents)}


def build_tag_vocab(examples: list[Example]) -> dict[str, int]:
    """BIO tag -> id mapping. "O" is always id 0 (safe default/pad label)."""
    tags = set()
    for ex in examples:
        tags.update(ex.bio_tags)
    other_tags = sorted(tags - {"O"})
    tag2id = {"O": 0}
    for i, tag in enumerate(other_tags, start=1):
        tag2id[tag] = i
    return tag2id
