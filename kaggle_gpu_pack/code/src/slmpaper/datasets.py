"""Dataset normalizer: map raw per-token BIO records to a common Example schema.

Public benchmarks (ATIS, SNIPS, MASSIVE) ship slots as per-token BIO tags. We
normalize to a single schema the whole pipeline consumes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .bio import bio_to_spans


@dataclass
class Example:
    text: str
    tokens: list[str]
    intent: str
    slots: dict[str, list[str]] = field(default_factory=dict)
    bio_tags: list[str] = field(default_factory=list)


def normalize_bio_record(tokens: list[str], bio_tags: list[str], intent: str) -> Example:
    """Build an Example from tokens + per-token BIO tags + intent label."""
    if len(tokens) != len(bio_tags):
        raise ValueError("tokens and bio_tags must be the same length")
    spans = bio_to_spans(tokens, bio_tags)
    slots: dict[str, list[str]] = {}
    for span in spans:
        slots.setdefault(span["type"], []).append(span["text"])
    return Example(
        text=" ".join(tokens),
        tokens=list(tokens),
        intent=intent,
        slots=slots,
        bio_tags=list(bio_tags),
    )
