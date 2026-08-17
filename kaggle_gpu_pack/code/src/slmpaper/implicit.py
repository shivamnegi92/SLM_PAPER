"""Implicit-slot detection.

A slot value is "implicit" when it does NOT appear as a contiguous token span in
the (normalized) utterance. Span-based taggers cannot recover implicit slots, so
their fraction sets a hard ceiling on span-tagger recall.
"""
from __future__ import annotations

from .bio import find_span


def _normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation to spaces, split on whitespace."""
    cleaned = []
    for ch in text.lower():
        cleaned.append(ch if ch.isalnum() or ch.isspace() else " ")
    return "".join(cleaned).split()


def is_implicit(utterance: str, value: str) -> bool:
    """True if `value` is not a contiguous token span of `utterance`."""
    utt_tokens = _normalize(utterance)
    val_tokens = _normalize(value)
    if not val_tokens:
        return True
    return find_span(utt_tokens, val_tokens) is None


def implicit_slot_rate(examples: list[dict]) -> dict:
    """Aggregate implicit-slot statistics over examples.

    Each example: {"text": str, "slots": {slot_type: [values]}}.
    Returns total slot values, implicit count, rate, and max span-tagger recall.
    """
    total = 0
    implicit = 0
    for ex in examples:
        text = ex.get("text", "")
        for values in ex.get("slots", {}).values():
            for value in values:
                total += 1
                if is_implicit(text, value):
                    implicit += 1
    if total == 0:
        return {"total": 0, "implicit": 0, "rate": 0.0, "max_span_recall": 1.0}
    rate = implicit / total
    return {
        "total": total,
        "implicit": implicit,
        "rate": rate,
        "max_span_recall": 1.0 - rate,
    }
