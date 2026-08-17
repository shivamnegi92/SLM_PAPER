"""BIO tag <-> span conversion for slot filling.

A span is a dict: {"type": str, "start": int, "end": int} with end exclusive.
bio_to_spans additionally attaches "text" (space-joined tokens).
"""
from __future__ import annotations

from typing import Optional


def spans_to_bio(tokens: list[str], spans: list[dict]) -> list[str]:
    """Convert typed spans into a BIO tag sequence aligned to tokens."""
    tags = ["O"] * len(tokens)
    for span in spans:
        start, end, stype = span["start"], span["end"], span["type"]
        if not 0 <= start < end <= len(tokens):
            raise ValueError(f"span out of bounds: {span} for {len(tokens)} tokens")
        tags[start] = f"B-{stype}"
        for i in range(start + 1, end):
            tags[i] = f"I-{stype}"
    return tags


def bio_to_spans(tokens: list[str], tags: list[str]) -> list[dict]:
    """Convert a BIO tag sequence into typed spans. Orphan I- tags are ignored."""
    if len(tokens) != len(tags):
        raise ValueError("tokens and tags must be the same length")
    spans: list[dict] = []
    cur_type: Optional[str] = None
    cur_start: Optional[int] = None

    def flush(end: int) -> None:
        nonlocal cur_type, cur_start
        if cur_type is not None:
            spans.append({
                "type": cur_type,
                "start": cur_start,
                "end": end,
                "text": " ".join(tokens[cur_start:end]),
            })
        cur_type, cur_start = None, None

    for i, tag in enumerate(tags):
        if tag == "O":
            flush(i)
        elif tag.startswith("B-"):
            flush(i)
            cur_type, cur_start = tag[2:], i
        elif tag.startswith("I-"):
            itype = tag[2:]
            if cur_type != itype:
                # orphan I- (no matching open span): drop it
                flush(i)
        else:
            raise ValueError(f"invalid BIO tag: {tag!r}")
    flush(len(tokens))
    return spans


def find_span(tokens: list[str], value_tokens: list[str]) -> Optional[tuple[int, int]]:
    """Return (start, end) of the first contiguous match of value_tokens, else None."""
    n, m = len(tokens), len(value_tokens)
    if m == 0:
        return None
    for i in range(n - m + 1):
        if tokens[i:i + m] == value_tokens:
            return (i, i + m)
    return None
