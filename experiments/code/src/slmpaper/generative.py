"""Structured-output parsing for the generative baseline.

Turns raw generated text into {"intent": str, "slots": dict} or None on any
schema violation. The failure rate here IS the paper's headline reliability
metric (C2): discriminative heads make this failure mode structurally
impossible, this module is how we measure it for the generative baseline.
"""
from __future__ import annotations

import json
from typing import Optional


def _extract_first_json_object(text: str) -> Optional[str]:
    """Return the substring spanning the first balanced {...} block, else None."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_structured_output(text: str) -> Optional[dict]:
    """Parse generated text into a validated {"intent", "slots"} dict, or None."""
    candidate = _extract_first_json_object(text)
    if candidate is None:
        return None
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    intent = obj.get("intent")
    if not isinstance(intent, str):
        return None
    slots = obj.get("slots", {})
    if not isinstance(slots, dict):
        return None
    return {"intent": intent, "slots": slots}


def parse_failure_rate(outputs: list[str]) -> float:
    """Fraction of generated outputs that fail to parse into a valid schema."""
    if not outputs:
        return 0.0
    failures = sum(1 for o in outputs if parse_structured_output(o) is None)
    return failures / len(outputs)
