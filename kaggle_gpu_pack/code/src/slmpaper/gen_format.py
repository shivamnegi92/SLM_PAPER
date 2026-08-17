"""Generative baseline (B1) formatting: prompt, target JSON, causal label mask."""
from __future__ import annotations

import json

from .datasets import Example

PROMPT_TEMPLATE = "Utterance: {text}\nOutput:"


def format_prompt(example: Example) -> str:
    """Prompt shown to the model (no answer leakage)."""
    return PROMPT_TEMPLATE.format(text=example.text)


def format_target(example: Example) -> str:
    """Gold completion as compact JSON: {"intent": ..., "slots": {...}}."""
    return json.dumps({"intent": example.intent, "slots": example.slots},
                      separators=(",", ":"), ensure_ascii=False)


def build_causal_labels(input_ids: list[int], prompt_len: int) -> list[int]:
    """Causal-LM labels: mask the prompt tokens (-100), supervise the completion."""
    labels = list(input_ids)
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100
    return labels
