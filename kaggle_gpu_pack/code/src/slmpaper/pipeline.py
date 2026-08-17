"""End-to-end normalization pipeline + dataset statistics.

Adapts HF-style slot-filling records (ATIS/SNIPS/MASSIVE) into the common
Example schema and computes the dataset stats we report in the paper (intent /
slot-type counts and the implicit-slot ceiling).
"""
from __future__ import annotations

from typing import Iterable, Optional

from .datasets import Example, normalize_bio_record
from .implicit import implicit_slot_rate


def records_to_examples(
    records: Iterable[dict],
    tokens_key: str,
    tags_key: str,
    intent_key: str,
    id2tag: Optional[dict[int, str]] = None,
) -> list[Example]:
    """Convert raw records into Examples.

    `tags_key` may hold integer tag ids (decoded via `id2tag`) or string BIO tags.
    """
    examples: list[Example] = []
    for rec in records:
        tokens = list(rec[tokens_key])
        raw_tags = rec[tags_key]
        if id2tag is not None:
            bio_tags = [id2tag[t] for t in raw_tags]
        else:
            bio_tags = list(raw_tags)
        examples.append(
            normalize_bio_record(tokens=tokens, bio_tags=bio_tags, intent=rec[intent_key])
        )
    return examples


def dataset_stats(examples: list[Example]) -> dict:
    """Summary statistics for a dataset split."""
    intents = set()
    slot_types = set()
    as_dicts = []
    for ex in examples:
        intents.add(ex.intent)
        slot_types.update(ex.slots.keys())
        as_dicts.append({"text": ex.text, "slots": ex.slots})
    return {
        "num_examples": len(examples),
        "num_intents": len(intents),
        "num_slot_types": len(slot_types),
        "implicit": implicit_slot_rate(as_dicts),
    }
