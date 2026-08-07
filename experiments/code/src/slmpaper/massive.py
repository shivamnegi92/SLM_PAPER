"""MASSIVE (Amazon Science) local loader.

Parses the `annot_utt` inline-bracket slot annotation format
("... [type : value] ...") into tokens + BIO tags, then reuses the common
normalize_bio_record pipeline.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Union

from .datasets import Example, normalize_bio_record

PathLike = Union[str, Path]

_SLOT_PATTERN = re.compile(r"\[(?P<type>[\w.]+)\s*:\s*(?P<value>[^\]]+)\]")


def parse_annotated_utterance(annot_utt: str) -> tuple[list[str], list[str]]:
    """Convert "... [type : value] ..." into (tokens, bio_tags)."""
    tokens: list[str] = []
    tags: list[str] = []
    pos = 0
    for match in _SLOT_PATTERN.finditer(annot_utt):
        before = annot_utt[pos:match.start()]
        for tok in before.split():
            tokens.append(tok)
            tags.append("O")

        slot_type = match.group("type")
        value_tokens = match.group("value").split()
        for i, tok in enumerate(value_tokens):
            tokens.append(tok)
            tags.append(f"B-{slot_type}" if i == 0 else f"I-{slot_type}")

        pos = match.end()

    for tok in annot_utt[pos:].split():
        tokens.append(tok)
        tags.append("O")

    return tokens, tags


def load_massive_local(jsonl_path: PathLike, split: str) -> list[Example]:
    """Load a MASSIVE per-locale JSONL file, filtered to `split` (train/dev/test)."""
    examples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["partition"] != split:
                continue
            tokens, bio_tags = parse_annotated_utterance(rec["annot_utt"])
            examples.append(normalize_bio_record(tokens=tokens, bio_tags=bio_tags, intent=rec["intent"]))
    return examples
