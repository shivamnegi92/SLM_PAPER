"""Local-file loaders: read manually-downloaded raw datasets from disk.

No network access required. Use these once you've downloaded a dataset via
browser/git-clone (see experiments/datasets.md) and dropped it under
data/raw/<name>/.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from .datasets import Example, normalize_bio_record

PathLike = Union[str, Path]


def load_atis_local(parquet_path: PathLike) -> list[Example]:
    """ATIS from a local parquet file with columns text/intent/slots."""
    import pandas as pd  # lazy import: not a core dependency

    df = pd.read_parquet(parquet_path)
    examples = []
    for _, row in df.iterrows():
        tokens = row["text"].split()
        bio_tags = row["slots"].split()
        examples.append(normalize_bio_record(tokens=tokens, bio_tags=bio_tags, intent=row["intent"]))
    return examples


def load_clinc150_local(json_path: PathLike, split: str) -> list[Example]:
    """CLINC150 from the original oos-eval data_full.json: {split: [[text, intent], ...]}."""
    data = json.loads(Path(json_path).read_text())
    if split not in data:
        raise KeyError(f"split {split!r} not found; available: {list(data.keys())}")
    examples = []
    for text, intent in data[split]:
        tokens = text.split()
        bio_tags = ["O"] * len(tokens)
        examples.append(normalize_bio_record(tokens=tokens, bio_tags=bio_tags, intent=intent))
    return examples


def load_banking77_local(csv_path: PathLike) -> list[Example]:
    """BANKING77 from the original PolyAI CSV: columns text,category."""
    import csv as csv_module

    examples = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv_module.DictReader(f):
            tokens = row["text"].split()
            bio_tags = ["O"] * len(tokens)
            examples.append(normalize_bio_record(tokens=tokens, bio_tags=bio_tags, intent=row["category"]))
    return examples
