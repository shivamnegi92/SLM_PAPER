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


def load_atis_iob_local(csv_path: PathLike) -> list[Example]:
    """ATIS from the community IOB-CSV mirror: columns id,tokens,slots,intent.

    tokens/slots are space-joined strings wrapped in BOS/EOS sentinels
    (e.g. "BOS find a flight EOS" / "O O O O O") which must be stripped.
    """
    import csv as csv_module

    examples = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv_module.DictReader(f):
            tokens = row["tokens"].split()
            bio_tags = row["slots"].split()
            if tokens and tokens[0] == "BOS":
                tokens = tokens[1:]
                bio_tags = bio_tags[1:]
            if tokens and tokens[-1] == "EOS":
                tokens = tokens[:-1]
                bio_tags = bio_tags[:-1]
            examples.append(normalize_bio_record(tokens=tokens, bio_tags=bio_tags, intent=row["intent"]))
    return examples


def _read_text_robust(path: Path) -> str:
    """Read text as UTF-8, falling back to Latin-1 on decode failure.

    Documented quirk in the original sonos/nlu-benchmark release: at least
    one file (PlayMusic/train_PlayMusic_full.json) has non-UTF-8 bytes in
    some artist/song-name entities. Latin-1 never raises on any byte
    sequence, so this is a safe last resort for a known-quirky public file.
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def load_snips_local(root_dir: PathLike, split: str) -> list[Example]:
    """SNIPS from the original sonos/nlu-benchmark repo.

    root_dir is the "2017-06-custom-intent-engines/" folder: one
    subdirectory per intent, each containing train_<Intent>_full.json
    (falls back to train_<Intent>.json if _full is absent) and
    validate_<Intent>.json. Each example is {"data": [{"text", "entity"?}]};
    intent labels come from the top-level JSON key / folder name, not a field.

    Note: this raw source ships no separate held-out test file -- only
    train and validate. split must be "train" or "validate".
    """
    if split not in ("train", "validate"):
        raise ValueError(f"split must be 'train' or 'validate', got {split!r}")

    root = Path(root_dir)
    examples = []
    for intent_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        intent = intent_dir.name
        if split == "train":
            candidates = [
                intent_dir / f"train_{intent}_full.json",
                intent_dir / f"train_{intent}.json",
            ]
        else:
            candidates = [intent_dir / f"validate_{intent}.json"]
        file_path = next((c for c in candidates if c.exists()), None)
        if file_path is None:
            continue

        data = json.loads(_read_text_robust(file_path))
        for record in data[intent]:
            tokens: list[str] = []
            bio_tags: list[str] = []
            for chunk in record["data"]:
                chunk_tokens = chunk["text"].split()
                entity = chunk.get("entity")
                for i, tok in enumerate(chunk_tokens):
                    tokens.append(tok)
                    if entity is None:
                        bio_tags.append("O")
                    else:
                        bio_tags.append(f"B-{entity}" if i == 0 else f"I-{entity}")
            examples.append(normalize_bio_record(tokens=tokens, bio_tags=bio_tags, intent=intent))
    return examples
