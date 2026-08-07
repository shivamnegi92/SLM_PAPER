"""Real public dataset loaders (ATIS, CLINC150, BANKING77) -> common Example schema.

Requires the `datasets` package (see pyproject `models` extra) and network access
to the Hub. No proxy configuration lives here -- set HTTP_PROXY/HTTPS_PROXY in
your shell if your network requires one.

Schemas verified 2026-08-07 (see tests/test_hf_loaders.py docstring). If the
upstream Hub repo changes its columns, these loaders will raise a clear KeyError
rather than silently mis-mapping data.
"""
from __future__ import annotations

from typing import Optional

from .datasets import Example, normalize_bio_record


def _hf_load(name: str, split: str, config: Optional[str] = None):
    from datasets import load_dataset  # imported lazily: not a core dependency

    return load_dataset(name, config, split=split) if config else load_dataset(name, split=split)


def load_atis(split: str = "train", limit: Optional[int] = None) -> list[Example]:
    """ATIS: joint intent + slot filling. `slots` is a whitespace-aligned BIO string."""
    split_expr = f"{split}[:{limit}]" if limit else split
    ds = _hf_load("tuetschek/atis", split_expr)
    examples = []
    for rec in ds:
        tokens = rec["text"].split()
        bio_tags = rec["slots"].split()
        examples.append(normalize_bio_record(tokens=tokens, bio_tags=bio_tags, intent=rec["intent"]))
    return examples


def load_clinc150(split: str = "train", limit: Optional[int] = None) -> list[Example]:
    """CLINC150: intent-only (incl. out-of-scope). Single 'complete' split filtered by `split` col."""
    ds = _hf_load("contemmcm/clinc150", "complete")
    ds = ds.filter(lambda rec: rec["split"] == split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    examples = []
    for rec in ds:
        tokens = rec["text"].split()
        bio_tags = ["O"] * len(tokens)
        examples.append(normalize_bio_record(tokens=tokens, bio_tags=bio_tags, intent=str(rec["intent"])))
    return examples


def load_banking77(split: str = "train", limit: Optional[int] = None) -> list[Example]:
    """BANKING77: 77 fine-grained banking intents, intent-only.

    Tries mteb/banking77 first, falls back to legacy-datasets/banking77 (same
    data, different label column) since Hub blob-transport backends vary by
    repo and some networks only allow one of them through.
    """
    split_expr = f"{split}[:{limit}]" if limit else split
    try:
        ds = _hf_load("mteb/banking77", split_expr)
        label_key = "label_text"
    except Exception:
        ds = _hf_load("legacy-datasets/banking77", split_expr)
        label_key = "label"

    features = ds.features.get(label_key)
    examples = []
    for rec in ds:
        tokens = rec["text"].split()
        bio_tags = ["O"] * len(tokens)
        raw_label = rec[label_key]
        intent = features.int2str(raw_label) if hasattr(features, "int2str") else str(raw_label)
        examples.append(normalize_bio_record(tokens=tokens, bio_tags=bio_tags, intent=intent))
    return examples
