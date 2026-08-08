"""Central dataset registry: (name, split) -> list[Example]. Shared by scripts."""
from __future__ import annotations

from .datasets import Example


def load_split(dataset: str, split: str) -> list[Example]:
    """split is 'train' or 'test' (mapped to each dataset's eval split)."""
    d = dataset.lower()
    if d == "snips":
        from .local_loaders import load_snips_local
        return load_snips_local("data/raw/snips", split="train" if split == "train" else "validate")
    if d == "banking77":
        from .local_loaders import load_banking77_local
        fname = "train.csv" if split == "train" else "test.csv"
        return load_banking77_local(f"data/raw/banking77/{fname}")
    if d == "atis":
        from .local_loaders import load_atis_iob_local
        fname = "atis.train.csv" if split == "train" else "atis.test.csv"
        return load_atis_iob_local(f"data/raw/atis_iob/{fname}")
    if d == "massive":
        from .massive import load_massive_local
        return load_massive_local("data/raw/massive/1.1/data/en-US.jsonl",
                                   split="train" if split == "train" else "test")
    if d == "clinc150":
        from .hf_loaders import load_clinc150
        return load_clinc150("train" if split == "train" else "test")
    raise ValueError(f"unknown dataset {dataset!r}")
