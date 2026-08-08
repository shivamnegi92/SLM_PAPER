"""Backbone pruning: truncate a pretrained GPT2-style transformer to the first
`depth` layers. This is the physical realization of C1's depth decision --
after the probe sweep picks a depth, we cut the model down to it and fine-tune.
"""
from __future__ import annotations

from transformers import GPT2LMHeadModel


def truncate_gpt2_backbone(model: GPT2LMHeadModel, depth: int) -> GPT2LMHeadModel:
    """Keep only the first `depth` transformer blocks (mutates and returns model)."""
    n_available = len(model.transformer.h)
    if not (1 <= depth <= n_available):
        raise ValueError(f"depth must be in [1, {n_available}], got {depth}")
    model.transformer.h = model.transformer.h[:depth]
    model.config.n_layer = depth
    return model
