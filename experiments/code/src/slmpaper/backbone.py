"""Backbone utilities for causal decoder models across architectures.

Supports GPT-style stacks (`model.h`), Llama-style stacks
(`model.model.layers`), and top-level layer layouts (`model.layers`)
used by models like Nemotron/Phi.
"""
from __future__ import annotations

from typing import Any


def get_num_layers(model: Any) -> int:
    """Return transformer block count from common decoder backbone layouts."""
    if hasattr(model, "h"):
        return len(model.h)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "layers"):
        return len(model.layers)

    cfg = getattr(model, "config", None)
    if cfg is not None:
        if hasattr(cfg, "n_layer"):
            return int(cfg.n_layer)
        if hasattr(cfg, "num_hidden_layers"):
            return int(cfg.num_hidden_layers)

    raise ValueError("Unable to infer layer count for backbone")


def truncate_backbone_layers(model: Any, depth: int) -> Any:
    """Truncate decoder backbone in-place to first `depth` blocks."""
    n_available = get_num_layers(model)
    if not (1 <= depth <= n_available):
        raise ValueError(f"depth must be in [1, {n_available}], got {depth}")

    if hasattr(model, "h"):
        model.h = model.h[:depth]
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        model.model.layers = model.model.layers[:depth]
    elif hasattr(model, "layers"):
        model.layers = model.layers[:depth]
    else:
        raise ValueError("Backbone layout unsupported for truncation")

    cfg = getattr(model, "config", None)
    if cfg is not None:
        if hasattr(cfg, "n_layer"):
            cfg.n_layer = depth
        if hasattr(cfg, "num_hidden_layers"):
            cfg.num_hidden_layers = depth

    return model
