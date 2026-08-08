"""Probe utilities: mean-pool a layer's hidden states, and fit a linear probe.

The probe-sweep IS contribution C1 (probe-guided depth selection): we train a
cheap linear classifier on frozen hidden states at several depths, and pick
the shallowest depth within epsilon of the best probe accuracy (see
depth_selection.pick_depth) -- *before* paying for full fine-tuning.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def pool_hidden_states(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool over the sequence dim, ignoring padded positions.

    hidden: (batch, seq, dim). attention_mask: (batch, seq), 1=real, 0=pad.
    """
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (batch, seq, 1)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def train_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int,
    epochs: int = 50,
    lr: float = 0.1,
) -> float:
    """Fit nn.Linear(dim, num_classes) on frozen features; return val accuracy.

    Deliberately cheap (no hidden layers, no dropout) -- the whole point of a
    probe is that it's fast enough to sweep many depths before real training.
    """
    dim = train_features.shape[1]
    probe = nn.Linear(dim, num_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    for _ in range(epochs):
        opt.zero_grad()
        logits = probe(train_features)
        loss = nn.functional.cross_entropy(logits, train_labels)
        loss.backward()
        opt.step()

    with torch.no_grad():
        preds = probe(val_features).argmax(-1)
        return (preds == val_labels).float().mean().item()
