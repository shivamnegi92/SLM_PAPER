"""Depth-selection decision rule (C1): pick the shallowest depth within epsilon
of the best probe accuracy. This is the paper's "probe predicts depth" claim.
"""
from __future__ import annotations


def pick_depth(depth_to_accuracy: dict[int, float], epsilon: float = 0.02) -> int:
    """Smallest depth whose accuracy is within `epsilon` of the max accuracy."""
    if not depth_to_accuracy:
        raise ValueError("depth_to_accuracy must not be empty")
    best_acc = max(depth_to_accuracy.values())
    candidates = [d for d, acc in depth_to_accuracy.items() if best_acc - acc <= epsilon]
    return min(candidates)
