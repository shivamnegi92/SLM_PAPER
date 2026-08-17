"""Aggregation helpers for multi-seed result reporting."""
from __future__ import annotations

import statistics


def mean_std(values: list[float]) -> tuple[float, float]:
    """Return (mean, population-std). Raises on empty input."""
    if not values:
        raise ValueError("cannot aggregate an empty list")
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std
