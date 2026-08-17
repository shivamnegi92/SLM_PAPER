"""Latency measurement harness: P50/P95 over a set of inputs, warmup excluded.

Timer is injectable so tests are deterministic without real sleeps, and so the
same harness can wrap CPU wall-clock timing in the real experiments.
"""
from __future__ import annotations

import math
import statistics
import time
from typing import Callable, Iterable


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (inclusive), matching common latency-report convention."""
    if not values:
        raise ValueError("cannot take a percentile of an empty list")
    ordered = sorted(values)
    rank = math.ceil((pct / 100) * len(ordered))
    rank = max(1, min(rank, len(ordered)))
    return ordered[rank - 1]


def measure_latency(
    fn: Callable,
    inputs: Iterable,
    warmup: int = 0,
    timer: Callable[[], float] = time.perf_counter,
) -> dict:
    """Run fn over inputs, discarding the first `warmup` calls from timing.

    Returns {"n", "mean", "p50", "p95"}, all zeroed if no timed calls remain.
    """
    inputs = list(inputs)
    for inp in inputs[:warmup]:
        fn(inp)

    durations: list[float] = []
    for inp in inputs[warmup:]:
        start = timer()
        fn(inp)
        end = timer()
        durations.append(end - start)

    if not durations:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0}

    return {
        "n": len(durations),
        "mean": statistics.mean(durations),
        "p50": percentile(durations, 50),
        "p95": percentile(durations, 95),
    }
