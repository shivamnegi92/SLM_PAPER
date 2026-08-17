"""Value-based slot F1: compare {type: [values]} dicts across examples.

Used for the generative baseline (emits values, not positions) and for any
apples-to-apples cross-model comparison. Multiset match on (type, value).
"""
from __future__ import annotations


def _pairs(slots: dict) -> list[tuple]:
    out = []
    for stype, values in slots.items():
        for v in values:
            out.append((stype, v))
    return out


def value_slot_f1(pred: list[dict], gold: list[dict]) -> dict:
    if len(pred) != len(gold):
        raise ValueError("pred and gold must be the same length")

    tp = fp = fn = 0
    for p, g in zip(pred, gold):
        gold_pool = _pairs(g)
        for key in _pairs(p):
            if key in gold_pool:
                tp += 1
                gold_pool.remove(key)
            else:
                fp += 1
        fn += len(gold_pool)

    if tp == 0 and fp == 0 and fn == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
