"""Evaluation metrics: intent accuracy and span-level slot F1."""
from __future__ import annotations


def intent_accuracy(preds: list[str], golds: list[str]) -> float:
    """Exact-match intent accuracy. Empty input -> 0.0."""
    if len(preds) != len(golds):
        raise ValueError("preds and golds must be the same length")
    if not preds:
        return 0.0
    correct = sum(1 for p, g in zip(preds, golds) if p == g)
    return correct / len(preds)


def _span_key(span: dict) -> tuple:
    return (span["type"], span["start"], span["end"])


def span_f1(pred_spans: list[list[dict]], gold_spans: list[list[dict]]) -> dict:
    """Span-level (type + boundary) precision/recall/F1 over a list of examples.

    Convention: if there are no gold and no predicted spans anywhere, F1 = 1.0.
    """
    if len(pred_spans) != len(gold_spans):
        raise ValueError("pred_spans and gold_spans must be the same length")

    tp = fp = fn = 0
    for preds, golds in zip(pred_spans, gold_spans):
        gold_keys = [_span_key(s) for s in golds]
        pred_keys = [_span_key(s) for s in preds]
        gold_pool = list(gold_keys)
        for k in pred_keys:
            if k in gold_pool:
                tp += 1
                gold_pool.remove(k)
            else:
                fp += 1
        fn += len(gold_pool)

    if tp == 0 and fp == 0 and fn == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
