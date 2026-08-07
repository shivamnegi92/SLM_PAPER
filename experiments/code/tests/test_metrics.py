"""RED: evaluation metrics."""
import math

from slmpaper.metrics import intent_accuracy, span_f1


def test_intent_accuracy_basic():
    preds = ["a", "b", "c", "a"]
    golds = ["a", "b", "x", "a"]
    assert math.isclose(intent_accuracy(preds, golds), 0.75)


def test_intent_accuracy_empty_is_zero():
    assert intent_accuracy([], []) == 0.0


def test_intent_accuracy_length_mismatch_raises():
    try:
        intent_accuracy(["a"], ["a", "b"])
    except ValueError:
        return
    raise AssertionError("expected ValueError on length mismatch")


def test_span_f1_perfect():
    gold = [[{"type": "item", "start": 0, "end": 1}]]
    pred = [[{"type": "item", "start": 0, "end": 1}]]
    r = span_f1(pred, gold)
    assert math.isclose(r["precision"], 1.0)
    assert math.isclose(r["recall"], 1.0)
    assert math.isclose(r["f1"], 1.0)


def test_span_f1_half_precision_half_recall():
    # gold has 2 spans, pred has 2 spans, 1 correct.
    gold = [[
        {"type": "item", "start": 0, "end": 1},
        {"type": "dept", "start": 2, "end": 3},
    ]]
    pred = [[
        {"type": "item", "start": 0, "end": 1},   # TP
        {"type": "dept", "start": 4, "end": 5},   # FP (wrong span)
    ]]
    r = span_f1(pred, gold)
    assert math.isclose(r["precision"], 0.5)
    assert math.isclose(r["recall"], 0.5)
    assert math.isclose(r["f1"], 0.5)


def test_span_f1_type_must_match():
    gold = [[{"type": "item", "start": 0, "end": 1}]]
    pred = [[{"type": "dept", "start": 0, "end": 1}]]
    r = span_f1(pred, gold)
    assert r["f1"] == 0.0


def test_span_f1_no_entities_anywhere_is_one():
    # Convention: nothing to predict and nothing predicted -> perfect.
    r = span_f1([[]], [[]])
    assert r["f1"] == 1.0
