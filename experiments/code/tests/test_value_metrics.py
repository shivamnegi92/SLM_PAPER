"""RED: value-based slot F1 (type + surface value), for the generative baseline.

The generative model emits slot *values*, not token positions, so span-index
F1 doesn't apply. This compares {type: [values]} dicts, which both the
generative model (emitted) and encoder (spans -> values) can produce.
"""
from slmpaper.value_metrics import value_slot_f1


def test_value_slot_f1_perfect():
    pred = [{"city": ["boston"], "date": ["april"]}]
    gold = [{"city": ["boston"], "date": ["april"]}]
    r = value_slot_f1(pred, gold)
    assert r["f1"] == 1.0


def test_value_slot_f1_partial():
    pred = [{"city": ["boston"]}]
    gold = [{"city": ["boston"], "date": ["april"]}]
    r = value_slot_f1(pred, gold)
    assert r["tp"] == 1 and r["fn"] == 1 and r["fp"] == 0
    assert abs(r["recall"] - 0.5) < 1e-9
    assert r["precision"] == 1.0


def test_value_slot_f1_wrong_value_counts_fp_and_fn():
    pred = [{"city": ["denver"]}]
    gold = [{"city": ["boston"]}]
    r = value_slot_f1(pred, gold)
    assert r["tp"] == 0 and r["fp"] == 1 and r["fn"] == 1


def test_value_slot_f1_all_empty_is_one():
    r = value_slot_f1([{}], [{}])
    assert r["f1"] == 1.0
