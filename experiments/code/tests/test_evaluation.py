"""RED: decode model predictions -> word-level tags + eval metrics."""
from slmpaper.evaluation import decode_slot_row, tags_to_spans


def test_decode_slot_row_filters_ignore_index():
    pred_ids = [0, 1, 2, 0, 0]          # raw subword predictions
    gold_label_ids = [-100, 1, -100, 2, -100]  # only positions 1 and 3 are real words
    id2tag = {0: "O", 1: "B-item", 2: "I-item"}

    pred_tags, gold_tags = decode_slot_row(pred_ids, gold_label_ids, id2tag)
    assert gold_tags == ["B-item", "I-item"]
    assert pred_tags == ["B-item", "O"]  # preds taken at the same real positions


def test_tags_to_spans_uses_bio_conversion():
    spans = tags_to_spans(["O", "B-item", "I-item", "O", "B-date"])
    keys = {(s["type"], s["start"], s["end"]) for s in spans}
    assert keys == {("item", 1, 3), ("date", 4, 5)}
