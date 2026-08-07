"""RED: BIO <-> span conversion behavior."""
from slmpaper.bio import spans_to_bio, bio_to_spans, find_span


def test_spans_to_bio_single_multitoken_span():
    tokens = ["find", "the", "samsung", "galaxy", "in", "electronics"]
    spans = [
        {"type": "item", "start": 2, "end": 4},   # samsung galaxy
        {"type": "dept", "start": 5, "end": 6},    # electronics
    ]
    assert spans_to_bio(tokens, spans) == [
        "O", "O", "B-item", "I-item", "O", "B-dept",
    ]


def test_spans_to_bio_empty_is_all_O():
    tokens = ["hello", "world"]
    assert spans_to_bio(tokens, []) == ["O", "O"]


def test_bio_to_spans_roundtrip():
    tokens = ["find", "the", "samsung", "galaxy", "in", "electronics"]
    tags = ["O", "O", "B-item", "I-item", "O", "B-dept"]
    spans = bio_to_spans(tokens, tags)
    assert spans == [
        {"type": "item", "start": 2, "end": 4, "text": "samsung galaxy"},
        {"type": "dept", "start": 5, "end": 6, "text": "electronics"},
    ]


def test_bio_to_spans_ignores_orphan_I_tag():
    # An I- tag with no preceding B- of the same type must not crash and
    # must not fabricate a span from nothing.
    tokens = ["a", "b", "c"]
    tags = ["O", "I-item", "O"]
    assert bio_to_spans(tokens, tags) == []


def test_find_span_returns_first_match():
    tokens = ["show", "the", "great", "value", "milk"]
    assert find_span(tokens, ["great", "value", "milk"]) == (2, 5)


def test_find_span_missing_returns_none():
    tokens = ["show", "the", "milk"]
    assert find_span(tokens, ["bread"]) is None
