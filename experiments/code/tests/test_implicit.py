"""RED: implicit-slot detection (the Paper 2 core, used in Paper 1 too)."""
import math

from slmpaper.implicit import is_implicit, implicit_slot_rate


def test_span_value_is_not_implicit():
    assert is_implicit("where is the great value milk", "great value milk") is False


def test_case_and_space_insensitive_match_is_not_implicit():
    assert is_implicit("Where is the MILK?", "milk") is False


def test_value_absent_from_text_is_implicit():
    # "self" implied by a possessive but never spoken.
    assert is_implicit("what are my points", "self") is True


def test_non_contiguous_tokens_are_implicit():
    # both words present but not as a contiguous span -> span tagger can't get it
    assert is_implicit("milk and fresh bread", "milk bread") is True


def test_implicit_slot_rate_counts_correctly():
    examples = [
        {"text": "where is the milk", "slots": {"item": ["milk"]}},          # not implicit
        {"text": "what are my points", "slots": {"self": ["self"]}},         # implicit
        {"text": "friday schedule", "slots": {"time": ["this friday"]}},     # implicit
    ]
    stats = implicit_slot_rate(examples)
    assert stats["total"] == 3
    assert stats["implicit"] == 2
    assert math.isclose(stats["rate"], 2 / 3)
    # ceiling = best possible span-tagger recall = 1 - rate
    assert math.isclose(stats["max_span_recall"], 1 / 3)


def test_implicit_slot_rate_empty():
    stats = implicit_slot_rate([])
    assert stats == {"total": 0, "implicit": 0, "rate": 0.0, "max_span_recall": 1.0}
