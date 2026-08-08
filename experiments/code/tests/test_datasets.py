"""RED: dataset normalizer -> common Example schema."""
from slmpaper.datasets import normalize_bio_record, Example


def test_normalize_bio_record_builds_example():
    ex = normalize_bio_record(
        tokens=["find", "samsung", "galaxy", "in", "electronics"],
        bio_tags=["O", "B-item", "I-item", "O", "B-dept"],
        intent="find_item",
    )
    assert isinstance(ex, Example)
    assert ex.text == "find samsung galaxy in electronics"
    assert ex.intent == "find_item"
    assert ex.slots == {"item": ["samsung galaxy"], "dept": ["electronics"]}
    assert ex.bio_tags == ["O", "B-item", "I-item", "O", "B-dept"]


def test_normalize_bio_record_no_slots():
    ex = normalize_bio_record(
        tokens=["hi", "there"],
        bio_tags=["O", "O"],
        intent="greeting",
    )
    assert ex.slots == {}
    assert ex.text == "hi there"


def test_normalize_bio_record_length_mismatch_raises():
    try:
        normalize_bio_record(tokens=["a", "b"], bio_tags=["O"], intent="x")
    except ValueError:
        return
    raise AssertionError("expected ValueError on token/tag length mismatch")


def test_normalize_repeated_slot_type_collects_multiple_values():
    ex = normalize_bio_record(
        tokens=["milk", "and", "bread"],
        bio_tags=["B-item", "O", "B-item"],
        intent="list",
    )
    assert ex.slots == {"item": ["milk", "bread"]}
