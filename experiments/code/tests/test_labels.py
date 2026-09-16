"""RED: label vocabularies for intent + BIO slot tags."""
from slmpaper.datasets import Example
from slmpaper.labels import build_intent_vocab, build_tag_vocab, has_only_known_bio_tags


def _ex(intent, tags):
    return Example(text=" ".join(["w"] * len(tags)), tokens=["w"] * len(tags),
                    intent=intent, slots={}, bio_tags=tags)


def test_build_intent_vocab_sorted_deterministic():
    examples = [_ex("b_intent", ["O"]), _ex("a_intent", ["O"]), _ex("b_intent", ["O"])]
    intent2id = build_intent_vocab(examples)
    assert intent2id == {"a_intent": 0, "b_intent": 1}


def test_build_tag_vocab_includes_both_bi_variants_per_type():
    # Even though train only shows B-date (never I-date), the vocab must include
    # I-date too, so an unseen I-date in the test set never KeyErrors.
    examples = [_ex("x", ["O", "B-city", "I-city"]), _ex("x", ["O", "B-date"])]
    tag2id = build_tag_vocab(examples)
    assert set(tag2id) == {"O", "B-city", "I-city", "B-date", "I-date"}
    assert tag2id["O"] == 0  # O always id 0 by convention (pad-safe default)


def test_has_only_known_bio_tags_filters_unknown_tags():
    train = [_ex("x", ["O", "B-city", "I-city"])]
    tag2id = build_tag_vocab(train)

    ok = _ex("x", ["O", "B-city", "I-city"]) 
    bad = _ex("x", ["O", "B-return_date.month_name", "I-return_date.month_name"])

    assert has_only_known_bio_tags(ok, tag2id) is True
    assert has_only_known_bio_tags(bad, tag2id) is False
