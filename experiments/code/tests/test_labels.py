"""RED: label vocabularies for intent + BIO slot tags."""
from slmpaper.datasets import Example
from slmpaper.labels import build_intent_vocab, build_tag_vocab


def _ex(intent, tags):
    return Example(text=" ".join(["w"] * len(tags)), tokens=["w"] * len(tags),
                    intent=intent, slots={}, bio_tags=tags)


def test_build_intent_vocab_sorted_deterministic():
    examples = [_ex("b_intent", ["O"]), _ex("a_intent", ["O"]), _ex("b_intent", ["O"])]
    intent2id = build_intent_vocab(examples)
    assert intent2id == {"a_intent": 0, "b_intent": 1}


def test_build_tag_vocab_includes_o_and_all_bio_tags():
    examples = [_ex("x", ["O", "B-city", "I-city"]), _ex("x", ["O", "B-date"])]
    tag2id = build_tag_vocab(examples)
    assert set(tag2id) == {"O", "B-city", "I-city", "B-date"}
    assert tag2id["O"] == 0  # O always id 0 by convention (pad-safe default)
