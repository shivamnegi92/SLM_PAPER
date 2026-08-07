"""RED: HF-style BIO record adapter + end-to-end pipeline stats."""
import math

from slmpaper.pipeline import records_to_examples, dataset_stats


# A tiny in-memory dataset shaped like HF slot-filling datasets (ATIS/SNIPS/MASSIVE):
# integer ner_tags decoded via id2tag, whitespace tokens, string intent.
FAKE_RECORDS = [
    {"tokens": ["play", "some", "jazz"], "ner_tags": [0, 0, 1], "intent": "PlayMusic"},
    {"tokens": ["what", "is", "my", "balance"], "ner_tags": [0, 0, 0, 0], "intent": "AccountQuery"},
]
ID2TAG = {0: "O", 1: "B-genre"}


def test_records_to_examples_decodes_int_tags():
    exs = records_to_examples(
        FAKE_RECORDS, tokens_key="tokens", tags_key="ner_tags",
        intent_key="intent", id2tag=ID2TAG,
    )
    assert len(exs) == 2
    assert exs[0].intent == "PlayMusic"
    assert exs[0].slots == {"genre": ["jazz"]}
    assert exs[1].slots == {}


def test_records_to_examples_accepts_string_tags():
    recs = [{"toks": ["play", "jazz"], "tags": ["O", "B-genre"], "lbl": "PlayMusic"}]
    exs = records_to_examples(recs, tokens_key="toks", tags_key="tags", intent_key="lbl")
    assert exs[0].slots == {"genre": ["jazz"]}


def test_dataset_stats_reports_counts_and_implicit_rate():
    exs = records_to_examples(
        FAKE_RECORDS, tokens_key="tokens", tags_key="ner_tags",
        intent_key="intent", id2tag=ID2TAG,
    )
    stats = dataset_stats(exs)
    assert stats["num_examples"] == 2
    assert stats["num_intents"] == 2
    assert stats["num_slot_types"] == 1
    # "jazz" is a real span -> not implicit -> rate 0
    assert math.isclose(stats["implicit"]["rate"], 0.0)
