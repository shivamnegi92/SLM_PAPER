"""RED: MASSIVE (Amazon Science) local loader.

Verified schema (2026-08-07) from the official S3 tarball
(amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz),
file data/en-US.jsonl, one JSON object per line:
  {"id", "locale", "partition" (train/dev/test), "scenario", "intent",
   "utt" (raw text), "annot_utt" (slots marked as "[type : value]")}
"""
import json

from slmpaper.massive import parse_annotated_utterance, load_massive_local


def test_parse_annotated_utterance_single_slot():
    tokens, tags = parse_annotated_utterance("wake me up at [time : five am] this week")
    assert tokens == ["wake", "me", "up", "at", "five", "am", "this", "week"]
    assert tags == ["O", "O", "O", "O", "B-time", "I-time", "O", "O"]


def test_parse_annotated_utterance_two_slots():
    tokens, tags = parse_annotated_utterance(
        "wake me up at [time : nine am] on [date : friday]"
    )
    assert tokens == ["wake", "me", "up", "at", "nine", "am", "on", "friday"]
    assert tags == ["O", "O", "O", "O", "B-time", "I-time", "O", "B-date"]


def test_parse_annotated_utterance_no_slots():
    tokens, tags = parse_annotated_utterance("hello there")
    assert tokens == ["hello", "there"]
    assert tags == ["O", "O"]


def test_load_massive_local_filters_by_partition(tmp_path):
    rows = [
        {"partition": "train", "intent": "alarm_set", "utt": "wake me up at five am",
         "annot_utt": "wake me up at [time : five am]"},
        {"partition": "test", "intent": "alarm_set", "utt": "hello", "annot_utt": "hello"},
    ]
    path = tmp_path / "en-US.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))

    train_examples = load_massive_local(path, split="train")
    assert len(train_examples) == 1
    assert train_examples[0].intent == "alarm_set"
    assert train_examples[0].slots == {"time": ["five am"]}

    test_examples = load_massive_local(path, split="test")
    assert len(test_examples) == 1
    assert test_examples[0].slots == {}
