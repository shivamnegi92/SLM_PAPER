"""RED: turn a list of Examples into model-ready padded tensors.

Uses a real (cached) fast tokenizer for word_ids() -- distilbert-base-uncased
was already downloaded in this session, so this runs offline from cache.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoTokenizer

from slmpaper.datasets import Example
from slmpaper.batch import prepare_batch


def _tokenizer():
    return AutoTokenizer.from_pretrained("distilbert-base-uncased")


def test_prepare_batch_shapes_and_padding():
    tok = _tokenizer()
    examples = [
        Example(text="find a flight", tokens=["find", "a", "flight"], intent="flight",
                slots={}, bio_tags=["O", "O", "O"]),
        Example(text="hi", tokens=["hi"], intent="greeting", slots={}, bio_tags=["O"]),
    ]
    intent2id = {"flight": 0, "greeting": 1}
    tag2id = {"O": 0}

    batch = prepare_batch(examples, tok, intent2id, tag2id)

    assert batch.input_ids.shape[0] == 2
    assert batch.input_ids.shape == batch.attention_mask.shape
    assert batch.slot_labels.shape == batch.input_ids.shape
    assert batch.intent_labels.tolist() == [0, 1]
    # shorter example's padded positions should be -100 (ignored) in slot labels
    assert (batch.slot_labels[1] == -100).sum() > 0


def test_prepare_batch_aligns_bio_tags_to_first_subword():
    tok = _tokenizer()
    examples = [
        Example(text="find samsung galaxy", tokens=["find", "samsung", "galaxy"],
                intent="search", slots={"item": ["samsung galaxy"]},
                bio_tags=["O", "B-item", "I-item"]),
    ]
    intent2id = {"search": 0}
    tag2id = {"O": 0, "B-item": 1, "I-item": 2}

    batch = prepare_batch(examples, tok, intent2id, tag2id)
    labels = batch.slot_labels[0].tolist()
    real_labels = [l for l in labels if l != -100]
    # first real (non-special/non-continuation) label per word should reconstruct O,B-item,I-item
    assert real_labels == [0, 1, 2]
