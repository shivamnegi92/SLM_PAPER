"""RED: align word-level BIO tags to subword tokens (standard NER convention).

Special tokens and non-first subwords of a word get label -100 (ignored by
the loss function). This works with any HF fast tokenizer's `word_ids()`
output, so it's testable without downloading a real tokenizer/model.
"""
from slmpaper.align import align_labels_with_tokens


def test_align_labels_special_tokens_get_ignore_index():
    # [CLS] w0 w0(##piece) w1 [SEP]
    word_ids = [None, 0, 0, 1, None]
    word_tags = ["B-city", "O"]
    tag2id = {"O": 0, "B-city": 1}

    aligned = align_labels_with_tokens(word_ids, word_tags, tag2id)
    assert aligned == [-100, 1, -100, 0, -100]


def test_align_labels_single_subword_per_word():
    word_ids = [None, 0, 1, 2, None]
    word_tags = ["O", "B-item", "I-item"]
    tag2id = {"O": 0, "B-item": 1, "I-item": 2}

    aligned = align_labels_with_tokens(word_ids, word_tags, tag2id)
    assert aligned == [-100, 0, 1, 2, -100]


def test_align_labels_label_all_subtokens_true():
    word_ids = [None, 0, 0, None]
    word_tags = ["B-city"]
    tag2id = {"O": 0, "B-city": 1}

    aligned = align_labels_with_tokens(word_ids, word_tags, tag2id, label_all_subtokens=True)
    assert aligned == [-100, 1, 1, -100]
