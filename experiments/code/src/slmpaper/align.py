"""Align word-level BIO tags to subword tokens (standard NER convention)."""
from __future__ import annotations

from typing import Optional


def align_labels_with_tokens(
    word_ids: list[Optional[int]],
    word_tags: list[str],
    tag2id: dict[str, int],
    label_all_subtokens: bool = False,
) -> list[int]:
    """Map per-word BIO tags onto subword tokens.

    Special tokens (word_id is None) always get -100. For a word split into
    multiple subwords, only the first subword gets the real label unless
    label_all_subtokens=True, in which case every subword of that word gets
    the same label. -100 is ignored by torch's CrossEntropyLoss by default.
    """
    aligned: list[int] = []
    previous_word_id: Optional[int] = None
    for word_id in word_ids:
        if word_id is None:
            aligned.append(-100)
        elif word_id != previous_word_id:
            aligned.append(tag2id[word_tags[word_id]])
        else:
            aligned.append(tag2id[word_tags[word_id]] if label_all_subtokens else -100)
        previous_word_id = word_id
    return aligned
