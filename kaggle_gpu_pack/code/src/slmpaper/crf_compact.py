"""Compact subword-level emissions/labels into contiguous word-level CRF inputs.

Our slot supervision marks non-first-subwords and padding with -100. A CRF needs
one contiguous emission per word with a length mask, so we gather the valid
(label != -100) positions per example and right-pad to the batch max word count.
"""
from __future__ import annotations

import torch


def compact_to_word_level(emissions: torch.Tensor, slot_labels: torch.Tensor):
    """emissions: (B, L, T), slot_labels: (B, L) with -100 for non-word/pad.

    Returns (word_emissions (B, W, T), word_tags (B, W), word_mask (B, W) bool),
    where W is the max number of valid words in the batch.
    """
    batch_size, seq_len, num_tags = emissions.shape
    valid = slot_labels != -100
    counts = valid.long().sum(dim=1)
    max_words = int(counts.max().item()) if batch_size else 0
    max_words = max(max_words, 1)

    w_emit = emissions.new_zeros((batch_size, max_words, num_tags))
    w_tags = slot_labels.new_zeros((batch_size, max_words))
    w_mask = torch.zeros((batch_size, max_words), dtype=torch.bool, device=emissions.device)

    for b in range(batch_size):
        idx = valid[b].nonzero(as_tuple=True)[0]
        n = idx.numel()
        if n == 0:
            w_mask[b, 0] = True  # keep at least one valid slot (all-O) to avoid empty seq
            continue
        w_emit[b, :n] = emissions[b, idx]
        w_tags[b, :n] = slot_labels[b, idx]
        w_mask[b, :n] = True
    return w_emit, w_tags, w_mask
