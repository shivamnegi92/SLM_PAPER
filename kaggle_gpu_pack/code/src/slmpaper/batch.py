"""Turn a list of Examples into model-ready padded tensors."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .align import align_labels_with_tokens
from .datasets import Example


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    intent_labels: torch.Tensor
    slot_labels: torch.Tensor


def prepare_batch(
    examples: list[Example],
    tokenizer,
    intent2id: dict[str, int],
    tag2id: dict[str, int],
    max_length: int = 64,
    label_all_subtokens: bool = False,
) -> Batch:
    """Tokenize (word-split) + pad + align BIO tags to subwords."""
    all_tokens = [ex.tokens for ex in examples]
    encoding = tokenizer(
        all_tokens,
        is_split_into_words=True,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    slot_labels = []
    for i, ex in enumerate(examples):
        word_ids = encoding.word_ids(batch_index=i)
        aligned = align_labels_with_tokens(
            word_ids, ex.bio_tags, tag2id, label_all_subtokens=label_all_subtokens
        )
        slot_labels.append(aligned)

    intent_labels = torch.tensor([intent2id[ex.intent] for ex in examples], dtype=torch.long)
    slot_labels_t = torch.tensor(slot_labels, dtype=torch.long)

    return Batch(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"],
        intent_labels=intent_labels,
        slot_labels=slot_labels_t,
    )
