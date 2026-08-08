"""Decode model predictions to word-level tags and compute eval metrics."""
from __future__ import annotations

import torch

from .bio import bio_to_spans
from .metrics import intent_accuracy, span_f1


def decode_slot_row(
    pred_ids: list[int],
    gold_label_ids: list[int],
    id2tag: dict[int, str],
) -> tuple[list[str], list[str]]:
    """Filter to real (non -100) word positions; return (pred_tags, gold_tags)."""
    pred_tags, gold_tags = [], []
    for p, g in zip(pred_ids, gold_label_ids):
        if g == -100:
            continue
        gold_tags.append(id2tag[g])
        pred_tags.append(id2tag.get(p, "O"))
    return pred_tags, gold_tags


def tags_to_spans(tags: list[str]) -> list[dict]:
    """Word-level BIO tags -> typed spans (placeholder tokens; only indices matter)."""
    return bio_to_spans([""] * len(tags), tags)


@torch.no_grad()
def evaluate_model(model, batches, id2intent, id2tag, device="cpu") -> dict:
    """Run model over prepared batches; return intent acc + slot span F1."""
    model.eval()
    pred_intents, gold_intents = [], []
    pred_span_lists, gold_span_lists = [], []

    for batch in batches:
        out = model(
            input_ids=batch.input_ids.to(device),
            attention_mask=batch.attention_mask.to(device),
        )
        intent_pred = out.intent_logits.argmax(-1).tolist()
        slot_pred = out.slot_logits.argmax(-1).tolist()

        for i in range(len(intent_pred)):
            pred_intents.append(id2intent.get(intent_pred[i], ""))
            gold_intents.append(id2intent[batch.intent_labels[i].item()])
            p_tags, g_tags = decode_slot_row(
                slot_pred[i], batch.slot_labels[i].tolist(), id2tag
            )
            pred_span_lists.append(tags_to_spans(p_tags))
            gold_span_lists.append(tags_to_spans(g_tags))

    return {
        "intent_accuracy": intent_accuracy(pred_intents, gold_intents),
        "slot_f1": span_f1(pred_span_lists, gold_span_lists),
    }
