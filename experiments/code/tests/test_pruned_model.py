"""RED: pruned discriminative classifier -- truncated GPT2 backbone + heads."""
import torch
from transformers import GPT2Config

from slmpaper.pruned_model import PrunedGenerativeClassifier


def _tiny_config(n_layer=6):
    return GPT2Config(vocab_size=50, n_positions=32, n_embd=8, n_layer=n_layer, n_head=2)


def test_forward_shapes_at_full_depth():
    model = PrunedGenerativeClassifier(_tiny_config(), depth=6, num_intents=3, num_tags=4)
    input_ids = torch.randint(0, 50, (2, 5))
    attention_mask = torch.ones(2, 5, dtype=torch.long)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    assert out.intent_logits.shape == (2, 3)
    assert out.slot_logits.shape == (2, 5, 4)


def test_forward_at_pruned_depth_has_fewer_layers():
    model = PrunedGenerativeClassifier(_tiny_config(), depth=2, num_intents=3, num_tags=4)
    assert len(model.backbone.h) == 2
    input_ids = torch.randint(0, 50, (1, 4))
    attention_mask = torch.ones(1, 4, dtype=torch.long)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    assert out.intent_logits.shape == (1, 3)


def test_forward_with_labels_computes_loss():
    model = PrunedGenerativeClassifier(_tiny_config(), depth=3, num_intents=3, num_tags=4)
    input_ids = torch.randint(0, 50, (2, 5))
    attention_mask = torch.ones(2, 5, dtype=torch.long)
    intent_labels = torch.tensor([0, 2])
    slot_labels = torch.zeros((2, 5), dtype=torch.long)
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                intent_labels=intent_labels, slot_labels=slot_labels)
    assert out.loss is not None and out.loss.item() > 0


def test_forward_handles_bfloat16_backbone_outputs():
    model = PrunedGenerativeClassifier(_tiny_config(), depth=2, num_intents=3, num_tags=4)
    model.backbone = model.backbone.to(torch.bfloat16)

    input_ids = torch.randint(0, 50, (1, 4))
    attention_mask = torch.ones(1, 4, dtype=torch.long)

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    assert out.intent_logits.shape == (1, 3)
