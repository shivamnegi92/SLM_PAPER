"""RED: JointBERT-style encoder model (shared backbone, intent + slot heads).

Unit-tested with a tiny synthetic DistilBertConfig -- no network/download
needed. The real distilbert-base-uncased is exercised separately in the
training script (integration-level, not a fast unit test).
"""
import torch
from transformers import DistilBertConfig

from slmpaper.encoder_model import JointEncoderModel


def _tiny_config():
    return DistilBertConfig(
        vocab_size=50, dim=8, n_layers=1, n_heads=2, hidden_dim=16,
        max_position_embeddings=32,
    )


def test_forward_shapes():
    model = JointEncoderModel(_tiny_config(), num_intents=3, num_tags=4)
    input_ids = torch.randint(0, 50, (2, 5))
    attention_mask = torch.ones(2, 5, dtype=torch.long)

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    assert out.intent_logits.shape == (2, 3)
    assert out.slot_logits.shape == (2, 5, 4)


def test_forward_with_labels_computes_combined_loss():
    model = JointEncoderModel(_tiny_config(), num_intents=3, num_tags=4)
    input_ids = torch.randint(0, 50, (2, 5))
    attention_mask = torch.ones(2, 5, dtype=torch.long)
    intent_labels = torch.tensor([0, 2])
    slot_labels = torch.full((2, 5), -100, dtype=torch.long)
    slot_labels[:, 0] = 1  # only first token supervised, rest ignored

    out = model(
        input_ids=input_ids, attention_mask=attention_mask,
        intent_labels=intent_labels, slot_labels=slot_labels,
    )
    assert out.loss is not None
    assert out.loss.ndim == 0  # scalar
    assert out.loss.item() > 0


def test_slot_loss_weight_scales_slot_component():
    torch.manual_seed(0)
    model = JointEncoderModel(_tiny_config(), num_intents=3, num_tags=4, slot_loss_weight=1.0)
    input_ids = torch.randint(0, 50, (2, 5))
    attention_mask = torch.ones(2, 5, dtype=torch.long)
    intent_labels = torch.tensor([0, 2])
    slot_labels = torch.zeros((2, 5), dtype=torch.long)

    torch.manual_seed(1)
    out_w1 = model(input_ids=input_ids, attention_mask=attention_mask,
                    intent_labels=intent_labels, slot_labels=slot_labels)

    model.slot_loss_weight = 5.0
    torch.manual_seed(1)
    out_w5 = model(input_ids=input_ids, attention_mask=attention_mask,
                    intent_labels=intent_labels, slot_labels=slot_labels)

    # Higher slot weight -> higher combined loss (same intent loss, scaled slot loss)
    assert out_w5.loss.item() > out_w1.loss.item()
