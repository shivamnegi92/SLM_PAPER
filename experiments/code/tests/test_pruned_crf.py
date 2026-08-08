"""RED: PrunedGenerativeClassifier with a CRF slot head option."""
import torch
from transformers import GPT2Config

from slmpaper.pruned_model import PrunedGenerativeClassifier


def _cfg(n_layer=4):
    return GPT2Config(vocab_size=50, n_positions=32, n_embd=8, n_layer=n_layer, n_head=2)


def test_crf_model_computes_loss():
    model = PrunedGenerativeClassifier(_cfg(), depth=2, num_intents=3, num_tags=4, use_crf=True)
    assert model.crf is not None
    input_ids = torch.randint(0, 50, (2, 6))
    attention_mask = torch.ones(2, 6, dtype=torch.long)
    intent_labels = torch.tensor([0, 2])
    # first-subword supervised at a couple positions, rest -100
    slot_labels = torch.full((2, 6), -100)
    slot_labels[:, 0] = 1
    slot_labels[:, 2] = 2
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                intent_labels=intent_labels, slot_labels=slot_labels)
    assert out.loss is not None and out.loss.item() > 0


def test_softmax_model_has_no_crf():
    model = PrunedGenerativeClassifier(_cfg(), depth=2, num_intents=3, num_tags=4, use_crf=False)
    assert model.crf is None
