"""RED: linear probe training (frozen features -> intent label) and backbone truncation."""
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from slmpaper.probe import train_linear_probe
from slmpaper.pruning import truncate_gpt2_backbone


def test_train_linear_probe_fits_separable_synthetic_data():
    torch.manual_seed(0)
    # 2 classes, linearly separable in 4-dim feature space
    n = 200
    labels = torch.randint(0, 2, (n,))
    features = torch.randn(n, 4) + labels.unsqueeze(1).float() * 5.0

    train_feat, train_lab = features[:150], labels[:150]
    val_feat, val_lab = features[150:], labels[150:]

    acc = train_linear_probe(train_feat, train_lab, val_feat, val_lab,
                              num_classes=2, epochs=50)
    assert acc > 0.9  # trivially separable -> should nail it


def _tiny_gpt2_config():
    return GPT2Config(vocab_size=50, n_positions=32, n_embd=8, n_layer=6, n_head=2)


def test_truncate_gpt2_backbone_reduces_layers():
    model = GPT2LMHeadModel(_tiny_gpt2_config())
    truncated = truncate_gpt2_backbone(model, depth=3)
    assert len(truncated.transformer.h) == 3
    assert truncated.config.n_layer == 3


def test_truncated_model_still_runs_forward():
    model = GPT2LMHeadModel(_tiny_gpt2_config())
    truncated = truncate_gpt2_backbone(model, depth=2)
    input_ids = torch.randint(0, 50, (1, 5))
    out = truncated.transformer(input_ids=input_ids)
    assert out.last_hidden_state.shape == (1, 5, 8)
