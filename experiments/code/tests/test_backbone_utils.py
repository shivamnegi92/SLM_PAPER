"""RED: backbone helpers must support non-GPT2 causal backbones (e.g., Nemotron/Llama-style)."""

from types import SimpleNamespace

import torch.nn as nn

from slmpaper.backbone import get_num_layers, truncate_backbone_layers


class _FakeGPTStyle(nn.Module):
    def __init__(self, n=6):
        super().__init__()
        self.h = nn.ModuleList([nn.Linear(2, 2) for _ in range(n)])
        self.config = SimpleNamespace(n_layer=n)


class _FakeLlamaStyle(nn.Module):
    def __init__(self, n=8):
        super().__init__()
        self.model = SimpleNamespace(layers=nn.ModuleList([nn.Linear(2, 2) for _ in range(n)]))
        self.config = SimpleNamespace(num_hidden_layers=n)


class _FakeTopLevelLayers(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(n)])
        self.config = SimpleNamespace(num_hidden_layers=n)


def test_get_num_layers_supports_num_hidden_layers_when_n_layer_missing():
    m = _FakeLlamaStyle(n=8)
    assert get_num_layers(m) == 8


def test_truncate_backbone_layers_supports_llama_style_layers():
    m = _FakeLlamaStyle(n=8)
    out = truncate_backbone_layers(m, depth=3)
    assert out is m
    assert len(m.model.layers) == 3
    assert m.config.num_hidden_layers == 3


def test_truncate_backbone_layers_supports_gpt_style_h():
    m = _FakeGPTStyle(n=6)
    out = truncate_backbone_layers(m, depth=2)
    assert out is m
    assert len(m.h) == 2
    assert m.config.n_layer == 2


def test_truncate_backbone_layers_supports_top_level_layers():
    m = _FakeTopLevelLayers(n=10)
    out = truncate_backbone_layers(m, depth=4)
    assert out is m
    assert len(m.layers) == 4
    assert m.config.num_hidden_layers == 4
