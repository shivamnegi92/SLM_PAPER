"""Pruned discriminative classifier: truncated GPT2 backbone + intent/slot heads.

This IS "ours" -- the C1/C2 payoff. A generative decoder backbone, cut to a
probe-selected depth, converted into a single-pass discriminative model
(mean-pooled intent head + per-token softmax slot head). No autoregressive
decoding, no parse failures by construction, and a fraction of the compute of
the full-depth generative model.

Note: causal (left-to-right) attention means each token's slot prediction only
sees left context -- a real, honestly-reported tradeoff vs a bidirectional
encoder (see PLAN.md M4 ablations), not hidden from the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

from .probe import pool_hidden_states
from .pruning import truncate_gpt2_backbone


@dataclass
class PrunedOutput:
    intent_logits: torch.Tensor
    slot_logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    intent_loss: Optional[torch.Tensor] = None
    slot_loss: Optional[torch.Tensor] = None


class PrunedGenerativeClassifier(nn.Module):
    def __init__(
        self,
        config: GPT2Config,
        depth: int,
        num_intents: int,
        num_tags: int,
        slot_loss_weight: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        backbone = GPT2Model(config)
        if depth < config.n_layer:
            backbone.h = backbone.h[:depth]
            backbone.config.n_layer = depth
        self.backbone = backbone
        hidden_size = config.n_embd
        self.dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(hidden_size, num_intents)
        self.slot_head = nn.Linear(hidden_size, num_tags)
        self.slot_loss_weight = slot_loss_weight

    @classmethod
    def from_pretrained_backbone(
        cls, model_path: str, depth: int, num_intents: int, num_tags: int,
        slot_loss_weight: float = 1.0, dropout: float = 0.1,
    ) -> "PrunedGenerativeClassifier":
        """Load a real pretrained GPT2 backbone, then truncate + attach heads."""
        pretrained = GPT2Model.from_pretrained(model_path)
        model = cls(pretrained.config, depth=depth, num_intents=num_intents,
                    num_tags=num_tags, slot_loss_weight=slot_loss_weight, dropout=dropout)
        model.backbone = truncate_gpt2_backbone_model(pretrained, depth)
        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        intent_labels: Optional[torch.Tensor] = None,
        slot_labels: Optional[torch.Tensor] = None,
    ) -> PrunedOutput:
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = self.dropout(hidden)

        pooled = pool_hidden_states(hidden, attention_mask)
        intent_logits = self.intent_head(pooled)
        slot_logits = self.slot_head(hidden)

        loss = intent_loss = slot_loss = None
        if intent_labels is not None and slot_labels is not None:
            intent_loss = nn.functional.cross_entropy(intent_logits, intent_labels)
            slot_loss = nn.functional.cross_entropy(
                slot_logits.view(-1, slot_logits.size(-1)), slot_labels.view(-1),
                ignore_index=-100,
            )
            loss = intent_loss + self.slot_loss_weight * slot_loss

        return PrunedOutput(
            intent_logits=intent_logits, slot_logits=slot_logits,
            loss=loss, intent_loss=intent_loss, slot_loss=slot_loss,
        )


def truncate_gpt2_backbone_model(backbone: GPT2Model, depth: int) -> GPT2Model:
    """Same truncation as pruning.truncate_gpt2_backbone, but for a bare GPT2Model
    (not GPT2LMHeadModel) -- reused here to avoid loading the LM head we don't need.
    """
    n_available = len(backbone.h)
    if not (1 <= depth <= n_available):
        raise ValueError(f"depth must be in [1, {n_available}], got {depth}")
    backbone.h = backbone.h[:depth]
    backbone.config.n_layer = depth
    return backbone
