"""JointBERT-style encoder model: shared transformer backbone + two heads.

Baseline B2 for M1 (see paper1_efficient_slm/PLAN.md). Matches the original
JointBERT convention (Chen et al. 2019): plain softmax token classification
for slots. CRF is a separate ablation reserved for M2/M4, not this baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, PretrainedConfig


@dataclass
class JointOutput:
    intent_logits: torch.Tensor
    slot_logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    intent_loss: Optional[torch.Tensor] = None
    slot_loss: Optional[torch.Tensor] = None


class JointEncoderModel(nn.Module):
    """Shared encoder + linear intent head + linear (softmax) slot head."""

    def __init__(
        self,
        config: PretrainedConfig,
        num_intents: int,
        num_tags: int,
        slot_loss_weight: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_config(config)
        hidden_size = getattr(config, "hidden_size", None) or getattr(config, "dim")
        self.dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(hidden_size, num_intents)
        self.slot_head = nn.Linear(hidden_size, num_tags)
        self.slot_loss_weight = slot_loss_weight

    @classmethod
    def from_pretrained_backbone(
        cls,
        model_name: str,
        num_intents: int,
        num_tags: int,
        slot_loss_weight: float = 1.0,
        dropout: float = 0.1,
    ) -> "JointEncoderModel":
        """Build with a real pretrained encoder backbone (heads randomly init)."""
        backbone = AutoModel.from_pretrained(model_name)
        model = cls(backbone.config, num_intents, num_tags,
                    slot_loss_weight=slot_loss_weight, dropout=dropout)
        model.encoder = backbone
        return model
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        intent_labels: Optional[torch.Tensor] = None,
        slot_labels: Optional[torch.Tensor] = None,
    ) -> JointOutput:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = self.dropout(hidden)

        pooled = hidden[:, 0, :]  # [CLS]/first-token pooling
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

        return JointOutput(
            intent_logits=intent_logits, slot_logits=slot_logits,
            loss=loss, intent_loss=intent_loss, slot_loss=slot_loss,
        )
