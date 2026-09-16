"""Pruned discriminative classifier: truncated causal decoder + intent/slot heads.

This IS "ours" -- the C1/C2 payoff. A generative decoder backbone, cut to a
probe-selected depth, converted into a single-pass discriminative model
(mean-pooled intent head + per-token softmax slot head). No autoregressive
decoding, no parse failures by construction, and a fraction of the compute of
the full-depth generative model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, PretrainedConfig

from .backbone import truncate_backbone_layers
from .crf import LinearChainCRF
from .crf_compact import compact_to_word_level
from .probe import pool_hidden_states


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
        config: PretrainedConfig,
        depth: int,
        num_intents: int,
        num_tags: int,
        slot_loss_weight: float = 1.0,
        dropout: float = 0.1,
        use_crf: bool = False,
    ):
        super().__init__()
        backbone = AutoModel.from_config(config)
        truncate_backbone_layers(backbone, depth)
        self.backbone = backbone

        hidden_size = getattr(config, "n_embd", None) or getattr(config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden size from config (expected n_embd or hidden_size)")

        self.dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(hidden_size, num_intents)
        self.slot_head = nn.Linear(hidden_size, num_tags)
        self.slot_loss_weight = slot_loss_weight
        self.crf = LinearChainCRF(num_tags) if use_crf else None

    @classmethod
    def from_pretrained_backbone(
        cls,
        model_path: str,
        depth: int,
        num_intents: int,
        num_tags: int,
        slot_loss_weight: float = 1.0,
        dropout: float = 0.1,
        use_crf: bool = False,
    ) -> "PrunedGenerativeClassifier":
        """Load pretrained decoder backbone, truncate, and attach discriminative heads."""
        pretrained = AutoModel.from_pretrained(model_path)
        model = cls(
            pretrained.config,
            depth=depth,
            num_intents=num_intents,
            num_tags=num_tags,
            slot_loss_weight=slot_loss_weight,
            dropout=dropout,
            use_crf=use_crf,
        )
        model.backbone = truncate_backbone_layers(pretrained, depth)
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

        # Modern local checkpoints can run bf16 on CPU while classification
        # heads remain fp32 by default; align hidden dtype to head dtype to
        # avoid matmul dtype mismatch.
        hidden = hidden.to(self.intent_head.weight.dtype)

        pooled = pool_hidden_states(hidden, attention_mask)
        intent_logits = self.intent_head(pooled)
        slot_logits = self.slot_head(hidden)

        loss = intent_loss = slot_loss = None
        if intent_labels is not None and slot_labels is not None:
            intent_loss = nn.functional.cross_entropy(intent_logits, intent_labels)
            if self.crf is not None:
                w_emit, w_tags, w_mask = compact_to_word_level(slot_logits, slot_labels)
                slot_loss = self.crf.neg_log_likelihood(w_emit, w_tags, w_mask)
            else:
                slot_loss = nn.functional.cross_entropy(
                    slot_logits.view(-1, slot_logits.size(-1)),
                    slot_labels.view(-1),
                    ignore_index=-100,
                )
            loss = intent_loss + self.slot_loss_weight * slot_loss

        return PrunedOutput(
            intent_logits=intent_logits,
            slot_logits=slot_logits,
            loss=loss,
            intent_loss=intent_loss,
            slot_loss=slot_loss,
        )
