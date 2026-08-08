"""Linear-chain CRF layer (batched, masked) for the slot-tagging head ablation.

Standard CRF: learnable transition scores + start/end transitions, forward
algorithm for the log-partition, and Viterbi for decoding. Convention: the
first timestep of every example must be valid (mask[:, 0] == True).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LinearChainCRF(nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)
        self.start_transitions = nn.Parameter(torch.randn(num_tags) * 0.1)
        self.end_transitions = nn.Parameter(torch.randn(num_tags) * 0.1)

    def _gold_score(self, emissions, tags, mask):
        # emissions: (B, L, T); tags: (B, L); mask: (B, L) bool
        batch_size, seq_len, _ = emissions.shape
        mask_f = mask.float()
        score = self.start_transitions[tags[:, 0]]
        score = score + emissions[torch.arange(batch_size), 0, tags[:, 0]]
        for i in range(1, seq_len):
            trans = self.transitions[tags[:, i - 1], tags[:, i]]
            emit = emissions[torch.arange(batch_size), i, tags[:, i]]
            score = score + (trans + emit) * mask_f[:, i]
        # end transition at each example's last valid position
        last_idx = mask.long().sum(dim=1) - 1
        last_tags = tags[torch.arange(batch_size), last_idx]
        score = score + self.end_transitions[last_tags]
        return score

    def _log_partition(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (B, T)
        for i in range(1, seq_len):
            # broadcast: (B, T_prev, 1) + (T_prev, T_next) + (B, 1, T_next)
            broadcast = (alpha.unsqueeze(2)
                         + self.transitions.unsqueeze(0)
                         + emissions[:, i].unsqueeze(1))
            new_alpha = torch.logsumexp(broadcast, dim=1)  # (B, T)
            m = mask[:, i].unsqueeze(1).float()
            alpha = m * new_alpha + (1 - m) * alpha
        alpha = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)  # (B,)

    def neg_log_likelihood(self, emissions, tags, mask):
        """Per-token-normalized NLL, so its scale is comparable to a token-mean
        softmax cross-entropy (otherwise the CRF term, which sums over the whole
        sequence, dominates the joint loss and starves the intent head).
        """
        gold = self._gold_score(emissions, tags, mask)
        logz = self._log_partition(emissions, mask)
        n_tokens = mask.float().sum().clamp(min=1.0)
        return (logz - gold).sum() / n_tokens

    @torch.no_grad()
    def decode(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape
        lengths = mask.long().sum(dim=1)
        paths = []
        for b in range(batch_size):
            L = int(lengths[b].item())
            emit = emissions[b, :L]  # (L, T)
            score = self.start_transitions + emit[0]  # (T,)
            backptrs = []
            for i in range(1, L):
                broadcast = score.unsqueeze(1) + self.transitions  # (T_prev, T_next)
                best_prev = broadcast.argmax(dim=0)  # (T_next,)
                score = broadcast.max(dim=0).values + emit[i]
                backptrs.append(best_prev)
            score = score + self.end_transitions
            best_last = int(score.argmax().item())
            path = [best_last]
            for bp in reversed(backptrs):
                best_last = int(bp[best_last].item())
                path.append(best_last)
            path.reverse()
            paths.append(path)
        return paths
