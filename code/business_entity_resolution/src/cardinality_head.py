"""
EXP_N028 — Cardinality Head Module.

Predicts the discrete set cardinality distribution P(|Matches| = k) for k in [0, 1, ..., max_cardinality].
Conditions entity subset selection on the predicted global cluster size.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class CardinalityHead(nn.Module):
    """
    Predicts probability distribution over cluster size [0..max_k].
    Trained via Cross-Entropy on true cluster cardinality.
    """

    def __init__(self, input_dim: int = 128, max_k: int = 10):
        super().__init__()
        self.max_k = max_k
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, max_k + 1),
        )

    def forward(self, entity_context_token: torch.Tensor) -> torch.Tensor:
        """
        entity_context_token: [batch_size, input_dim]
        returns: [batch_size, max_k + 1] logits
        """
        return self.net(entity_context_token)

    def predict_expected_cardinality(self, entity_context_token: torch.Tensor) -> torch.Tensor:
        """
        Returns expected cardinality E[K] = sum(k * P(k)).
        """
        probs = F.softmax(self.forward(entity_context_token), dim=-1)
        k_values = torch.arange(0, self.max_k + 1, device=probs.device, dtype=torch.float32)
        return torch.sum(probs * k_values, dim=-1)
