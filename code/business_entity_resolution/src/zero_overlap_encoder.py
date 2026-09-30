"""
EXP_N027 — Zero-Overlap Specialist Retrieval Encoder.

Specialist dual-encoder trained specifically on hard pairs where token Jaccard == 0.0,
enabling dense nearest-neighbor retrieval for radical business abbreviations,
acronyms, renamings, and transliterated names.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from char_entity_encoder import MultiScaleCharCNNBiGRU

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ZeroOverlapSpecialistEncoder(nn.Module):
    """
    Dedicated Zero-Overlap Specialist Encoder.
    Uses deeper character subword representations to bridge semantic gaps
    when lexical token matching is completely orthogonal.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        out_dim: int = 256,
    ):
        super().__init__()
        self.char_net = MultiScaleCharCNNBiGRU(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            filter_sizes=(2, 3, 4, 5, 6),
            num_filters=64,
        )
        self.metric_head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, char_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        base_rep = self.char_net(char_ids, mask)
        metric_rep = self.metric_head(base_rep)
        return F.normalize(metric_rep, p=2, dim=-1)
