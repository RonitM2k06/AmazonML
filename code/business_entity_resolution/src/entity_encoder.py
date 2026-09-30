"""
EXP_N020: Phase 2 & 3 — Multi-View Latent Entity Encoder & Neural Matcher.

Takes candidate pair features, base probabilities, intra-pool ranks,
and Latent Entity Memory representations, and projects them through:
- Multi-View Attention Head
- Relational Cross-Source Memory Block
- Contrastive Entity Matching Projector
- Final Multi-Label Calibrated Match Logit Head
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiViewEntityEncoder(nn.Module):
    """
    Encodes candidate observations and latent entity memory into a multi-view representation.
    Views:
    1. Lexical & character similarity view
    2. Address & structured location view
    3. Neural semantic cosine view
    4. Latent memory anchor & centroid view
    5. Cross-source relational support view
    """
    def __init__(self, d_in: int = 71, d_model: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # View encoders
        self.view_proj = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Multi-view self-attention over candidate pool
        self.pool_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-forward refinement
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Contrastive projection head (for InfoNCE hard-case learning)
        self.contrastive_proj = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
        )

        # Final multi-label classification head
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        features: torch.Tensor,              # [B, K, d_in]
        apply_attention: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        features: [B, K, d_in]
        Returns:
            logits: [B, K]
            contrastive_embs: [B, K, 64] (L2-normalized)
        """
        B, K, _ = features.shape
        h = self.view_proj(features) # [B, K, d_model]

        if apply_attention:
            attn_out, _ = self.pool_attention(h, h, h)
            h = self.norm1(h + attn_out)
            h = self.norm2(h + self.ffn(h))

        logits = self.head(h).squeeze(-1) # [B, K]
        proj = self.contrastive_proj(h)   # [B, K, 64]
        contrastive_embs = F.normalize(proj, dim=-1)

        return logits, contrastive_embs
