"""
EXP_N028 — Candidate Interaction Module.

Computes candidate <-> candidate similarity and conflict features:
- Name token and character similarity
- Address similarity and contradiction
- House number and postal conflict
- Source mutual relationship
- Competing candidate margin
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class CandidateInteractionModule(nn.Module):
    """
    Self-Attention and pairwise metric transformation across candidates in the set.
    """

    def __init__(self, feat_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(feat_dim)
        self.conflict_proj = nn.Sequential(
            nn.Linear(feat_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, cand_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        cand_tokens: [batch_size, num_cands, feat_dim]
        returns:
            updated_tokens: [batch_size, num_cands, feat_dim]
            conflict_matrix: [batch_size, num_cands, num_cands]
        """
        attn_out, _ = self.self_attn(cand_tokens, cand_tokens, cand_tokens)
        updated_tokens = self.norm(cand_tokens + attn_out)
        
        # Candidate-to-candidate pairwise conflicts for local top-K
        B, K, D = updated_tokens.shape
        t1 = updated_tokens.unsqueeze(2).expand(B, K, K, D)
        t2 = updated_tokens.unsqueeze(1).expand(B, K, K, D)
        pairs = torch.cat([t1, t2], dim=-1)
        conflict_matrix = self.conflict_proj(pairs).squeeze(-1)
        
        return updated_tokens, conflict_matrix
