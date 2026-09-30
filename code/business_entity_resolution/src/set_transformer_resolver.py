"""
EXP_N028 — Set Transformer Resolver Architecture.

Implements Set Transformer / Perceiver cross-attention:
- S1 Entity Context Token acts as Query
- Candidate tokens act as Keys and Values
- Joint candidate states model subset cohesion
- Multi-task heads: candidate match logits + global cluster cardinality logits
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from cardinality_head import CardinalityHead
from candidate_interaction import CandidateInteractionModule

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class SetTransformerResolver(nn.Module):
    """
    Set Transformer Entity Resolver.
    """

    def __init__(
        self,
        s1_dim: int = 32,
        cand_dim: int = 32,
        hidden_dim: int = 128,
        num_heads: int = 4,
        max_k: int = 10,
    ):
        super().__init__()
        self.s1_proj = nn.Linear(s1_dim, hidden_dim)
        self.cand_proj = nn.Linear(cand_dim, hidden_dim)
        
        # Cross Attention: S1 queries candidates
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Candidate self-interaction
        self.cand_interaction = CandidateInteractionModule(feat_dim=hidden_dim, num_heads=num_heads)
        
        # Dual Heads
        self.cardinality_head = CardinalityHead(input_dim=hidden_dim, max_k=max_k)
        self.match_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        s1_rep: torch.Tensor,       # [B, 1, s1_dim]
        cand_reps: torch.Tensor,    # [B, K, cand_dim]
        mask: torch.Tensor | None = None, # [B, K] bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returns:
            match_logits: [B, K]
            cardinality_logits: [B, max_k + 1]
            conflict_matrix: [B, K, K]
        """
        B, K, _ = cand_reps.shape
        q = self.s1_proj(s1_rep)       # [B, 1, H]
        k = self.cand_proj(cand_reps)   # [B, K, H]
        v = k
        
        # Cross Attention: Entity context query
        context_out, _ = self.cross_attn(q, k, v)
        entity_context = self.norm1(q + context_out).squeeze(1)  # [B, H]
        
        # Candidate interaction
        inter_cands, conflict_mat = self.cand_interaction(k)  # [B, K, H]
        
        # Match Logits
        context_expanded = entity_context.unsqueeze(1).expand(B, K, -1)
        joint_state = torch.cat([inter_cands, context_expanded], dim=-1)  # [B, K, 2H]
        match_logits = self.match_head(joint_state).squeeze(-1)            # [B, K]
        
        # Cardinality Logits
        card_logits = self.cardinality_head(entity_context)               # [B, max_k + 1]
        
        return match_logits, card_logits, conflict_mat
