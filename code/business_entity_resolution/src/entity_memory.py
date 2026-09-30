"""
EXP_N026: Multi-View Neural Entity Memory & Permutation-Invariant Aggregator.

Constructs permutation-invariant latent entity memories from:
- S1 canonical record
- Accepted S2/S3 observations
- Cross-source relational embeddings
- View-specific prototypes: z_name, z_address, z_joint, z_cross_source
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class EntityMemoryPooler(nn.Module):
    """
    Permutation-invariant DeepSets / Attention Pooler over variable-length accepted observations.
    Computes an adaptive latent entity centroid vector: z_entity ∈ R^d.
    """
    def __init__(self, d_in: int = 57, d_model: int = 128, n_heads: int = 4, dropout: float = 0.05):
        super().__init__()
        self.d_model = d_model

        # View projections
        self.name_proj = nn.Sequential(
            nn.Linear(29, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.addr_proj = nn.Sequential(
            nn.Linear(17, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.joint_proj = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Permutation-invariant attention query
        self.entity_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

        # Output fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, observations: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        observations: [B, M, d_in] where M is the number of accepted observations
        mask: [B, M] bool, True indicates padded position
        Returns:
            z_entity: [B, d_model] L2-normalized entity prototype
        """
        B, M, _ = observations.shape
        obs_name = observations[:, :, :29]
        obs_addr = observations[:, :, 29:46]

        z_name = self.name_proj(obs_name)   # [B, M, d_model]
        z_addr = self.addr_proj(obs_addr)   # [B, M, d_model]
        z_joint = self.joint_proj(observations) # [B, M, d_model]

        # Order-invariant attention pooling with learnable entity prototype query
        q = self.entity_query.expand(B, -1, -1) # [B, 1, d_model]
        pool_out, _ = self.mha(q, z_joint, z_joint, key_padding_mask=mask)
        z_joint_pooled = self.norm(pool_out.squeeze(1)) # [B, d_model]

        # Masked mean for name and address prototypes
        if mask is not None:
            w = (~mask).float().unsqueeze(-1) # [B, M, 1]
            z_name_pooled = (z_name * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
            z_addr_pooled = (z_addr * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        else:
            z_name_pooled = z_name.mean(dim=1)
            z_addr_pooled = z_addr.mean(dim=1)

        fused = self.fusion(torch.cat([z_name_pooled, z_addr_pooled, z_joint_pooled], dim=-1))
        return F.normalize(fused, dim=-1)
