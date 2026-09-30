"""
EXP_N025: Cross-Source Entity Translation Heads & Latent Reconstruction Model.

Implements directional source translation transformations:
T_S1->S2, T_S1->S3, T_S2->S1, T_S3->S1, T_S2->S3, T_S3->S2
predicting the expected latent representation in target source space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class CrossSourceTranslationHead(nn.Module):
    """
    Directional non-linear translation head mapping latent representation from src -> tgt.
    Residual architecture with LayerNorm and projection.
    """
    def __init__(self, d_model: int = 256, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, z_src: torch.Tensor) -> torch.Tensor:
        # Residual mapping
        z_trans = z_src + self.net(z_src)
        return F.normalize(z_trans, dim=-1)


class CrossSourceEntityTranslator(nn.Module):
    """
    Manages all directional source transformations:
    - S1 -> S2, S1 -> S3
    - S2 -> S1, S3 -> S1
    - S2 -> S3, S3 -> S2
    """
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        
        # Translation modules
        self.translators = nn.ModuleDict({
            "S1_to_S2": CrossSourceTranslationHead(d_model),
            "S1_to_S3": CrossSourceTranslationHead(d_model),
            "S2_to_S1": CrossSourceTranslationHead(d_model),
            "S3_to_S1": CrossSourceTranslationHead(d_model),
            "S2_to_S3": CrossSourceTranslationHead(d_model),
            "S3_to_S2": CrossSourceTranslationHead(d_model),
        })

    def translate(self, z: torch.Tensor, src_type: str, tgt_type: str) -> torch.Tensor:
        key = f"{src_type}_to_{tgt_type}"
        if key in self.translators:
            return self.translators[key](z)
        return z

    def compute_translation_loss(
        self,
        z_src: torch.Tensor,       # [B, D]
        z_tgt: torch.Tensor,       # [B, D]
        src_types: list[str],
        tgt_types: list[str],
        temperature: float = 0.05,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes reconstruction cosine loss + translation InfoNCE contrastive loss.
        """
        B = z_src.shape[0]
        z_hat_tgt = torch.zeros_like(z_tgt)
        
        for i in range(B):
            s_t = src_types[i]
            t_t = tgt_types[i]
            key = f"{s_t}_to_{t_t}"
            if key in self.translators:
                z_hat_tgt[i] = self.translators[key](z_src[i : i + 1])[0]
            else:
                z_hat_tgt[i] = z_src[i]

        # 1. Cosine reconstruction loss: 1 - cosine(z_hat_tgt, z_tgt)
        recon_cos = (z_hat_tgt * z_tgt).sum(dim=-1) # [B]
        loss_recon = (1.0 - recon_cos).mean()

        # 2. Translated InfoNCE loss: similarity(z_hat_tgt, z_tgt) vs in-batch negatives
        sim_matrix = torch.matmul(z_hat_tgt, z_tgt.T) / temperature # [B, B]
        labels = torch.arange(B, device=z_src.device)
        loss_infonce = F.cross_entropy(sim_matrix, labels)

        return loss_recon, loss_infonce
