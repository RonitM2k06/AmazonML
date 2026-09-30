"""
EXP_N026: Set Completion Network & Cluster Coherence Scoring.

Evaluates candidates as set completion candidates:
- compatibility(candidate, entity_memory)
- ΔC = Coherence(entity ∪ candidate) - Coherence(entity)
- Competing entity margin: score(target) - max_score(competing entities)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SetCompletionNetwork(nn.Module):
    """
    Evaluates whether candidate c completes the latent entity set.
    Computes coherence gain, compatibility logit, and margin penalty.
    """
    def __init__(self, d_model: int = 128, d_cand: int = 57):
        super().__init__()
        self.cand_proj = nn.Sequential(
            nn.Linear(d_cand, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.coherence_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        z_entity: torch.Tensor,       # [B, d_model]
        cand_feats: torch.Tensor,     # [B, d_cand]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            compatibility: [B] cosine affinity
            coherence_score: [B] learned cluster coherence
        """
        z_cand = F.normalize(self.cand_proj(cand_feats), dim=-1)
        compatibility = (z_entity * z_cand).sum(dim=-1)
        pair = torch.cat([z_entity, z_cand], dim=-1)
        coherence = self.coherence_head(pair).squeeze(-1)
        return compatibility, coherence


def compute_cluster_coherence_gain(
    accepted_matrix: np.ndarray,      # [M, 57]
    candidate_feat: np.ndarray,       # [57]
    base_prob: float,
) -> Tuple[float, float, float]:
    """
    Computes non-neural statistical and lexical coherence gain:
    1. variance reduction in neural cosine
    2. house number alignment (+1 if matches any accepted, -1 if contradicts)
    3. token Jaccard consistency to accepted cluster centroid
    """
    if len(accepted_matrix) == 0:
        return 0.0, 0.0, 0.0

    acc_cosines = accepted_matrix[:, 56]
    cand_cosine = candidate_feat[56]
    acc_hn = accepted_matrix[:, 40]
    cand_hn = candidate_feat[40]
    acc_aj = accepted_matrix[:, 30]
    cand_aj = candidate_feat[30]

    # 1. Delta cosine variance
    initial_var = float(np.var(acc_cosines)) if len(acc_cosines) > 1 else 0.0
    combined_cosines = np.append(acc_cosines, cand_cosine)
    new_var = float(np.var(combined_cosines))
    delta_variance = initial_var - new_var # positive means cluster became more compact

    # 2. House number consistency
    if np.any(acc_hn == 1.0) and cand_hn == 1.0:
        hn_signal = 1.0
    elif np.any(acc_hn == 1.0) and cand_hn == 0.0:
        hn_signal = -1.0 # hard contradiction with confirmed house number
    else:
        hn_signal = 0.0

    # 3. Mean address Jaccard to accepted
    mean_aj_gain = float(cand_aj - np.mean(acc_aj))

    return delta_variance, hn_signal, mean_aj_gain
