"""
EXP_N020: Phase 4 & 5 — Contrastive Hard-Negative Entity Training & Iterative Self-Refinement.

Loss:
L_total = L_BCE(pos_weight=5.0) + lambda_cont * L_InfoNCE + lambda_rank * L_Rank

Iterative Self-Refinement:
Iteration 0: Initial entity memory from high-confidence predictions (p >= 0.954)
Iteration 1: Re-encode entity memory with newly identified matches
Iteration 2: Final calibrated inference over candidate set
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def compute_contrastive_loss(
    contrastive_embs: torch.Tensor,    # [B, K, D]
    labels: torch.Tensor,              # [B, K] (0 or 1)
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised contrastive loss over candidate pool:
    Pulls positive candidate observations towards each other while repelling confusing negatives.
    """
    B, K, D = contrastive_embs.shape
    total_loss = torch.tensor(0.0, device=contrastive_embs.device)
    valid_batches = 0

    for b in range(B):
        y = labels[b]
        pos_idx = torch.where(y == 1)[0]
        neg_idx = torch.where(y == 0)[0]

        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue

        embs = contrastive_embs[b] # [K, D]
        sim_mat = torch.matmul(embs, embs.T) / temperature # [K, K]

        # Numerator: similarities between positives
        # Denominator: similarities against all candidates
        pos_sim = sim_mat[pos_idx][:, pos_idx]
        denom = torch.logsumexp(sim_mat[pos_idx], dim=-1)

        # Mask out self-similarity from positive sum
        mask = ~torch.eye(len(pos_idx), dtype=torch.bool, device=contrastive_embs.device)
        if mask.any():
            pos_term = (pos_sim * mask.float()).sum(dim=-1) / mask.float().sum(dim=-1).clamp(min=1.0)
            loss_b = (denom - pos_term).mean()
            total_loss = total_loss + loss_b
            valid_batches += 1

    if valid_batches > 0:
        return total_loss / valid_batches
    return torch.tensor(0.0, device=contrastive_embs.device)


def compute_ranking_loss(
    logits: torch.Tensor,              # [B, K]
    labels: torch.Tensor,              # [B, K]
    margin: float = 0.5,
) -> torch.Tensor:
    """
    Margin ranking loss inside candidate graph:
    Ensures positive matches score higher than hard negative candidates by at least margin.
    """
    pos_mask = (labels == 1)
    neg_mask = (labels == 0)

    if not pos_mask.any() or not neg_mask.any():
        return torch.tensor(0.0, device=logits.device)

    pos_scores = logits[pos_mask]
    neg_scores = logits[neg_mask]

    n_pairs = min(512, len(pos_scores), len(neg_scores))
    p_samp = pos_scores[torch.randint(0, len(pos_scores), (n_pairs,))]
    n_samp = neg_scores[torch.randint(0, len(neg_scores), (n_pairs,))]

    return F.relu(margin - (p_samp - n_samp)).mean()
