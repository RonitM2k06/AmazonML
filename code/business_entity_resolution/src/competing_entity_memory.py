"""
EXP_N026: Competing Entity Memory & Separation Margin.

Calculates competitive margins between target entity memory and competing cluster memories
to prevent branch / same-chain collisions.
"""

from __future__ import annotations

import numpy as np


def compute_competing_cluster_margin(
    cand_prob: float,
    cand_cosine: float,
    cand_branch_flag: float,
    target_cluster_mean_p: float,
    target_cluster_mean_cos: float,
    pool_probs: np.ndarray,
    pool_cosines: np.ndarray,
    cand_idx: int,
) -> Tuple[float, float]:
    """
    Computes separation margin between target cluster and the strongest competing alternative in the pool.
    """
    # Exclude candidate itself
    other_indices = [i for i in range(len(pool_probs)) if i != cand_idx]
    if len(other_indices) == 0:
        return 1.0, 0.0

    other_p = pool_probs[other_indices]
    other_cos = pool_cosines[other_indices]

    # Strongest competing candidate outside accepted set
    max_comp_p = float(np.max(other_p)) if len(other_p) > 0 else 0.0
    comp_margin = cand_prob - max_comp_p

    # Branch collision penalty
    branch_penalty = 1.0 if cand_branch_flag == 1.0 and comp_margin < 0.15 else 0.0

    return comp_margin, branch_penalty
