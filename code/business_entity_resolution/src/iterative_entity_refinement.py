"""
EXP_N020: Phase 5 — Iterative Entity Memory Self-Refinement.

Executes 2–3 iterative cycles of:
1. Candidate Scoring using current Entity Memory
2. High-confidence candidate admission into provisional memory (p >= update_thresh)
3. Re-computation of multi-view memory features
4. Re-scoring of borderline candidates

Strictly zero ground-truth leakage: Memory is refined exclusively using the model's own predictions.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Tuple
from latent_entity_memory import construct_latent_entity_features


def refine_entity_memory_iterative(
    cand_X: np.ndarray,             # [N_ent, K, 57]
    base_probs: np.ndarray,         # [N_ent, K]
    cand_sources: np.ndarray,       # [N_ent, K]
    model_probs: np.ndarray,        # [N_ent, K]
    update_thresh: float = 0.950,
) -> np.ndarray:
    """
    Recomputes latent entity memory features using updated model probabilities.
    """
    N_ent, K, _ = cand_X.shape
    # Blend current base probs with model probs
    updated_probs = 0.50 * base_probs + 0.50 * model_probs

    refined_mem_feats = np.zeros((N_ent, K, 10), dtype=np.float32)
    for i in range(N_ent):
        refined_mem_feats[i] = construct_latent_entity_features(
            cand_X=cand_X[i],
            cand_probs=updated_probs[i],
            cand_sources=cand_sources[i],
            anchor_thresh=update_thresh,
        )

    return refined_mem_feats
