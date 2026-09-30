"""
EXP_N030 — Pseudo-Label Trainer (Training-Only Self-Training).

Performs semi-supervised pseudo-labeling exclusively on training split entities:
- Mines extremely high-confidence (p >= 0.995) cross-source pairs
- Verifies absence of branch conflict and address contradiction
- Generates enriched pseudo-positive pairs to teach the reranker subtle entity mutations
- Never touches validation or test entities (zero leakage)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class PseudoLabelTrainer:
    """
    Constructs high-confidence pseudo-labels on training splits for iterative self-training.
    """

    def __init__(self, confidence_threshold: float = 0.995):
        self.confidence_threshold = confidence_threshold

    def mine_pseudo_labels(
        self,
        probabilities: np.ndarray,
        branch_flags: np.ndarray,
        address_contradictions: np.ndarray,
    ) -> np.ndarray:
        """
        Returns boolean mask of candidates meeting strict pseudo-label criteria.
        """
        valid_pseudo = (
            (probabilities >= self.confidence_threshold)
            & (branch_flags == 0)
            & (address_contradictions == 0)
        )
        return valid_pseudo
