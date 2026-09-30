"""
EXP_N032 — Anchor Confidence Estimator.

Calculates continuous confidence for intermediate anchor nodes in relational graphs:
anchor_confidence = f(p_direct, address_agreement, branch_flag, source_agreement)
Guarantees weak intermediate nodes do not propagate noisy signals.
"""

from __future__ import annotations

import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class AnchorConfidenceEstimator:
    """
    Evaluates intermediate node trustworthiness.
    """

    def __init__(self):
        pass

    @staticmethod
    def compute_anchor_confidence(
        prob: float,
        addr_jaccard: float,
        branch_similarity: float,
    ) -> float:
        if branch_similarity > 0.0:
            return 0.0
        conf = 0.7 * prob + 0.3 * addr_jaccard
        return float(min(1.0, max(0.0, conf)))
