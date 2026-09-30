"""
EXP_N027 — Sibling Retrieval and Path Scoring Engine.

Computes path scoring and source-consistency metrics for candidates reached through
accepted sibling anchors:
- direct_similarity(S1, C)
- sibling_similarity(R, C)
- path_strength = similarity(S1, R) * similarity(R, C)
- source compatibility and mutual exclusion checks
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class SiblingRetrievalScorer:
    """
    Evaluates multi-hop candidate paths and scores candidate viability.
    """

    def __init__(self):
        pass

    @staticmethod
    def compute_path_score(
        direct_score: float,
        sibling_score: float,
        anchor_confidence: float = 1.0,
    ) -> float:
        """
        Combines direct signal with transitive sibling anchor signal.
        """
        path_strength = anchor_confidence * sibling_score
        # Soft combination bounded in [0, 1]
        combined = 0.6 * direct_score + 0.4 * path_strength
        return float(combined)

    @staticmethod
    def filter_precision_safe(
        candidate_id: str,
        path_score: float,
        min_path_threshold: float = 0.85,
    ) -> bool:
        """Precision gate for sibling-admitted candidates."""
        return path_score >= min_path_threshold
