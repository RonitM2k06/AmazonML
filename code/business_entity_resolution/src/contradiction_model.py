"""
EXP_N029 — Learned Contradiction and Anti-Collision Model.

Evaluates independent evidence contradicting candidate membership:
- Branch / Same-chain conflict
- House number contradiction (both entities have house numbers, but they differ)
- Postal code contradiction (differing geographic postal prefixes)
- Competing same-name / same-address candidate suppression
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ContradictionModel:
    """
    Computes a penalty score representing irreconcilable conflict between S1 and Candidate.
    """

    def __init__(self):
        pass

    @staticmethod
    def compute_contradiction_score(
        branch_similarity: float,
        house_num_match: float,
        postal_match: float,
        addr_missing: float,
        addr_jaccard: float,
    ) -> float:
        """
        Returns contradiction score in [0.0, 1.0].
        High score indicates irreconcilable geographic/branch contradiction.
        """
        penalty = 0.0
        # Severe branch collision flag
        if branch_similarity > 0.0 and addr_jaccard < 0.20:
            penalty += 0.50
        # House number mismatch (both present, but match is 0)
        if addr_missing == 0.0 and house_num_match == 0.0 and addr_jaccard < 0.10:
            penalty += 0.30
        # Geographic postal contradiction
        if addr_missing == 0.0 and postal_match == 0.0 and addr_jaccard < 0.05:
            penalty += 0.20
        return min(1.0, float(penalty))
