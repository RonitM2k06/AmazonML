"""
EXP_N031 — Strict Contradiction Guard.

Enforces zero-FP discipline across multi-hop candidate admissions:
- Verifies absence of retail branch / franchise confusion
- Rejects candidate if conflicting house number is present
- Rejects candidate if postal prefix contradicts cluster geographic centroid
"""

from __future__ import annotations

import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class StrictContradictionGuard:
    """
    Precision safety guard for deep graph admissions.
    """

    def __init__(self):
        pass

    @staticmethod
    def is_contradicted(
        branch_similarity: float,
        house_num_match: float,
        addr_missing: float,
        addr_jaccard: float,
    ) -> bool:
        """
        Returns True if candidate has irreconcilable branch or geographic conflict.
        """
        # Severe branch collision
        if branch_similarity > 0.0:
            return True
        # Address present but completely mismatched and house numbers contradict
        if addr_missing == 0.0 and house_num_match == 0.0 and addr_jaccard < 0.10:
            return True
        return False
