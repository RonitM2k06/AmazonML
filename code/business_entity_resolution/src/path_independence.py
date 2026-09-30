"""
EXP_N031 — Path Independence and Evidence Family Counter.

Evaluates independent evidence families supporting candidate admission:
- NAME: Token Jaccard, normalized exact match
- ADDRESS: Address Jaccard, house number match, postal match
- NEURAL: MiniLM / multilingual cosine
- SIBLING: Direct length-2 cross-source graph link
- MULTI-HOP: Length-3 transitive graph link
Requires at least 2 independent families before borderline admission.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class PathIndependenceAnalyzer:
    """
    Counts independent evidence families supporting an entity hypothesis.
    """

    def __init__(self):
        pass

    @staticmethod
    def count_independent_families(
        name_jaccard: float,
        addr_jaccard: float,
        house_num_match: float,
        postal_match: float,
        neural_cosine: float,
        is_length2: bool,
        is_length3: bool,
    ) -> int:
        """
        Returns count of distinct, non-collinear evidence families.
        """
        families = 0
        if name_jaccard >= 0.50:
            families += 1
        if addr_jaccard >= 0.30 or house_num_match == 1.0 or postal_match == 1.0:
            families += 1
        if neural_cosine >= 0.70:
            families += 1
        if is_length2 or is_length3:
            families += 1
        return families
