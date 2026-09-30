"""
EXP_N029 — Evidence Path Builder.

Constructs independent evidence paths for each (S1, Candidate) candidate:
- PATH A: Direct model probability and feature alignment
- PATH B: Opposite-source sibling consensus (S1 -> Candidate -> Opposite Sibling)
- PATH C: Sibling bridge through another accepted candidate (S1 -> Accepted -> Candidate)
- PATH D: Shared address / house number / postal agreement evidence
- PATH E: Shared name / phonetic / transliteration / subword agreement evidence
- PATH F: Multi-hop graph neighborhood density
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class EvidencePathBuilder:
    """
    Constructs multi-channel independent evidence representations for entity candidate pairs.
    """

    def __init__(self):
        pass

    @staticmethod
    def compute_direct_path(
        prob: float,
        name_exact: float,
        name_jaccard: float,
        addr_jaccard: float,
        neural_cosine: float,
    ) -> np.ndarray:
        """Vector representing direct S1 <-> Candidate alignment."""
        return np.array([prob, name_exact, name_jaccard, addr_jaccard, neural_cosine], dtype=np.float32)

    @staticmethod
    def compute_address_evidence_path(
        addr_jaccard: float,
        house_num_match: float,
        postal_match: float,
        addr_missing: float,
    ) -> float:
        """Composite address path score."""
        if addr_missing == 1.0:
            return 0.0
        score = 0.4 * addr_jaccard + 0.35 * house_num_match + 0.25 * postal_match
        return float(score)

    @staticmethod
    def compute_name_phonetic_path(
        name_jaccard: float,
        translit_sim: float,
        skeleton_sim: float,
        lev_sim: float,
    ) -> float:
        """Composite name/phonetic/script path score."""
        score = max(name_jaccard, translit_sim, skeleton_sim, lev_sim)
        return float(score)
