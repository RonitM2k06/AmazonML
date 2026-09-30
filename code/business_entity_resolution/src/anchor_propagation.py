"""
EXP_N031 — Anchor Propagation Module.

Restricts transitive path search exclusively to high-confidence anchor matches.
Prevents unconstrained path explosion and error percolation through ambiguous intermediate nodes.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class AnchorPropagationEngine:
    """
    Identifies high-confidence anchor entities for multi-hop propagation.
    """

    def __init__(self, min_anchor_prob: float = 0.90):
        self.min_anchor_prob = min_anchor_prob

    def extract_anchor_set(
        self,
        accepted_candidates: Set[str],
        candidate_probabilities: Dict[str, float],
    ) -> Set[str]:
        """
        Filters accepted candidates down to verified high-certainty anchors.
        """
        anchors = set()
        for c in accepted_candidates:
            if candidate_probabilities.get(c, 0.0) >= self.min_anchor_prob:
                anchors.add(c)
        return anchors if anchors else accepted_candidates
