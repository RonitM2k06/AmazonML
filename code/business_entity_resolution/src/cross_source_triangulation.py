"""
EXP_N030 — Cross-Source Triangulation Module.

Triangulates evidence across S1 <-> S2, S1 <-> S3, S2 <-> S3 candidate paths:
- Direct support: S1 <-> C
- Opposite sibling support: S1 <-> Sibling(C)
- Triangulation strength: similarity(S1, Sibling) * similarity(Sibling, C)
- Source diversity balance: S2 + S3 joint representation
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class CrossSourceTriangulation:
    """
    Computes reciprocal cross-source triangulation scores.
    """

    def __init__(self, training_sibling_edges: Optional[Dict[str, Set[str]]] = None):
        self.edges = training_sibling_edges or defaultdict(set)

    def compute_triangulation_feature(
        self,
        candidate_id: str,
        accepted_siblings: Set[str],
        candidate_prob: float,
    ) -> float:
        """
        Returns triangulation strength:
        High when candidate is structurally linked to an already verified cluster sibling.
        """
        if not accepted_siblings:
            return 0.0
        connected = self.edges.get(candidate_id, set())
        shared = connected & accepted_siblings
        if len(shared) > 0:
            return float(1.0 + candidate_prob)
        return float(candidate_prob)
