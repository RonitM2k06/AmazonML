"""
EXP_N029 — Cross-Source Consensus Engine.

Evaluates opposite-source consensus for candidate pairs:
- S1 -> S2-A -> S3-B (validates S2-A through co-occurrence with S3-B)
- S1 -> S3-A -> S2-B (validates S3-A through co-occurrence with S2-B)
Learned without ground truth leakage using strictly training graph structure.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class CrossSourceConsensusEngine:
    """
    Computes cross-source consensus and reciprocal sibling support.
    """

    def __init__(self, training_sibling_edges: Optional[Dict[str, Set[str]]] = None):
        self.sibling_graph = training_sibling_edges or defaultdict(set)

    def evaluate_candidate_consensus(
        self,
        candidate_id: str,
        accepted_siblings_for_s1: Set[str],
    ) -> float:
        """
        Returns consensus score: 1.0 if candidate is independently supported by
        an accepted opposite-source sibling in the entity cluster, else 0.0.
        """
        if not accepted_siblings_for_s1:
            return 0.0
        
        # Check if candidate shares an edge with any accepted sibling
        connected_nodes = self.sibling_graph.get(candidate_id, set())
        overlap = connected_nodes & accepted_siblings_for_s1
        if len(overlap) > 0:
            return 1.0
        return 0.0
