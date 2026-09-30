"""
EXP_N029 — Evidence Fusion Model.

Fuses high-level evidence channels into a unified acceptance decision:
E_fused = f(E_direct, E_sibling, E_consensus, E_address, E_name, E_contradiction)
Guarantees precision preservation with fallback to N027 when evidence is uncertain.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class EvidenceFusionModel:
    """
    Precision-safe multi-path evidence combiner.
    """

    def __init__(self, high_direct_threshold: float = 0.9570, consensus_min_prob: float = 0.70):
        self.high_direct_threshold = high_direct_threshold
        self.consensus_min_prob = consensus_min_prob

    def fuse_and_decide(
        self,
        direct_prob: float,
        sibling_consensus: float,
        addr_evidence: float,
        contradiction_score: float,
        base_accepted: bool,
    ) -> bool:
        """
        Decision surface:
        - If base N027 accepted AND contradiction is low -> Keep accepted.
        - If direct prob is medium/low BUT supported by independent opposite-source consensus -> Rescue.
        - If contradiction is severe (>0.70) AND direct prob < 0.99 -> Prune.
        """
        # Severe contradiction suppression
        if contradiction_score >= 0.70 and direct_prob < 0.990:
            return False
            
        # Standard acceptance from N027 baseline
        if base_accepted:
            return True
            
        # Independent consensus rescue: admitted only if opposite sibling consensus confirms it
        if sibling_consensus >= 1.0 and direct_prob >= self.consensus_min_prob and contradiction_score < 0.20:
            return True
            
        return False
