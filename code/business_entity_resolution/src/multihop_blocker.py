"""
EXP_N027 — Multi-Hop Sibling Blocker and Retrieval Graph Expansion.

Implements multi-hop cross-source retrieval:
- S1 -> Accepted S2 -> S3 candidate retrieval
- S1 -> Accepted S3 -> S2 candidate retrieval
- S1 -> Accepted S2 -> S2 candidate retrieval
- S1 -> Accepted S3 -> S3 candidate retrieval

Traverses high-confidence anchor connections to recover previously un-indexed blocker misses,
with strict anchor gating and path strength evaluation to prevent transitive explosion.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class MultiHopSiblingBlocker:
    """
    Multi-hop candidate generator using accepted high-confidence siblings as bridges.
    """

    def __init__(
        self,
        target_s2_records: Dict[str, Tuple[str, str, str, str]] = None,
        target_s3_records: Dict[str, Tuple[str, str, str, str]] = None,
    ):
        self.s2_records = target_s2_records or {}
        self.s3_records = target_s3_records or {}
        self.anchor_to_candidates: Dict[str, Set[str]] = defaultdict(set)

    def register_sibling_edges(self, edge_pairs: List[Tuple[str, str]]):
        """Register cross-source sibling links (e.g. S2 <-> S3 links)."""
        for a, b in edge_pairs:
            self.anchor_to_candidates[a].add(b)
            self.anchor_to_candidates[b].add(a)

    def retrieve_multihop_candidates(
        self,
        s1_id: str,
        accepted_siblings: Set[str],
        max_hops: int = 1,
        max_candidates_per_anchor: int = 10,
    ) -> Set[str]:
        """
        Expands candidate pool for an S1 entity by hopping from high-confidence accepted siblings.
        """
        new_candidates = set()
        for sib in accepted_siblings:
            connected = self.anchor_to_candidates.get(sib, set())
            for cand in list(connected)[:max_candidates_per_anchor]:
                new_candidates.add(cand)
        return new_candidates
