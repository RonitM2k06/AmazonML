#!/usr/bin/env python3
"""
Test-Time Reciprocal Entity Induction Engine for EXP_N037.

Implements reciprocal cross-source entity induction for unanchored entities:
S1 -> S2
S1 -> S3
S2 <-> S3

Candidates are promoted only when cross-source records mutually support
the same latent entity under strict multi-modal alignment guards.

Strict Zero-Leakage Invariants:
- Never accesses validation or test ground truth labels.
- Operates purely on observable pairwise candidate features:
  * Neural embedding cosine similarity
  * Address token Jaccard similarity
  * Postal code matching
  * Branch conflict signals
  * Training-derived co-occurrence edges between candidate records
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np


class TestTimeReciprocalEntityInducer:
    """
    Decodes entity sets for unanchored entities via reciprocal cross-source induction.
    """

    def __init__(
        self,
        min_cos: float = 0.62,
        min_aj_direct: float = 0.30,
        min_aj_postal: float = 0.20,
        min_postal_ratio: float = 0.66,
        high_prob_floor: float = 0.940,
        max_rank: int = 10,
    ):
        self.min_cos = min_cos
        self.min_aj_direct = min_aj_direct
        self.min_aj_postal = min_aj_postal
        self.min_postal_ratio = min_postal_ratio
        self.high_prob_floor = high_prob_floor
        self.max_rank = max_rank

    def compute_reciprocal_features(
        self,
        r2: int,
        r3: int,
        probs: np.ndarray,
        cosines: np.ndarray,
        addr_jaccs: np.ndarray,
        postal_matches: np.ndarray,
        branch_sims: np.ndarray,
    ) -> dict:
        """
        Computes explicit reciprocal cross-source features for candidate pair (S2, S3).
        """
        p2 = float(probs[r2])
        p3 = float(probs[r3])
        cos2 = float(cosines[r2])
        cos3 = float(cosines[r3])
        aj2 = float(addr_jaccs[r2])
        aj3 = float(addr_jaccs[r3])
        post2 = float(postal_matches[r2])
        post3 = float(postal_matches[r3])
        bs2 = float(branch_sims[r2])
        bs3 = float(branch_sims[r3])

        return {
            "s1_s2_score": p2,
            "s1_s3_score": p3,
            "reciprocal_rank_s2": r2,
            "reciprocal_rank_s3": r3,
            "score_product": p2 * p3,
            "score_min": min(p2, p3),
            "score_margin": abs(p2 - p3),
            "rank_product": r2 * r3,
            "rank_sum": r2 + r3,
            "cross_source_cosine_min": min(cos2, cos3),
            "cross_source_address_jaccard_min": min(aj2, aj3),
            "postal_match_min": min(post2, post3),
            "branch_contradiction": (bs2 > 0) or (bs3 > 0),
        }

    def decode_unanchored_candidates(
        self,
        cands: np.ndarray,           # [K] candidate IDs
        sources: np.ndarray,         # [K] candidate sources ('S2' or 'S3')
        probs: np.ndarray,           # [K] pairwise probabilities
        cosines: np.ndarray,         # [K] neural cosines
        addr_jaccs: np.ndarray,      # [K] address Jaccards
        postal_matches: np.ndarray,  # [K] postal match ratios
        branch_sims: np.ndarray,     # [K] branch similarity flags
        training_edges: Dict[str, Set[str]], # Training co-occurrence edges
    ) -> List[int]:
        """
        Returns admitted candidate indices for an unanchored S1 entity.
        """
        admitted = set()
        K = min(self.max_rank, len(cands))

        # 1. Clean High-Confidence Bootstrap
        for r in range(min(3, K)):
            if probs[r] >= self.high_prob_floor and branch_sims[r] == 0:
                admitted.add(r)

        # 2. Reciprocal Cross-Source Induction
        s2_indices = [r for r in range(K) if sources[r] == "S2"]
        s3_indices = [r for r in range(K) if sources[r] == "S3"]

        for r2 in s2_indices:
            c2 = cands[r2]
            for r3 in s3_indices:
                c3 = cands[r3]
                if c3 in training_edges.get(c2, set()):
                    # Mutually supported latent entity in cross-source graph
                    if branch_sims[r2] == 0 and branch_sims[r3] == 0:
                        cos2, cos3 = cosines[r2], cosines[r3]
                        if cos2 >= self.min_cos and cos3 >= self.min_cos:
                            g2 = (addr_jaccs[r2] >= self.min_aj_direct) or (
                                postal_matches[r2] >= self.min_postal_ratio
                                and addr_jaccs[r2] >= self.min_aj_postal
                            )
                            g3 = (addr_jaccs[r3] >= self.min_aj_direct) or (
                                postal_matches[r3] >= self.min_postal_ratio
                                and addr_jaccs[r3] >= self.min_aj_postal
                            )
                            if g2 and g3:
                                admitted.add(r2)
                                admitted.add(r3)

        return sorted(admitted)


if __name__ == "__main__":
    print("Self-testing TestTimeReciprocalEntityInducer...")
    inducer = TestTimeReciprocalEntityInducer(min_cos=0.62)

    mock_cands = np.array(["S2-101", "S3-202", "S2-303", "S3-404"])
    mock_srcs = np.array(["S2", "S3", "S2", "S3"])
    mock_probs = np.array([0.945, 0.400, 0.200, 0.350])
    mock_cos = np.array([0.80, 0.75, 0.65, 0.64])
    mock_aj = np.array([0.50, 0.40, 0.35, 0.32])
    mock_post = np.array([1.0, 1.0, 1.0, 1.0])
    mock_bs = np.array([0.0, 0.0, 0.0, 0.0])
    mock_edges = {"S2-303": {"S3-404"}, "S3-404": {"S2-303"}}

    admitted = inducer.decode_unanchored_candidates(
        mock_cands, mock_srcs, mock_probs, mock_cos, mock_aj, mock_post, mock_bs, mock_edges
    )
    print(f"Admitted candidate indices: {admitted}")
    assert 0 in admitted, "Candidate 0 should be admitted by prob floor!"
    assert 2 in admitted and 3 in admitted, "Candidates 2 and 3 should be admitted by reciprocal induction!"
    print("[SUCCESS] TestTimeReciprocalEntityInducer passed all test-time safety unit tests.")
