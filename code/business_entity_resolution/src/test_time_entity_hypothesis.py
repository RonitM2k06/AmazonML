#!/usr/bin/env python3
"""
Test-Time Entity Hypothesis & Set Resolution Engine for EXP_N036.

Constructs small cross-source connected components from top observed candidates
for unanchored entities using strictly observable pairwise features, frozen models,
and learned threshold invariants.

Strict Zero-Leakage Invariants:
- Never imports ground truth labels.
- Operates purely on observable pairwise candidate features:
  * Neural embedding cosine similarity
  * Address token Jaccard similarity
  * House number agreement
  * Branch conflict signals
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


class TestTimeEntityHypothesisDecoder:
    """
    Decodes entity sets for unanchored entities via observable multi-modal components.
    """

    def __init__(
        self,
        min_cos: float = 0.70,
        min_aj: float = 0.35,
        high_prob_floor: float = 0.940,
    ):
        self.min_cos = min_cos
        self.min_aj = min_aj
        self.high_prob_floor = high_prob_floor

    def decode_unanchored_candidates(
        self,
        cands: np.ndarray,          # [K] candidate IDs
        probs: np.ndarray,          # [K] pairwise probabilities
        cosines: np.ndarray,        # [K] neural cosines
        addr_jaccs: np.ndarray,     # [K] address Jaccards
        branch_sims: np.ndarray,    # [K] branch similarity flags
        training_edges: dict,       # Training co-occurrence edges
    ) -> list[int]:
        """
        Returns admitted candidate indices for an unanchored S1 entity.
        """
        admitted = []

        # 1. High-Probability Clean Bootstrap
        for r in range(min(3, len(cands))):
            if probs[r] >= self.high_prob_floor and branch_sims[r] == 0:
                admitted.append(r)

        # 2. Cross-Source Mutual Anchor Pair
        for r1 in range(min(2, len(cands))):
            c1 = cands[r1]
            for r2 in range(r1 + 1, min(2, len(cands))):
                c2 = cands[r2]
                if c1[:2] != c2[:2] and c2 in training_edges.get(c1, set()):
                    for r_cur in [r1, r2]:
                        if (
                            cosines[r_cur] >= self.min_cos
                            and addr_jaccs[r_cur] >= self.min_aj
                            and r_cur not in admitted
                        ):
                            admitted.append(r_cur)

        return admitted


if __name__ == "__main__":
    print("Testing TestTimeEntityHypothesisDecoder...")
    decoder = TestTimeEntityHypothesisDecoder()
    dummy_cands = np.array(["S2-101", "S3-202", "S2-303"])
    dummy_probs = np.array([0.945, 0.850, 0.100])
    dummy_cos = np.array([0.80, 0.75, 0.50])
    dummy_aj = np.array([0.60, 0.40, 0.10])
    dummy_bs = np.array([0.0, 0.0, 0.0])
    dummy_edges = {"S2-101": {"S3-202"}, "S3-202": {"S2-101"}}

    admitted = decoder.decode_unanchored_candidates(
        dummy_cands, dummy_probs, dummy_cos, dummy_aj, dummy_bs, dummy_edges
    )
    print(f"Admitted indices: {admitted}")
    assert 0 in admitted, "Index 0 should be admitted by prob floor!"
    print("[SUCCESS] TestTimeEntityHypothesisDecoder verified.")
