"""
EXP_N032 — Edge Encoder Module.

Encodes heterogeneous edge feature representations across:
- Direct S1-Candidate edges (pairwise prob, rank, name, address, neural cosine)
- S2-S3 sibling edges (training graph cross-source co-occurrence)
- S2-S2 / S3-S3 intra-source edges
- Relational metadata (path distance, anchor quality)
"""

from __future__ import annotations

import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class EdgeEncoder:
    """
    Constructs compact edge vectors for relational graph propagation.
    """

    def __init__(self):
        pass

    @staticmethod
    def encode_edge(
        prob: float,
        name_jaccard: float,
        addr_jaccard: float,
        neural_cosine: float,
        branch_similarity: float,
        edge_type: int, # 0: S1-Cand, 1: S2-S3, 2: S2-S2/S3-S3
    ) -> np.ndarray:
        return np.array([
            prob,
            name_jaccard,
            addr_jaccard,
            neural_cosine,
            branch_similarity,
            1.0 if edge_type == 0 else 0.0,
            1.0 if edge_type == 1 else 0.0,
            1.0 if edge_type == 2 else 0.0,
        ], dtype=np.float32)
