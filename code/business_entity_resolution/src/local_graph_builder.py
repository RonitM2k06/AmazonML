"""
HierGAT+ Local Candidate Graph Builder & Feature Extractor.
Constructs local heterogeneous graphs for each S1 query entity and its candidate pool.
Extracts:
- Entity & Attribute Node Features (Name, Address, Country, Source)
- Candidate Comparison Features (57 Lexical/Neural + Probabilities + Rank)
- S2 <-> S3 Cross-Source Relational Edges (Name Sim, Address Sim, Postal Agreement, House Num Agreement)
"""

from __future__ import annotations

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

class LocalEntityGraph:
    """
    Data container for a single S1 entity and its candidate pool.
    """
    def __init__(
        self,
        s1_id: str,
        cand_ids: List[str],
        sources: List[str],
        comparison_feats: torch.Tensor,    # [K, D_comp]
        relational_edges: torch.Tensor,    # [2, num_edges] (src, dst)
        relational_feats: torch.Tensor,    # [num_edges, D_rel]
        labels: Optional[torch.Tensor] = None, # [K]
    ):
        self.s1_id = s1_id
        self.cand_ids = cand_ids
        self.sources = sources
        self.comparison_feats = comparison_feats
        self.relational_edges = relational_edges
        self.relational_feats = relational_feats
        self.labels = labels
        self.num_candidates = len(cand_ids)


def build_s2_s3_relational_edges(
    cand_sources: np.ndarray,      # [K] strings 'S2' or 'S3'
    cand_X: np.ndarray,            # [K, 57]
    cand_probs: np.ndarray,        # [K]
    edge_prob_thresh: float = 0.50
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs cross-source S2 <-> S3 edges between candidates in the same pool.
    Features include:
    - S2 prob, S3 prob, min prob, max prob
    - house number agreement (+1, -1, 0)
    - postal agreement (+1, -1, 0)
    - address token jaccard approximation / diff
    - name similarity diff
    """
    K = len(cand_sources)
    s2_indices = np.where((cand_sources == "S2") & (cand_probs >= edge_prob_thresh))[0]
    s3_indices = np.where((cand_sources == "S3") & (cand_probs >= edge_prob_thresh))[0]

    edge_list_src = []
    edge_list_dst = []
    edge_features = []

    for i in s2_indices:
        for j in s3_indices:
            p_i = cand_probs[i]
            p_j = cand_probs[j]
            
            # Cross features
            hn_i = cand_X[i, 40]
            hn_j = cand_X[j, 40]
            hn_agree = 1.0 if (hn_i == 1.0 and hn_j == 1.0) else (0.0 if (hn_i == 0.0 or hn_j == 0.0) else -1.0)

            post_i = cand_X[i, 37]
            post_j = cand_X[j, 37]
            post_agree = 1.0 if (post_i == 1.0 and post_j == 1.0) else 0.0

            addr_diff = abs(cand_X[i, 30] - cand_X[j, 30])
            name_diff = abs(cand_X[i, 11] - cand_X[j, 11])
            cos_diff = abs(cand_X[i, 56] - cand_X[j, 56]) if cand_X.shape[1] > 56 else 0.0

            feat = [
                p_i, p_j, min(p_i, p_j), max(p_i, p_j),
                hn_agree, post_agree, addr_diff, name_diff, cos_diff,
                cand_X[i, 30], cand_X[j, 30], # addr jaccards
                cand_X[i, 11], cand_X[j, 11]  # name jaccards
            ]

            # Undirected message passing: S2 -> S3 and S3 -> S2
            edge_list_src.extend([i, j])
            edge_list_dst.extend([j, i])
            edge_features.extend([feat, feat])

    if not edge_list_src:
        # Dummy self-loop if no edges meet threshold
        edges = torch.zeros((2, 0), dtype=torch.long)
        feats = torch.zeros((0, 13), dtype=torch.float32)
    else:
        edges = torch.tensor([edge_list_src, edge_list_dst], dtype=torch.long)
        feats = torch.tensor(edge_features, dtype=torch.float32)

    return edges, feats
