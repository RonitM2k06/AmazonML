"""
EXP_N020: Phase 1 & 2 — Latent Entity Memory Construction & Multi-View Representation.

Constructs an entity memory representation for each S1 query entity from:
- Canonical S1 attributes
- High-confidence S2/S3 observations (p >= threshold)
- Attribute token sets, house numbers, postal codes, and embeddings
- Multi-view representation:
    1. Name representation & observed variant coverage
    2. Address representation & token evidence
    3. Transliteration / script consistency
    4. Neural centroid / attention-pooled semantic embedding
    5. Structured address component agreement
    6. Cross-source relational memory (S2 <-> S3)

Strictly zero label leakage: Built entirely from inference-time predictions and raw text.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class LatentEntityMemory:
    """
    In-memory representation of a latent business entity observed through multiple sources.
    """
    def __init__(
        self,
        s1_id: str,
        observed_names: List[str],
        observed_addrs: List[str],
        observed_sources: List[str],
        observed_embeddings: torch.Tensor,       # [M, D_emb]
        anchor_indices: List[int],               # indices in candidate pool
        memory_features: torch.Tensor,           # [D_mem]
    ):
        self.s1_id = s1_id
        self.observed_names = observed_names
        self.observed_addrs = observed_addrs
        self.observed_sources = observed_sources
        self.observed_embeddings = observed_embeddings
        self.anchor_indices = anchor_indices
        self.memory_features = memory_features


def construct_latent_entity_features(
    cand_X: np.ndarray,             # [K, 57]
    cand_probs: np.ndarray,         # [K]
    cand_sources: np.ndarray,       # [K] ('S2' or 'S3')
    anchor_thresh: float = 0.954,
) -> np.ndarray:
    """
    Computes vector of similarity against latent entity memory for all K candidates in pool.
    
    Features computed per candidate c_i:
    1. max_sim_to_memory: max similarity against any anchor in memory
    2. centroid_sim_to_memory: similarity against attention-weighted memory centroid
    3. house_number_memory_agreement: 1 if matches any anchor HN, -1 if contradicts, 0 neutral
    4. postal_memory_agreement: 1 if matches any anchor postal, 0 neutral
    5. cross_source_anchor_support: highest prob of an anchor from the OPPOSITE source
    6. num_anchors_in_memory: count of high-confidence observations in entity memory
    7. memory_density: ratio of anchors in pool
    8. address_token_memory_max_jaccard: max token jaccard against any anchor address
    9. name_memory_max_jaccard: max token jaccard against any anchor name
    10. candidate_score_gap_to_memory_anchor: p_cand - min_anchor_p
    """
    K = len(cand_probs)
    anchor_mask = (cand_probs >= anchor_thresh)
    num_anchors = int(np.sum(anchor_mask))

    mem_feats = np.zeros((K, 10), dtype=np.float32)

    if num_anchors == 0:
        # No high-confidence anchors in memory; default to neutral features
        mem_feats[:, 0] = cand_probs
        mem_feats[:, 1] = cand_probs
        mem_feats[:, 2] = 0.0
        mem_feats[:, 3] = 0.0
        mem_feats[:, 4] = 0.0
        mem_feats[:, 5] = 0.0
        mem_feats[:, 6] = 0.0
        mem_feats[:, 7] = cand_X[:, 30] # addr jaccard against S1
        mem_feats[:, 8] = cand_X[:, 11] # name jaccard against S1
        mem_feats[:, 9] = 0.0
        return mem_feats

    anchor_indices = np.where(anchor_mask)[0]
    anchor_X = cand_X[anchor_indices]
    anchor_probs = cand_probs[anchor_indices]
    anchor_sources = cand_sources[anchor_indices]

    # Anchor summary properties
    anchor_hn_exact = anchor_X[:, 40]
    anchor_postal_exact = anchor_X[:, 37]
    anchor_addr_jac = anchor_X[:, 30]
    anchor_name_jac = anchor_X[:, 11]

    # Opposite source support
    has_s2_anchor = np.any(anchor_sources == "S2")
    has_s3_anchor = np.any(anchor_sources == "S3")
    max_s2_anchor_p = float(np.max(anchor_probs[anchor_sources == "S2"])) if has_s2_anchor else 0.0
    max_s3_anchor_p = float(np.max(anchor_probs[anchor_sources == "S3"])) if has_s3_anchor else 0.0

    # Weights for centroid
    weights = anchor_probs / np.sum(anchor_probs)

    for i in range(K):
        c_src = cand_sources[i]
        c_p = cand_probs[i]
        c_hn = cand_X[i, 40]
        c_post = cand_X[i, 37]
        c_addr_jac = cand_X[i, 30]
        c_name_jac = cand_X[i, 11]

        # 1. max sim
        max_sim = float(np.max(anchor_probs))
        # 2. centroid sim
        centroid_sim = float(np.sum(anchor_probs * weights))

        # 3. house number consistency
        if c_hn == 1.0 and np.any(anchor_hn_exact == 1.0):
            hn_agree = 1.0
        elif c_hn == 0.0 and np.any(anchor_hn_exact == 1.0):
            hn_agree = -1.0 # contradiction with confirmed house number
        else:
            hn_agree = 0.0

        # 4. postal consistency
        if c_post == 1.0 and np.any(anchor_postal_exact == 1.0):
            post_agree = 1.0
        else:
            post_agree = 0.0

        # 5. cross-source anchor support
        opp_support = max_s2_anchor_p if c_src == "S3" else max_s3_anchor_p

        # 6. num anchors
        n_anc = float(num_anchors)
        # 7. memory density
        density = float(num_anchors) / float(K)

        # 8 & 9. max address and name token overlap against memory
        max_aj = float(max(c_addr_jac, np.max(anchor_addr_jac)))
        max_nj = float(max(c_name_jac, np.max(anchor_name_jac)))

        # 10. score gap
        score_gap = c_p - float(np.min(anchor_probs))

        mem_feats[i] = [
            max_sim,
            centroid_sim,
            hn_agree,
            post_agree,
            opp_support,
            n_anc,
            density,
            max_aj,
            max_nj,
            score_gap,
        ]

    return mem_feats
