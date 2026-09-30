#!/usr/bin/env python3
"""
Authoritative Shared Prediction Function for EXP_N039.
Provides ONE single source of truth for entity resolution predictions across
both validation audit and test production inference.

Invariants:
- Zero heuristic fallbacks.
- Exact N039 prediction logic.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple
import numpy as np


def n039_predict_candidates_from_scores(
    cand_ids: List[str],
    cand_sources: List[str],
    probs: np.ndarray,
    cosines: np.ndarray,
    addr_jaccards: np.ndarray,
    postal_matches: np.ndarray,
    branch_sims: np.ndarray,
    house_nums: np.ndarray,
    training_edges: Dict[str, Set[str]],
    recovered_cands: Set[str] | None = None,
) -> List[str]:
    """
    Authoritative N039 prediction function for a single Source 1 entity given candidate scores.
    """
    K = len(cand_ids)
    if K == 0:
        return list(recovered_cands) if recovered_cands else []

    c_arr = np.array(cand_ids, dtype=object)
    src_arr = np.array(cand_sources, dtype=object)

    # 1. Sort descending by probability
    sort_idx = np.argsort(-probs)
    p_s = probs[sort_idx]
    c_s = c_arr[sort_idx]
    src_s = src_arr[sort_idx]
    cos_s = cosines[sort_idx]
    aj_s = addr_jaccards[sort_idx]
    post_s = postal_matches[sort_idx]
    bs_s = branch_sims[sort_idx]
    hn_s = house_nums[sort_idx]

    # 2. Base N020 Frontier Acceptance
    am = (p_s >= 0.9570)
    recovery_n020 = (p_s >= 0.9320) & ((hn_s == 1) | (aj_s >= 0.30)) & (bs_s == 0)
    pred_mask = am | recovery_n020

    # 3. Multi-anchor & relational set completion
    acc_sibs = set(c_s[pred_mask])
    if acc_sibs:
        for r in range(min(80, K)):
            if pred_mask[r] or bs_s[r] > 0:
                continue
            cid = c_s[r]
            conn_count = sum(1 for sib in acc_sibs if cid in training_edges.get(sib, set()))
            if conn_count == 0:
                continue

            cos = cos_s[r]
            aj = aj_s[r]
            post = post_s[r]

            # Multi-anchor dual support (conn >= 2, cos >= 0.57)
            r_multi = (conn_count >= 2) and (cos >= 0.57)

            # Single-anchor guarded support (conn >= 1, cos >= 0.65, address guard)
            r_single = (
                (conn_count >= 1)
                and (cos >= 0.65)
                and (aj >= 0.25 or (post >= 0.66 and aj >= 0.20))
            )

            if r_multi or r_single:
                pred_mask[r] = True

    final_preds = set(c_s[pred_mask])
    if recovered_cands:
        final_preds.update(recovered_cands)

    return sorted(final_preds)
