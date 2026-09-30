#!/usr/bin/env python3
"""
Authoritative "N039_NO_FRONTIER" decision module.

This is a NEW, additive definition of the base-admission chain that produces
pred_n038_matrix.npz, MINUS the frontier-ranker step identified and ablated
in an earlier investigation (see reports/N039_NO_FRONTIER_MANIFEST.md).

Relationship to existing modules:
  - src/n038_threshold_decision.py::compute_pred_n038() remains UNCHANGED and
    continues to reproduce the ORIGINAL (frontier-ranker-included) N038/N039
    baseline. It is not modified or removed by this file.
  - This file provides an alternative base-admission function that skips the
    y_s-label-dependent frontier-ranker cascade entirely. Because this
    function has NO ground-truth-label dependency anywhere in its call
    graph (confirmed: no argument to any function here is a labels array),
    it is -- unlike compute_pred_n038() -- structurally eligible to run on
    unlabeled data (e.g. test), modulo the separate, unresolved candidate
    generation / feature / test-embedding gaps documented elsewhere.

Traced load-bearing chain WITHOUT the frontier-ranker:
  1. Base frontier formula (identical to run_n021.py lines 178-181).
  2. [SKIPPED] Frontier-ranker 3-round XGBoost CV cascade -- REMOVED.
     Whatever admission state existed immediately after step 1 (pred_base)
     flows directly into step 3 as pred_A_base, taking the place of the
     frontier-ranker's output.
  3. Deterministic multi-hop training_edges graph rescue chain
     (N031 -> N032 -> N033 -> N034 -> N035), verbatim from run_n036.py Step 5.
  4. N036's "Clean Unanchored Bootstrap" (p >= 0.940, bs == 0, zero-prediction
     entities only).
  5. N037's TestTimeReciprocalEntityInducer, applied only to entities still
     unanchored after step 4.
  6. N038's TestTimeEntitySetDecoder, applied to all entities.
  7. N039's TestTimeSetCompletionDecoder (multi-anchor / single-anchor set
     completion), applied on top of step 6's result.
  8. Sibling/graph recovery union (recovered_map from n014_recovered_matches.npz
     + blocker_recovery_matrix.csv) -- caller's responsibility, same as the
     original pipeline; NOT performed inside this module (see caller code in
     the Phase-1 freeze script / validation scripts for the union step).

Upstream scoring (candidate generation, 57-feature extraction, LGB/XGB
heuristic-or-zero treatment of ANN-new pairs, XLM-R blending) is UNCHANGED
and NOT reimplemented here -- callers must supply the same p_hyb (the
existing authoritative heuristic-based hybrid probability), sorted per
entity, exactly as scripts/run_n039.py does today. This module only replaces
the base-admission/threshold layer.
"""

from __future__ import annotations

from typing import Dict, Set

import numpy as np

from test_time_entity_set import TestTimeEntitySetDecoder
from test_time_reciprocal_entity import TestTimeReciprocalEntityInducer
from test_time_set_completion import TestTimeSetCompletionDecoder


def compute_pred_n038_no_frontier(
    p_s: np.ndarray,
    X_s: np.ndarray,
    c_s: np.ndarray,
    src_s: np.ndarray,
    training_edges: Dict[str, Set[str]],
) -> np.ndarray:
    """
    Reproduces the "no-frontier-ranker" equivalent of pred_n038_matrix.npz.

    Inputs must already be sorted descending by hybrid probability per
    entity (row_idx, sort_idx as done in every run_n0XX.py script). Unlike
    src/n038_threshold_decision.py::compute_pred_n038(), this function takes
    NO ground-truth labels array anywhere in its signature or call graph.
    """
    N_ent = p_s.shape[0]
    hn_s = X_s[:, :, 40]
    bs_s = X_s[:, :, 51]
    aj_s = X_s[:, :, 30]
    postal_s = X_s[:, :, 41]
    neural_cos_s = X_s[:, :, 56]

    # --- Step 1: base frontier formula ---
    am = (p_s >= 0.9570)
    recovery_n020 = (p_s >= 0.9320) & ((hn_s == 1) | (aj_s >= 0.30)) & (bs_s == 0)
    pred_base = am | recovery_n020

    # --- Step 2: [SKIPPED] frontier-ranker. pred_base flows straight through. ---
    pred_A_base = pred_base

    # --- Step 3: deterministic multi-hop rescue chain (run_n036.py Step 5) ---
    pred_N031 = pred_A_base.copy()
    s1_acc_sets = [set(c_s[i][pred_A_base[i]]) for i in range(N_ent)]
    for i in range(N_ent):
        acc_sibs = s1_acc_sets[i]
        for r in range(1, 10):
            if not pred_N031[i, r] and bs_s[i, r] == 0:
                cid = c_s[i, r]
                if (p_s[i, r] >= 0.30) and any(cid in training_edges.get(sib, set()) for sib in acc_sibs):
                    pred_N031[i, r] = True

    pred_N032 = pred_N031.copy()
    for i in range(N_ent):
        acc_sibs = s1_acc_sets[i]
        if not acc_sibs:
            continue
        for r in range(1, 20):
            if not pred_N032[i, r] and bs_s[i, r] == 0:
                cid = c_s[i, r]
                is_len2 = (p_s[i, r] >= 0.05) and any(cid in training_edges.get(sib, set()) for sib in acc_sibs)
                is_len3 = False
                if not is_len2 and p_s[i, r] >= 0.20:
                    for sib in acc_sibs:
                        for nbr in training_edges.get(sib, set()):
                            if cid in training_edges.get(nbr, set()):
                                is_len3 = True
                                break
                        if is_len3:
                            break
                if is_len2 or is_len3:
                    pred_N032[i, r] = True

    pred_N033 = pred_N032.copy()
    for i in range(N_ent):
        acc_sibs = s1_acc_sets[i]
        if not acc_sibs:
            continue
        for r in range(1, 80):
            if not pred_N033[i, r] and bs_s[i, r] == 0:
                cid = c_s[i, r]
                conn_count = sum(1 for sib in acc_sibs if cid in training_edges.get(sib, set()))
                if conn_count >= 2 and neural_cos_s[i, r] >= 0.45:
                    pred_N033[i, r] = True
                elif conn_count >= 1:
                    cos = neural_cos_s[i, r]
                    aj = aj_s[i, r]
                    hn = hn_s[i, r]
                    p = p_s[i, r]
                    guard = (p >= 0.01) or (cos >= 0.685) or (cos >= 0.505 and aj >= 0.25) or (hn == 1 and cos >= 0.505)
                    if guard:
                        pred_N033[i, r] = True

    pred_N034 = pred_N033.copy()
    for i in range(N_ent):
        acc_sibs = s1_acc_sets[i]
        if not acc_sibs:
            continue
        for r in range(1, 80):
            if not pred_N034[i, r]:
                cid = c_s[i, r]
                conn_count = sum(1 for sib in acc_sibs if cid in training_edges.get(sib, set()))
                rule_b = (conn_count >= 1) and (bs_s[i, r] > 0)
                rule_m3 = (conn_count >= 3) and (bs_s[i, r] == 0)
                if rule_b or rule_m3:
                    pred_N034[i, r] = True

    pred_N035 = pred_N034.copy()
    for i in range(N_ent):
        if np.sum(pred_N034[i]) == 0:
            cands_i = c_s[i]
            for r1 in range(0, 2):
                c1 = cands_i[r1]
                for r2 in range(r1 + 1, 2):
                    c2 = cands_i[r2]
                    if c1[:2] != c2[:2] and c2 in training_edges.get(c1, set()):
                        for r_cur, c_cur in [(r1, c1), (r2, c2)]:
                            if neural_cos_s[i, r_cur] >= 0.70 and aj_s[i, r_cur] >= 0.35:
                                pred_N035[i, r_cur] = True

    # --- Step 4: N036's load-bearing addition ("Clean Unanchored Bootstrap") ---
    pred_N036 = pred_N035.copy()
    for i in range(N_ent):
        if np.sum(pred_N035[i]) > 0:
            continue
        for r in range(0, 3):
            if p_s[i, r] >= 0.940 and bs_s[i, r] == 0:
                pred_N036[i, r] = True

    # --- Step 5: N037's load-bearing addition (TestTimeReciprocalEntityInducer) ---
    unanchored_mask = (np.sum(pred_N036, axis=1) == 0)
    unanchored_indices = np.where(unanchored_mask)[0]
    inducer = TestTimeReciprocalEntityInducer(min_cos=0.62, max_rank=10)
    pred_N037 = pred_N036.copy()
    for i in unanchored_indices:
        admitted = inducer.decode_unanchored_candidates(
            c_s[i], src_s[i], p_s[i], neural_cos_s[i], aj_s[i], postal_s[i], bs_s[i], training_edges
        )
        for r in admitted:
            pred_N037[i, r] = True

    # --- Step 6: N038's load-bearing addition (TestTimeEntitySetDecoder) ---
    decoder38 = TestTimeEntitySetDecoder()
    pred_N038_nf = pred_N037.copy()
    for i in range(N_ent):
        new_admitted = decoder38.decode_entity_set(
            c_s[i], src_s[i], p_s[i], neural_cos_s[i], aj_s[i], postal_s[i], bs_s[i], pred_N037[i], training_edges
        )
        for r in new_admitted:
            pred_N038_nf[i, r] = True

    return pred_N038_nf


def compute_authoritative_no_frontier_predictions(
    p_s: np.ndarray,
    X_s: np.ndarray,
    c_s: np.ndarray,
    src_s: np.ndarray,
    training_edges: Dict[str, Set[str]],
) -> np.ndarray:
    """
    Full "N039_NO_FRONTIER" pipeline entry point: base-admission-without-
    frontier-ranker (compute_pred_n038_no_frontier) followed by the
    UNCHANGED N039 set-completion decoder (TestTimeSetCompletionDecoder).

    Does NOT perform the sibling/graph recovered_map union -- callers must
    still union in recovered_map (from n014_recovered_matches.npz and
    blocker_recovery_matrix.csv) themselves, exactly as the original N039
    pipeline does, since that step is outside the admission/threshold layer
    this module replaces.
    """
    pred_N038_nf = compute_pred_n038_no_frontier(p_s, X_s, c_s, src_s, training_edges)

    hn_s = X_s[:, :, 40]
    bs_s = X_s[:, :, 51]
    aj_s = X_s[:, :, 30]
    postal_s = X_s[:, :, 41]
    neural_cos_s = X_s[:, :, 56]

    decoder39 = TestTimeSetCompletionDecoder()
    N_ent = p_s.shape[0]
    pred_final = pred_N038_nf.copy()
    for i in range(N_ent):
        new_admitted = decoder39.decode_entity_set_completion(
            c_s[i], src_s[i], p_s[i], neural_cos_s[i], aj_s[i], postal_s[i], bs_s[i], pred_N038_nf[i], training_edges
        )
        for r in new_admitted:
            pred_final[i, r] = True

    return pred_final
