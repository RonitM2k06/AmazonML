#!/usr/bin/env python3
"""
Extracted, load-bearing threshold/decision logic that produces pred_n038_matrix.npz.

Traced chain (see accompanying investigation report for full provenance):
  1. Base frontier formula (scripts/run_n021.py lines 178-181).
  2. Three rounds of a freshly-trained, 5-fold-CV XGBoost "frontier ranker"
     cascade (scripts/run_n036.py Step 3, using extract_frontier_features
     from scripts/run_n032.py). NOTE: this step trains on the validation
     set's OWN ground-truth labels (y_s) via out-of-fold CV predictions.
     This is validation-label-dependent and CANNOT be run on unlabeled
     (e.g. test) data -- see module docstring warning below.
  3. Deterministic multi-hop training-edge graph rescue chain
     (N031 -> N032 -> N033 -> N034 -> N035), scripts/run_n036.py Step 5.
  4. N036's own load-bearing addition: "Clean Unanchored Bootstrap"
     (p >= 0.940, bs == 0, only for entities with zero current predictions).
  5. N037's own load-bearing addition: TestTimeReciprocalEntityInducer
     applied only to entities still unanchored after step 4.
  6. N038's own load-bearing addition: TestTimeEntitySetDecoder applied to
     ALL entities on top of the N037 result.

IMPORTANT / DO NOT USE FOR TEST DATA AS-IS:
  Step 2 requires y_s (ground-truth match labels) to train the frontier
  ranker via K-fold cross-validation. This is available for the frozen
  validation set (val_pairs_n010.npz's own y array) but NOT available for
  a genuinely unlabeled test set. This function is therefore only valid
  for reproducing the frozen validation result; it is not a test-time-ready
  decision function without further work (see investigation report).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Set, Tuple

import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold

from test_time_entity_set import TestTimeEntitySetDecoder
from test_time_reciprocal_entity import TestTimeReciprocalEntityInducer


def extract_frontier_features(
    pred_mask: np.ndarray,
    p_s: np.ndarray,
    X_s: np.ndarray,
    y_s: np.ndarray,
    src_s: np.ndarray,
    aj_s: np.ndarray,
    hn_s: np.ndarray,
    bs_s: np.ndarray,
    addr_miss_s: np.ndarray,
    max_rank: int = 10,
    include_representation_features: bool = True,
):
    """Verbatim copy of scripts/run_n032.py::extract_frontier_features."""
    frontier_feats = []
    frontier_labels = []
    frontier_coords = []

    for r in range(1, max_rank):
        prev_acc = pred_mask[:, r - 1]
        not_acc = ~pred_mask[:, r]
        frontier_mask = prev_acc & not_acc
        e_idxs = np.where(frontier_mask)[0]

        for e in e_idxs:
            acc_s2 = (pred_mask[e] & (src_s[e] == "S2")).sum()
            acc_s3 = (pred_mask[e] & (src_s[e] == "S3")).sum()
            cand_src = 1.0 if src_s[e, r] == "S2" else 0.0
            acc_indices = np.where(pred_mask[e])[0]

            feat = [
                float(r),
                float(p_s[e, r]),
                float(p_s[e, r - 1] - p_s[e, r]),
                float(p_s[e, r] - (p_s[e, r + 1] if r < 79 else 0.0)),
                float(aj_s[e, r]),
                float(hn_s[e, r]),
                float(bs_s[e, r]),
                float(addr_miss_s[e, r]),
                float(X_s[e, r, 0]),
                float(X_s[e, r, 1]),
                float(X_s[e, r, 5]),
                float(X_s[e, r, 9]),
                float(X_s[e, r, 11]),
                float(X_s[e, r, 15]),
                float(X_s[e, r, 26]),
                float(X_s[e, r, 27]),
                float(X_s[e, r, 56]),
                float(acc_s2),
                float(acc_s3),
                cand_src,
            ]

            if include_representation_features:
                cand_cosine = float(X_s[e, r, 56])
                max_acc_cosine = float(np.max(X_s[e, acc_indices, 56]))
                min_acc_cosine = float(np.min(X_s[e, acc_indices, 56]))
                max_acc_aj = float(np.max(aj_s[e, acc_indices]))
                mean_acc_p = float(np.mean(p_s[e, acc_indices]))

                rep_feats = [
                    cand_cosine,
                    cand_cosine - max_acc_cosine,
                    cand_cosine - min_acc_cosine,
                    max_acc_aj,
                    float(len(acc_indices)),
                    mean_acc_p - float(p_s[e, r]),
                ]
                feat.extend(rep_feats)

            frontier_feats.append(feat)
            frontier_labels.append(int(y_s[e, r]))
            frontier_coords.append((e, r))

    return (
        np.array(frontier_feats, dtype=np.float32),
        np.array(frontier_labels, dtype=np.float32),
        frontier_coords,
    )


def compute_pred_n038(
    p_s: np.ndarray,
    X_s: np.ndarray,
    y_s: np.ndarray,
    c_s: np.ndarray,
    src_s: np.ndarray,
    training_edges: Dict[str, Set[str]],
) -> np.ndarray:
    """
    Reproduces pred_n038_matrix.npz from scratch given the same sorted-by-p
    per-entity inputs used throughout run_n021.py..run_n038.py.

    All inputs must already be sorted descending by hybrid probability per
    entity (row_idx, sort_idx as done in every run_n0XX.py script).
    """
    N_ent = p_s.shape[0]
    hn_s = X_s[:, :, 40]
    bs_s = X_s[:, :, 51]
    aj_s = X_s[:, :, 30]
    addr_miss_s = X_s[:, :, 44]
    postal_s = X_s[:, :, 41]
    neural_cos_s = X_s[:, :, 56]

    # --- Step 1: base frontier formula (run_n021.py lines 178-181) ---
    am = (p_s >= 0.9570)
    recovery_n020 = (p_s >= 0.9320) & ((hn_s == 1) | (aj_s >= 0.30)) & (bs_s == 0)
    pred_base = am | recovery_n020

    # --- Step 2: three rounds of frontier-ranker cascade (run_n036.py Step 3) ---
    f_feats, f_labels, f_coords = extract_frontier_features(
        pred_base, p_s, X_s, y_s, src_s, aj_s, hn_s, bs_s, addr_miss_s,
        max_rank=10, include_representation_features=True,
    )
    p1_cv = np.zeros(len(f_labels), dtype=np.float32)
    kf1 = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in kf1.split(f_feats):
        dtrain = xgb.DMatrix(f_feats[train_idx], label=f_labels[train_idx])
        dval = xgb.DMatrix(f_feats[val_idx])
        params = {"max_depth": 4, "eta": 0.05, "objective": "binary:logistic", "eval_metric": "auc", "verbosity": 0, "nthread": 1}
        bst1 = xgb.train(params, dtrain, num_boost_round=150)
        p1_cv[val_idx] = bst1.predict(dval)

    pred_E = pred_base.copy()
    admit_E = p1_cv >= 0.70
    for idx, (e, r) in enumerate(f_coords):
        if admit_E[idx]:
            pred_E[e, r] = True

    f2_feats, f2_labels, f2_coords = extract_frontier_features(
        pred_E, p_s, X_s, y_s, src_s, aj_s, hn_s, bs_s, addr_miss_s,
        max_rank=12, include_representation_features=True,
    )
    p2_cv = np.zeros(len(f2_labels), dtype=np.float32)
    kf2 = KFold(n_splits=5, shuffle=True, random_state=142)
    for train_idx, val_idx in kf2.split(f2_feats):
        dtrain = xgb.DMatrix(f2_feats[train_idx], label=f2_labels[train_idx])
        dval = xgb.DMatrix(f2_feats[val_idx])
        params = {"max_depth": 4, "eta": 0.04, "objective": "binary:logistic", "eval_metric": "auc", "verbosity": 0, "nthread": 1}
        bst2 = xgb.train(params, dtrain, num_boost_round=120)
        p2_cv[val_idx] = bst2.predict(dval)

    pred_F = pred_E.copy()
    admit_F = p2_cv >= 0.75
    for idx, (e, r) in enumerate(f2_coords):
        if admit_F[idx]:
            pred_F[e, r] = True

    f3_feats, f3_labels, f3_coords = extract_frontier_features(
        pred_F, p_s, X_s, y_s, src_s, aj_s, hn_s, bs_s, addr_miss_s,
        max_rank=14, include_representation_features=True,
    )
    p3_cv = np.zeros(len(f3_labels), dtype=np.float32)
    kf3 = KFold(n_splits=5, shuffle=True, random_state=242)
    for train_idx, val_idx in kf3.split(f3_feats):
        dtrain = xgb.DMatrix(f3_feats[train_idx], label=f3_labels[train_idx])
        dval = xgb.DMatrix(f3_feats[val_idx])
        params = {"max_depth": 4, "eta": 0.04, "objective": "binary:logistic", "eval_metric": "auc", "verbosity": 0, "nthread": 1}
        bst3 = xgb.train(params, dtrain, num_boost_round=100)
        p3_cv[val_idx] = bst3.predict(dval)

    pred_A_base = pred_F.copy()
    admit_G = p3_cv >= 0.70
    for idx, (e, r) in enumerate(f3_coords):
        if admit_G[idx]:
            pred_A_base[e, r] = True

    unique_idx_placeholder = None  # entity IDs not needed inside this function

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
    decoder = TestTimeEntitySetDecoder()
    pred_N038 = pred_N037.copy()
    for i in range(N_ent):
        new_admitted = decoder.decode_entity_set(
            c_s[i], src_s[i], p_s[i], neural_cos_s[i], aj_s[i], postal_s[i], bs_s[i], pred_N037[i], training_edges
        )
        for r in new_admitted:
            pred_N038[i, r] = True

    return pred_N038
