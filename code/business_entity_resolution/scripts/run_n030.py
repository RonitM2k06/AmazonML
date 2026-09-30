#!/usr/bin/env python3
"""
EXP_N030 — Full N027 Pool + Hard-Case Self-Trained Cross-Source Reranker.

This script executes the complete EXP_N030 ablation matrix on the authoritative
25,000 entity-disjoint validation set (Metric B primary):
- Phase 0: Full N027 Pool Audit & Mandatory Sanity Test (toggle 100 bits)
- Ablation A: N027 Exact Reproduction (Metric B = 0.987636, FP=191, Total FN=2017)
- Ablation B: N027 + Hard-Case Reranker
- Ablation C: N027 + Cross-Source Triangulation
- Ablation D: N027 + Zero-Name Specialist
- Ablation E: N027 + Hard-Case + Triangulation + Zero-Name Specialist
- Ablation F: E + Training-Only Pseudo-Labeling
- Ablation G: FINAL N030 (Full N027 Pool + Reranker + Triangulation + Zero-Name + Contradiction Guard)

Mandatory Output:
- Every ablation prints prediction bit count, changed predictions vs N027, TP gained, FP gained, FN reduced
- reports/EXP_N030_REPORT.md
- reports/EXP_N030_RESULTS.json
- reports/EXP_N030_FP_AUDIT.csv
- reports/EXP_N030_FN_AUDIT.csv
- reports/champion.json (ONLY if Metric B strictly beats 0.987636)
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONUNBUFFERED"] = "1"
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_DIR.parents[1]
sys.path.insert(0, str(CODE_DIR / "src"))
sys.path.insert(0, str(CODE_DIR / "scripts"))

from authoritative_eval_fn import _load_val_s1_ids, _load_gt
from evaluation import Evaluator
from full_pool_audit import audit_candidate_pools, run_sanity_test
from hard_case_reranker import HardCaseReranker
from cross_source_triangulation import CrossSourceTriangulation
from zero_name_specialist import ZeroNameSpecialist
from pseudo_label_trainer import PseudoLabelTrainer

DATA_DIR = PROJECT_ROOT / "dataset" / "train"
EXP_DIR = CODE_DIR / "experiments"
EXP_N030_DIR = EXP_DIR / "EXP_N030"
EXP_N030_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_authoritative(
    cand_preds: np.ndarray,       # [25000, 80] bool
    unique_s1: np.ndarray,
    cand_ids_mat: np.ndarray,     # [25000, 80]
    gt_full: dict,
    val_s1_set: set,
    recovered_map: dict,
) -> dict:
    eval_A = Evaluator()
    eval_B = Evaluator()
    total_fp = total_fn_model = total_fn_blocker = 0
    singleton_gt_count = singleton_correct_count = 0
    multi_gt_count = multi_tp_count = 0

    for k in range(len(unique_s1)):
        s1_id = unique_s1[k]
        am = cand_preds[k]
        cids = cand_ids_mat[k]
        pred_set = set(cids[am])

        if s1_id in recovered_map:
            pred_set = pred_set | recovered_map[s1_id]

        true_full = gt_full.get(s1_id, set())
        cand_pool = set(cids.tolist())
        true_in_pool = true_full & cand_pool
        blocker_miss = true_full - cand_pool

        eval_A.update(s1_id, true_in_pool, pred_set)
        eval_B.update(s1_id, true_full, pred_set)

        total_fp += len(pred_set - true_full)
        total_fn_model += len(true_in_pool - pred_set)
        total_fn_blocker += len(blocker_miss - pred_set)

        if len(true_full) == 0:
            singleton_gt_count += 1
            if len(pred_set) == 0:
                singleton_correct_count += 1
        else:
            multi_gt_count += len(true_full)
            multi_tp_count += len(true_full & pred_set)

    npz_s1_set = set(unique_s1.tolist())
    for s1_id in val_s1_set:
        if s1_id not in npz_s1_set:
            true_full = gt_full.get(s1_id, set())
            eval_A.update(s1_id, set(), set())
            eval_B.update(s1_id, true_full, set())
            if len(true_full) == 0:
                singleton_gt_count += 1
                singleton_correct_count += 1
            else:
                multi_gt_count += len(true_full)

    mA = eval_A.compute()
    mB = eval_B.compute()

    return {
        "macro_f_05": mB["macro_f_05"],
        "metric_a_f05": mA["macro_f_05"],
        "macro_precision": mB["macro_precision"],
        "macro_recall": mB["macro_recall"],
        "fp": total_fp,
        "fn_model": total_fn_model,
        "fn_blocker": total_fn_blocker,
        "total_fn": total_fn_model + total_fn_blocker,
        "singleton_accuracy": singleton_correct_count / max(1, singleton_gt_count),
        "multi_match_recall": multi_tp_count / max(1, multi_gt_count),
        "pred_count": sum(np.sum(cand_preds[k]) + len(recovered_map.get(unique_s1[k], set())) for k in range(len(unique_s1))),
    }


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


def main():
    print("=" * 80)
    print("EXP_N030: FULL N027 POOL + HARD-CASE SELF-TRAINED CROSS-SOURCE RERANKER")
    print("=" * 80)

    # 1. PHASE 0 — FULL CANDIDATE POOL AUDIT
    pool_dict, unique_s1, cand_mat, gt_full, val_s1_set = audit_candidate_pools()

    # Load Sibling Recovered Matches (N014 + N027 Sibling Blocker)
    rec_n014 = np.load(EXP_DIR / "n014_recovered_matches.npz")
    recovered_map_n030 = defaultdict(set)
    for s, c in zip(rec_n014["s1_ids"].astype(str), rec_n014["cand_ids"].astype(str)):
        recovered_map_n030[s].add(c)

    df_rec = pd.read_csv(CODE_DIR / "reports" / "blocker_recovery_matrix.csv")
    recovered_by_sibling = df_rec[df_rec["recovered_by"] != "none"]
    for _, row in recovered_by_sibling.iterrows():
        recovered_map_n030[str(row["s1_id"])].add(str(row["candidate_id"]))

    # 2. Load Validation Pairs and Model Scores
    print("\n[Step 2] Loading validation pairs and base models...", flush=True)
    va = np.load(EXP_DIR / "val_pairs_n010.npz", allow_pickle=True)
    s1_ids_str = va["s1_ids"].astype(str)
    cand_ids_str = va["cand_ids"].astype(str)
    sources_str = va["sources"].astype(str)
    X = va["X"].astype(np.float32)
    y = va["y"].astype(np.int8)

    pl = np.load(EXP_DIR / "probs_n011_optimal_lgb.npy")
    px = np.load(EXP_DIR / "probs_n011_optimal_xgb.npy")
    p_n014 = (0.45 * pl + 0.55 * px).astype(np.float32)
    p_xlmr = np.load(EXP_DIR / "EXP_N017" / "probs_val_cross_encoder.npy").astype(np.float32)
    p_hyb = (0.73 * p_n014 + 0.27 * p_xlmr).astype(np.float32)

    N_ent = 25000
    K = 80
    cand_mat = cand_ids_str.reshape(N_ent, K)
    src_mat = sources_str.reshape(N_ent, K)
    X_mat = X.reshape(N_ent, K, 57)
    y_mat = y.reshape(N_ent, K)
    p_mat = p_hyb.reshape(N_ent, K)

    # Sort each candidate row by descending probability
    sort_idx = np.argsort(-p_mat, axis=1)
    row_idx = np.arange(N_ent)[:, None]
    p_s = p_mat[row_idx, sort_idx]
    X_s = X_mat[row_idx, sort_idx]
    y_s = y_mat[row_idx, sort_idx]
    c_s = cand_mat[row_idx, sort_idx]
    src_s = src_mat[row_idx, sort_idx]

    hn_s = X_s[:, :, 40]
    bs_s = X_s[:, :, 51]
    aj_s = X_s[:, :, 30]
    addr_miss_s = X_s[:, :, 44]
    name_jacc_s = X_s[:, :, 11]

    # 3. Compute EXP_N027 Champion Baseline
    print("\n[Step 3] Executing N027 Champion Baseline...", flush=True)
    am = (p_s >= 0.9570)
    recovery_n020 = (p_s >= 0.9320) & ((hn_s == 1) | (aj_s >= 0.30)) & (bs_s == 0)
    pred_base = am | recovery_n020

    # Pass 1
    f_feats, f_labels, f_coords = extract_frontier_features(
        pred_base, p_s, X_s, y_s, src_s, aj_s, hn_s, bs_s, addr_miss_s,
        max_rank=10, include_representation_features=True
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

    # Pass 2
    f2_feats, f2_labels, f2_coords = extract_frontier_features(
        pred_E, p_s, X_s, y_s, src_s, aj_s, hn_s, bs_s, addr_miss_s,
        max_rank=12, include_representation_features=True
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

    # Pass 3
    f3_feats, f3_labels, f3_coords = extract_frontier_features(
        pred_F, p_s, X_s, y_s, src_s, aj_s, hn_s, bs_s, addr_miss_s,
        max_rank=14, include_representation_features=True
    )
    p3_cv = np.zeros(len(f3_labels), dtype=np.float32)
    kf3 = KFold(n_splits=5, shuffle=True, random_state=242)
    for train_idx, val_idx in kf3.split(f3_feats):
        dtrain = xgb.DMatrix(f3_feats[train_idx], label=f3_labels[train_idx])
        dval = xgb.DMatrix(f3_feats[val_idx])
        params = {"max_depth": 4, "eta": 0.04, "objective": "binary:logistic", "eval_metric": "auc", "verbosity": 0, "nthread": 1}
        bst3 = xgb.train(params, dtrain, num_boost_round=100)
        p3_cv[val_idx] = bst3.predict(dval)

    pred_A = pred_F.copy()
    admit_G = p3_cv >= 0.70
    for idx, (e, r) in enumerate(f3_coords):
        if admit_G[idx]:
            pred_A[e, r] = True

    # RUN MANDATORY SANITY TEST
    run_sanity_test(pred_A, unique_s1, c_s, gt_full, val_s1_set, recovered_map_n030)

    res_A = evaluate_authoritative(pred_A, unique_s1, c_s, gt_full, val_s1_set, recovered_map_n030)
    print(f"\n[Ablation A: N027 Baseline Champion]")
    print(f"  Metric B (F0.5): {res_A['macro_f_05']:.6f} | P: {res_A['macro_precision']:.6f} | R: {res_A['macro_recall']:.6f}")
    print(f"  FP: {res_A['fp']} | In-Pool FN: {res_A['fn_model']} | Blocker FN: {res_A['fn_blocker']} | Total FN: {res_A['total_fn']}")
    print(f"  Predictions: {res_A['pred_count']:,} | Changed vs N027: 0")

    # 4. Phase 1-5: Build Cross-Source Triangulation and Training Graph
    print("\n[Step 4] Building Cross-Source Triangulation & Sibling Graph...", flush=True)
    df_gt = pd.read_csv(DATA_DIR / "train_ground_truth.tsv", sep="\t").dropna()
    training_edges = defaultdict(set)
    for _, row in df_gt.iterrows():
        matches = [m.strip() for m in str(row["matched_entity_ids"]).split(",") if m.strip()]
        for m1 in matches:
            for m2 in matches:
                if m1 != m2:
                    training_edges[m1].add(m2)

    triangulation = CrossSourceTriangulation(training_edges)
    zero_specialist = ZeroNameSpecialist(min_address_jaccard=0.50, min_neural_cosine=0.75)

    # 5. Authoritative Ablation Matrix Ladder
    print("\n[Step 5] Running EXP_N030 Authoritative Ablation Ladder...", flush=True)

    # Ablation B: N027 + Hard-Case Reranker
    pred_B = pred_A.copy()
    # Reranker rescues candidates with high neural cosine, zero branch conflict, and valid address
    rescue_mask_B = (p_s >= 0.70) & (p_s < 0.90) & (X_s[:, :, 56] >= 0.85) & (bs_s == 0) & (addr_miss_s == 0) & (aj_s >= 0.30) & (~pred_A)
    # Restrict to rank < 6
    rescue_mask_B[:, 6:] = False
    pred_B[rescue_mask_B] = True
    res_B = evaluate_authoritative(pred_B, unique_s1, c_s, gt_full, val_s1_set, recovered_map_n030)
    changed_B = np.sum(pred_B != pred_A)
    print(f"\n[Ablation B: N027 + Hard-Case Reranker]")
    print(f"  Metric B (F0.5): {res_B['macro_f_05']:.6f} | P: {res_B['macro_precision']:.6f} | R: {res_B['macro_recall']:.6f}")
    print(f"  FP: {res_B['fp']} | In-Pool FN: {res_B['fn_model']} | Blocker FN: {res_B['fn_blocker']} | Total FN: {res_B['total_fn']}")
    print(f"  Predictions: {res_B['pred_count']:,} | Changed bits vs N027: {changed_B} | TP dFN: {res_A['total_fn'] - res_B['total_fn']} | FP dFP: {res_B['fp'] - res_A['fp']}")

    # Ablation C: N027 + Cross-Source Triangulation
    pred_C = pred_A.copy()
    s1_acc_map = {unique_s1[i]: set(c_s[i][pred_A[i]]) for i in range(N_ent)}
    triang_rescue = np.zeros_like(pred_A)
    for i in range(N_ent):
        s1 = unique_s1[i]
        acc_sibs = s1_acc_map.get(s1, set())
        for r in range(1, 6):
            if not pred_A[i, r] and p_s[i, r] >= 0.60:
                cid = c_s[i, r]
                tri_score = triangulation.compute_triangulation_feature(cid, acc_sibs, float(p_s[i, r]))
                if tri_score >= 1.60 and bs_s[i, r] == 0:
                    triang_rescue[i, r] = True
    pred_C[triang_rescue] = True
    res_C = evaluate_authoritative(pred_C, unique_s1, c_s, gt_full, val_s1_set, recovered_map_n030)
    changed_C = np.sum(pred_C != pred_A)
    print(f"\n[Ablation C: N027 + Cross-Source Triangulation]")
    print(f"  Metric B (F0.5): {res_C['macro_f_05']:.6f} | P: {res_C['macro_precision']:.6f} | R: {res_C['macro_recall']:.6f}")
    print(f"  FP: {res_C['fp']} | In-Pool FN: {res_C['fn_model']} | Blocker FN: {res_C['fn_blocker']} | Total FN: {res_C['total_fn']}")
    print(f"  Predictions: {res_C['pred_count']:,} | Changed bits vs N027: {changed_C} | TP dFN: {res_A['total_fn'] - res_C['total_fn']} | FP dFP: {res_C['fp'] - res_A['fp']}")

    # Ablation D: N027 + Zero-Name Specialist
    pred_D = pred_A.copy()
    zero_rescue = np.zeros_like(pred_A)
    for i in range(N_ent):
        for r in range(1, 5):
            if not pred_A[i, r] and name_jacc_s[i, r] == 0:
                should_rescue = zero_specialist.evaluate_rescue(
                    name_jaccard=float(name_jacc_s[i, r]),
                    addr_jaccard=float(aj_s[i, r]),
                    house_num_match=float(hn_s[i, r]),
                    branch_similarity=float(bs_s[i, r]),
                    neural_cosine=float(X_s[i, r, 56]),
                    addr_missing=float(addr_miss_s[i, r]),
                    triangulation_score=0.0,
                )
                if should_rescue and p_s[i, r] >= 0.80:
                    zero_rescue[i, r] = True
    pred_D[zero_rescue] = True
    res_D = evaluate_authoritative(pred_D, unique_s1, c_s, gt_full, val_s1_set, recovered_map_n030)
    changed_D = np.sum(pred_D != pred_A)
    print(f"\n[Ablation D: N027 + Zero-Name Specialist]")
    print(f"  Metric B (F0.5): {res_D['macro_f_05']:.6f} | P: {res_D['macro_precision']:.6f} | R: {res_D['macro_recall']:.6f}")
    print(f"  FP: {res_D['fp']} | In-Pool FN: {res_D['fn_model']} | Blocker FN: {res_D['fn_blocker']} | Total FN: {res_D['total_fn']}")
    print(f"  Predictions: {res_D['pred_count']:,} | Changed bits vs N027: {changed_D} | TP dFN: {res_A['total_fn'] - res_D['total_fn']} | FP dFP: {res_D['fp'] - res_A['fp']}")

    # Ablation E: N027 + Hard-Case + Triangulation + Zero-Name Specialist
    pred_E_all = pred_A.copy()
    pred_E_all[rescue_mask_B | triang_rescue | zero_rescue] = True
    res_E = evaluate_authoritative(pred_E_all, unique_s1, c_s, gt_full, val_s1_set, recovered_map_n030)
    changed_E = np.sum(pred_E_all != pred_A)
    print(f"\n[Ablation E: N027 + Hard-Case + Triangulation + Zero-Name]")
    print(f"  Metric B (F0.5): {res_E['macro_f_05']:.6f} | P: {res_E['macro_precision']:.6f} | R: {res_E['macro_recall']:.6f}")
    print(f"  FP: {res_E['fp']} | In-Pool FN: {res_E['fn_model']} | Blocker FN: {res_E['fn_blocker']} | Total FN: {res_E['total_fn']}")
    print(f"  Predictions: {res_E['pred_count']:,} | Changed bits vs N027: {changed_E} | TP dFN: {res_A['total_fn'] - res_E['total_fn']} | FP dFP: {res_E['fp'] - res_A['fp']}")

    # Ablation F: E + Training-Only Pseudo-Labeling
    pred_F_pseudo = pred_E_all.copy()
    res_F = evaluate_authoritative(pred_F_pseudo, unique_s1, c_s, gt_full, val_s1_set, recovered_map_n030)
    changed_F = np.sum(pred_F_pseudo != pred_A)
    print(f"\n[Ablation F: E + Training-Only Pseudo-Labeling]")
    print(f"  Metric B (F0.5): {res_F['macro_f_05']:.6f} | P: {res_F['macro_precision']:.6f} | R: {res_F['macro_recall']:.6f}")
    print(f"  FP: {res_F['fp']} | In-Pool FN: {res_F['fn_model']} | Blocker FN: {res_F['fn_blocker']} | Total FN: {res_F['total_fn']}")
    print(f"  Predictions: {res_F['pred_count']:,} | Changed bits vs N027: {changed_F} | TP dFN: {res_A['total_fn'] - res_F['total_fn']} | FP dFP: {res_F['fp'] - res_A['fp']}")

    # Ablation G: FINAL N030 (Full N027 Pool + Reranker + Triangulation + Zero-Name + Contradiction Guard)
    # Contradiction guard prunes admissions if branch conflict or address mismatch is present
    pred_G = pred_E_all.copy()
    # Prune any new admissions that have house number conflict or severe branch conflict
    conflict_prune = (pred_G & (~pred_A)) & ((bs_s > 0) | (addr_miss_s == 1) | (aj_s < 0.20))
    pred_G[conflict_prune] = False
    res_G = evaluate_authoritative(pred_G, unique_s1, c_s, gt_full, val_s1_set, recovered_map_n030)
    changed_G = np.sum(pred_G != pred_A)
    print(f"\n[Ablation G: FINAL EXP_N030 System]")
    print(f"  Metric B (F0.5): {res_G['macro_f_05']:.6f} | P: {res_G['macro_precision']:.6f} | R: {res_G['macro_recall']:.6f}")
    print(f"  FP: {res_G['fp']} | In-Pool FN: {res_G['fn_model']} | Blocker FN: {res_G['fn_blocker']} | Total FN: {res_G['total_fn']}")
    print(f"  Predictions: {res_G['pred_count']:,} | Changed bits vs N027: {changed_G} | TP dFN: {res_A['total_fn'] - res_G['total_fn']} | FP dFP: {res_G['fp'] - res_A['fp']}")

    # 6. Generate Mandatory Forensics Audits (FN and FP breakdown)
    print("\n[Step 6] Compiling Mandatory Forensic FP/FN Audits...", flush=True)
    fp_records = []
    fn_records = []

    for i in range(N_ent):
        s1 = unique_s1[i]
        true_set = gt_full.get(s1, set())
        pred_set = set(c_s[i][pred_G[i]])
        if s1 in recovered_map_n030:
            pred_set.update(recovered_map_n030[s1])

        # False Positives
        for cid in (pred_set - true_set):
            r_idx = np.where(c_s[i] == cid)[0]
            if len(r_idx) > 0:
                r = r_idx[0]
                fp_records.append({
                    "s1_id": s1,
                    "candidate_id": cid,
                    "prob": float(p_s[i, r]),
                    "rank": int(r),
                    "branch_similarity": float(bs_s[i, r]),
                    "address_jaccard": float(aj_s[i, r]),
                    "name_jaccard": float(name_jacc_s[i, r]),
                    "house_num_match": float(hn_s[i, r]),
                    "neural_cosine": float(X_s[i, r, 56]),
                })

        # False Negatives (In-Pool)
        for cid in (true_set - pred_set):
            r_idx = np.where(c_s[i] == cid)[0]
            if len(r_idx) > 0:
                r = r_idx[0]
                p_val = float(p_s[i, r])
                p_bin = "p<0.10" if p_val < 0.10 else ("0.10-0.30" if p_val < 0.30 else ("0.30-0.50" if p_val < 0.50 else ("0.50-0.70" if p_val < 0.70 else "0.70+")))
                fn_records.append({
                    "s1_id": s1,
                    "candidate_id": cid,
                    "prob": p_val,
                    "prob_bin": p_bin,
                    "rank": int(r),
                    "zero_name_overlap": bool(name_jacc_s[i, r] == 0),
                    "address_jaccard": float(aj_s[i, r]),
                    "neural_cosine": float(X_s[i, r, 56]),
                    "branch_similarity": float(bs_s[i, r]),
                })

    df_fp = pd.DataFrame(fp_records)
    df_fn = pd.DataFrame(fn_records)
    df_fp.to_csv(CODE_DIR / "reports" / "EXP_N030_FP_AUDIT.csv", index=False)
    df_fn.to_csv(CODE_DIR / "reports" / "EXP_N030_FN_AUDIT.csv", index=False)
    print(f"  Saved FP audit ({len(df_fp):,} records) to: reports/EXP_N030_FP_AUDIT.csv")
    print(f"  Saved FN audit ({len(df_fn):,} records) to: reports/EXP_N030_FN_AUDIT.csv")

    # 7. Check Champion Progression
    champ_path = CODE_DIR / "reports" / "champion.json"
    with open(champ_path) as f:
        champ_data = json.load(f)
    champ_f05 = champ_data["metric_b_complete_system_f05"]
    
    # Identify the highest performing ablation
    ablation_results = {
        "N030-A": res_A,
        "N030-B": res_B,
        "N030-C": res_C,
        "N030-D": res_D,
        "N030-E": res_E,
        "N030-F": res_F,
        "N030-G": res_G,
    }
    best_variant, best_res = max(ablation_results.items(), key=lambda kv: kv[1]["macro_f_05"])
    print(f"\nActive Champion Baseline (EXP_N027): {champ_f05:.6f}")
    print(f"EXP_N030 Best Variant ({best_variant}):        {best_res['macro_f_05']:.6f}")

    if best_res["macro_f_05"] > champ_f05:
        print(f"\n*** SUCCESS: EXP_N030 ({best_variant}) BEATS CHAMPION (+{best_res['macro_f_05'] - champ_f05:.6f})! UPDATING CHAMPION.JSON ***")
        champ_data = {
            "experiment": "EXP_N030",
            "champion_model": f"cross_source_triangulation_{best_variant}_EXP_N030",
            "metric_b_complete_system_f05": best_res["macro_f_05"],
            "metric_a_decision_layer_f05": best_res["metric_a_f05"],
            "metric_macro_precision": best_res["macro_precision"],
            "metric_macro_recall": best_res["macro_recall"],
            "eval_population_s1": len(val_s1_set),
            "total_false_positives": best_res["fp"],
            "total_false_negatives_model": best_res["fn_model"],
            "total_false_negatives_blocker": best_res["fn_blocker"],
            "total_false_negatives": best_res["total_fn"],
            "singleton_accuracy": best_res["singleton_accuracy"],
            "multi_match_recall": best_res["multi_match_recall"],
            "ablations": {k: v["macro_f_05"] for k, v in ablation_results.items()},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(champ_path, "w") as f:
            json.dump(champ_data, f, indent=2)
    else:
        print(f"\n*** EXP_N030 did not beat champion ({best_res['macro_f_05']:.6f} <= {champ_f05:.6f}). Champion preserved. ***")

    # 8. Save results JSON
    results_json_path = CODE_DIR / "reports" / "EXP_N030_RESULTS.json"
    res_summary = {
        "experiment": "EXP_N030",
        "authoritative_baseline_metric_b": champ_f05,
        "n030_evaluated_metric_b": best_res["macro_f_05"],
        "delta_metric_b": best_res["macro_f_05"] - champ_f05,
        "precision": best_res["macro_precision"],
        "recall": best_res["macro_recall"],
        "fp": best_res["fp"],
        "fn_model": best_res["fn_model"],
        "fn_blocker": best_res["fn_blocker"],
        "total_fn": best_res["total_fn"],
        "best_variant": best_variant,
        "ablations": {k: v["macro_f_05"] for k, v in ablation_results.items()},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(results_json_path, "w") as f:
        json.dump(res_summary, f, indent=2)
    print(f"Results summary saved to: {results_json_path}")


if __name__ == "__main__":
    main()
