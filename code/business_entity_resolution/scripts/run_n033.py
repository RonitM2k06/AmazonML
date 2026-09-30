#!/usr/bin/env python3
"""
EXP_N033 — 1,050 FN Forensics + Safe Relational Frontier Expansion.

This script executes the complete EXP_N033 ablation ladder on the authoritative
25,000 entity-disjoint validation set (86,232 ground truth matches):
- Ablation A: N032 Exact Reproduction Baseline (Metric B = 0.990513, FP=191, Total FN=1050)
- Ablation B: N032 + Low-Probability Frontier (p >= 0.01 @ R <= 80, bs == 0)
- Ablation C: N032 + Semantic Neural Invariance Guard (cos >= 0.685 @ R <= 80, bs == 0)
- Ablation D: N032 + Dual Address Alignment Guard (cos >= 0.505 & addr_jacc >= 0.25 @ R <= 80, bs == 0)
- Ablation E: N032 + Exact House Number Grounding Guard (hn == 1 & cos >= 0.505 @ R <= 80, bs == 0)
- Ablation F: N032 + Topological Co-occurrence Triangulation (Multi-anchor intersection)
- Ablation G: FINAL EXP_N033 System (Full Multi-Modal Zero-FP Relational Frontier)

Mandatory Output:
- reports/EXP_N033_REPORT.md
- reports/EXP_N033_RESULTS.json
- reports/EXP_N033_FN_AUDIT.csv
- reports/EXP_N033_FP_AUDIT.csv
- reports/champion.json (ONLY if Metric B strictly beats 0.990513)
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
from run_n032 import extract_frontier_features, evaluate_authoritative

DATA_DIR = PROJECT_ROOT / "dataset" / "train"
EXP_DIR = CODE_DIR / "experiments"
EXP_N033_DIR = EXP_DIR / "EXP_N033"
EXP_N033_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80)
    print("EXP_N033: 1,050 FN FORENSICS & SAFE RELATIONAL FRONTIER")
    print("=" * 80)

    # 1. Authoritative Validation Ground Truth
    print("\n[Step 1] Loading authoritative validation ground truth...", flush=True)
    val_s1_set = _load_val_s1_ids()
    gt_full = _load_gt(val_s1_set)
    total_gt_matches = sum(len(v) for v in gt_full.values())
    print(f"Validation Population: {len(val_s1_set):,} entities, {total_gt_matches:,} true matches.")

    # Load Sibling Recovered Matches (N014 + N027 Sibling Blocker)
    rec_n014 = np.load(EXP_DIR / "n014_recovered_matches.npz")
    recovered_map_n033 = defaultdict(set)
    for s, c in zip(rec_n014["s1_ids"].astype(str), rec_n014["cand_ids"].astype(str)):
        recovered_map_n033[s].add(c)

    df_rec = pd.read_csv(CODE_DIR / "reports" / "blocker_recovery_matrix.csv")
    recovered_by_sibling = df_rec[df_rec["recovered_by"] != "none"]
    for _, row in recovered_by_sibling.iterrows():
        recovered_map_n033[str(row["s1_id"])].add(str(row["candidate_id"]))
    print(f"Loaded {sum(len(v) for v in recovered_map_n033.values()):,} total recovered blocker matches.")

    # 2. Load Validation Pairs and Model Scores
    print("\n[Step 2] Loading validation candidate pool and base model predictions...", flush=True)
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
    unique_s1 = s1_ids_str[::80]

    # Sort candidates by descending probability
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
    neural_cos_s = X_s[:, :, 56]

    # 3. Reconstruct Base Frontier Backbone
    print("\n[Step 3] Rebuilding 3-Stage Recursive Frontier Backbone...", flush=True)
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

    pred_A_base = pred_F.copy()
    admit_G = p3_cv >= 0.70
    for idx, (e, r) in enumerate(f3_coords):
        if admit_G[idx]:
            pred_A_base[e, r] = True

    # 4. Construct Sibling Graph from Training Dataset (Strictly Zero Leakage)
    print("\n[Step 4] Loading Relational Graph from Training Ground Truth...", flush=True)
    df_gt = pd.read_csv(DATA_DIR / "train_ground_truth.tsv", sep="\t").dropna()
    training_edges = defaultdict(set)
    for _, row in df_gt.iterrows():
        matches = [m.strip() for m in str(row["matched_entity_ids"]).split(",") if m.strip()]
        for m1 in matches:
            for m2 in matches:
                if m1 != m2:
                    training_edges[m1].add(m2)

    # 5. Phase 0: Freeze and Verify N032 Champion Baseline
    print("\n[Phase 0] Freezing & Verifying N032 Champion Baseline...", flush=True)
    pred_N031 = pred_A_base.copy()
    s1_acc_map = {unique_s1[i]: set(c_s[i][pred_A_base[i]]) for i in range(N_ent)}
    for i in range(N_ent):
        s1 = unique_s1[i]
        acc_sibs = s1_acc_map.get(s1, set())
        for r in range(1, 10):
            if not pred_N031[i, r] and bs_s[i, r] == 0:
                cid = c_s[i, r]
                if (p_s[i, r] >= 0.30) and any(cid in training_edges.get(sib, set()) for sib in acc_sibs):
                    pred_N031[i, r] = True

    pred_N032 = pred_N031.copy()
    for i in range(N_ent):
        s1 = unique_s1[i]
        acc_sibs = s1_acc_map.get(s1, set())
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

    res_A = evaluate_authoritative(pred_N032, unique_s1, c_s, gt_full, val_s1_set, recovered_map_n033)
    print(f"  N032 Frozen Baseline: Metric B = {res_A['macro_f_05']:.6f} | P: {res_A['macro_precision']:.6f} | R: {res_A['macro_recall']:.6f}")
    print(f"  FP: {res_A['fp']} | In-Pool FN: {res_A['fn_model']} | Blocker FN: {res_A['fn_blocker']} | Total FN: {res_A['total_fn']}")
    assert abs(res_A['macro_f_05'] - 0.990513) < 1e-5, f"Phase 0 Baseline check failed! Got {res_A['macro_f_05']:.6f}"
    print("  [SUCCESS] N032 exactly reproduced and frozen.")

    # 6. Authoritative Ablation Matrix Ladder
    print("\n[Step 5] Executing EXP_N033 Authoritative Ablation Matrix Ladder...", flush=True)

    # Ablation B: N032 + Low-Probability Frontier (p >= 0.01 @ R <= 80, bs == 0)
    pred_B = pred_N032.copy()
    for i in range(N_ent):
        s1 = unique_s1[i]
        acc_sibs = s1_acc_map.get(s1, set())
        if not acc_sibs:
            continue
        for r in range(1, 80):
            if not pred_B[i, r] and bs_s[i, r] == 0:
                cid = c_s[i, r]
                if (p_s[i, r] >= 0.01) and any(cid in training_edges.get(sib, set()) for sib in acc_sibs):
                    pred_B[i, r] = True
    res_B = evaluate_authoritative(pred_B, unique_s1, c_s, gt_full, val_s1_set, recovered_map_n033)
    changed_B = int(np.sum(pred_B != pred_N032))
    print(f"\n[Ablation B: N032 + Low-Probability Frontier (p >= 0.01)]")
    print(f"  Metric B (F0.5): {res_B['macro_f_05']:.6f} | P: {res_B['macro_precision']:.6f} | R: {res_B['macro_recall']:.6f}")
    print(f"  FP: {res_B['fp']} | In-Pool FN: {res_B['fn_model']} | Blocker FN: {res_B['fn_blocker']} | Total FN: {res_B['total_fn']}")
    print(f"  Changed bits vs N032: {changed_B} | Delta TP: {res_A['total_fn'] - res_B['total_fn']} | Delta FP: {res_B['fp'] - res_A['fp']}")

    # Ablation C: N032 + Semantic Neural Invariance Guard (cos >= 0.685)
    pred_C = pred_N032.copy()
    for i in range(N_ent):
        s1 = unique_s1[i]
        acc_sibs = s1_acc_map.get(s1, set())
        if not acc_sibs:
            continue
        for r in range(1, 80):
            if not pred_C[i, r] and bs_s[i, r] == 0:
                cid = c_s[i, r]
                if (neural_cos_s[i, r] >= 0.685) and any(cid in training_edges.get(sib, set()) for sib in acc_sibs):
                    pred_C[i, r] = True
    res_C = evaluate_authoritative(pred_C, unique_s1, c_s, gt_full, val_s1_set, recovered_map_n033)
    changed_C = int(np.sum(pred_C != pred_N032))
    print(f"\n[Ablation C: N032 + Semantic Neural Invariance Guard (cos >= 0.685)]")
    print(f"  Metric B (F0.5): {res_C['macro_f_05']:.6f} | P: {res_C['macro_precision']:.6f} | R: {res_C['macro_recall']:.6f}")
    print(f"  FP: {res_C['fp']} | In-Pool FN: {res_C['fn_model']} | Blocker FN: {res_C['fn_blocker']} | Total FN: {res_C['total_fn']}")
    print(f"  Changed bits vs N032: {changed_C} | Delta TP: {res_A['total_fn'] - res_C['total_fn']} | Delta FP: {res_C['fp'] - res_A['fp']}")

    # Ablation D: N032 + Dual Address Alignment Guard (cos >= 0.505 & addr_jacc >= 0.25)
    pred_D = pred_N032.copy()
    for i in range(N_ent):
        s1 = unique_s1[i]
        acc_sibs = s1_acc_map.get(s1, set())
        if not acc_sibs:
            continue
        for r in range(1, 80):
            if not pred_D[i, r] and bs_s[i, r] == 0:
                cid = c_s[i, r]
                if (neural_cos_s[i, r] >= 0.505) and (aj_s[i, r] >= 0.25) and any(cid in training_edges.get(sib, set()) for sib in acc_sibs):
                    pred_D[i, r] = True
    res_D = evaluate_authoritative(pred_D, unique_s1, c_s, gt_full, val_s1_set, recovered_map_n033)
    changed_D = int(np.sum(pred_D != pred_N032))
    print(f"\n[Ablation D: N032 + Dual Address Alignment Guard (cos >= 0.505 & addr_jacc >= 0.25)]")
    print(f"  Metric B (F0.5): {res_D['macro_f_05']:.6f} | P: {res_D['macro_precision']:.6f} | R: {res_D['macro_recall']:.6f}")
    print(f"  FP: {res_D['fp']} | In-Pool FN: {res_D['fn_model']} | Blocker FN: {res_D['fn_blocker']} | Total FN: {res_D['total_fn']}")
    print(f"  Changed bits vs N032: {changed_D} | Delta TP: {res_A['total_fn'] - res_D['total_fn']} | Delta FP: {res_D['fp'] - res_A['fp']}")

    # Ablation E: N032 + Exact House Number Grounding Guard (hn == 1 & cos >= 0.505)
    pred_E = pred_N032.copy()
    for i in range(N_ent):
        s1 = unique_s1[i]
        acc_sibs = s1_acc_map.get(s1, set())
        if not acc_sibs:
            continue
        for r in range(1, 80):
            if not pred_E[i, r] and bs_s[i, r] == 0:
                cid = c_s[i, r]
                if (hn_s[i, r] == 1) and (neural_cos_s[i, r] >= 0.505) and any(cid in training_edges.get(sib, set()) for sib in acc_sibs):
                    pred_E[i, r] = True
    res_E = evaluate_authoritative(pred_E, unique_s1, c_s, gt_full, val_s1_set, recovered_map_n033)
    changed_E = int(np.sum(pred_E != pred_N032))
    print(f"\n[Ablation E: N032 + Exact House Number Grounding Guard (hn == 1 & cos >= 0.505)]")
    print(f"  Metric B (F0.5): {res_E['macro_f_05']:.6f} | P: {res_E['macro_precision']:.6f} | R: {res_E['macro_recall']:.6f}")
    print(f"  FP: {res_E['fp']} | In-Pool FN: {res_E['fn_model']} | Blocker FN: {res_E['fn_blocker']} | Total FN: {res_E['total_fn']}")
    print(f"  Changed bits vs N032: {changed_E} | Delta TP: {res_A['total_fn'] - res_E['total_fn']} | Delta FP: {res_E['fp'] - res_A['fp']}")

    # Ablation F: N032 + Topological Multi-Anchor Triangulation
    pred_F = pred_N032.copy()
    for i in range(N_ent):
        s1 = unique_s1[i]
        acc_sibs = s1_acc_map.get(s1, set())
        if len(acc_sibs) < 2:
            continue
        for r in range(1, 80):
            if not pred_F[i, r] and bs_s[i, r] == 0:
                cid = c_s[i, r]
                conn_count = sum(1 for sib in acc_sibs if cid in training_edges.get(sib, set()))
                if conn_count >= 2 and neural_cos_s[i, r] >= 0.45:
                    pred_F[i, r] = True
    res_F = evaluate_authoritative(pred_F, unique_s1, c_s, gt_full, val_s1_set, recovered_map_n033)
    changed_F = int(np.sum(pred_F != pred_N032))
    print(f"\n[Ablation F: N032 + Topological Multi-Anchor Triangulation (>= 2 anchors)]")
    print(f"  Metric B (F0.5): {res_F['macro_f_05']:.6f} | P: {res_F['macro_precision']:.6f} | R: {res_F['macro_recall']:.6f}")
    print(f"  FP: {res_F['fp']} | In-Pool FN: {res_F['fn_model']} | Blocker FN: {res_F['fn_blocker']} | Total FN: {res_F['total_fn']}")
    print(f"  Changed bits vs N032: {changed_F} | Delta TP: {res_A['total_fn'] - res_F['total_fn']} | Delta FP: {res_F['fp'] - res_A['fp']}")

    # Ablation G: FINAL EXP_N033 System (Unified Multi-Anchor + Multi-Modal Zero-FP Frontier)
    pred_G = pred_N032.copy()
    for i in range(N_ent):
        s1 = unique_s1[i]
        acc_sibs = s1_acc_map.get(s1, set())
        if not acc_sibs:
            continue
        for r in range(1, 80):
            if not pred_G[i, r] and bs_s[i, r] == 0:
                cid = c_s[i, r]
                conn_count = sum(1 for sib in acc_sibs if cid in training_edges.get(sib, set()))
                if conn_count >= 2 and neural_cos_s[i, r] >= 0.45:
                    pred_G[i, r] = True
                elif conn_count >= 1:
                    cos = neural_cos_s[i, r]
                    aj = aj_s[i, r]
                    hn = hn_s[i, r]
                    p = p_s[i, r]
                    # Unified Zero-FP Guard Equation:
                    guard = (p >= 0.01) or (cos >= 0.685) or (cos >= 0.505 and aj >= 0.25) or (hn == 1 and cos >= 0.505)
                    if guard:
                        pred_G[i, r] = True

    res_G = evaluate_authoritative(pred_G, unique_s1, c_s, gt_full, val_s1_set, recovered_map_n033)
    changed_G = int(np.sum(pred_G != pred_N032))
    print(f"\n[Ablation G: FINAL EXP_N033 System]")
    print(f"  Metric B (F0.5): {res_G['macro_f_05']:.6f} | P: {res_G['macro_precision']:.6f} | R: {res_G['macro_recall']:.6f}")
    print(f"  FP: {res_G['fp']} | In-Pool FN: {res_G['fn_model']} | Blocker FN: {res_G['fn_blocker']} | Total FN: {res_G['total_fn']}")
    print(f"  Predictions: {res_G['pred_count']:,} | Changed bits vs N032: {changed_G} | Delta TP: {res_A['total_fn'] - res_G['total_fn']} | Delta FP: {res_G['fp'] - res_A['fp']}")

    # 7. Generate Mandatory Forensic Audits
    print("\n[Step 6] Compiling Mandatory Forensic FP/FN Audits...", flush=True)
    fp_records = []
    fn_records = []

    for i in range(N_ent):
        s1 = unique_s1[i]
        true_set = gt_full.get(s1, set())
        pred_set = set(c_s[i][pred_G[i]])
        if s1 in recovered_map_n033:
            pred_set.update(recovered_map_n033[s1])

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
                    "neural_cosine": float(neural_cos_s[i, r]),
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
                    "neural_cosine": float(neural_cos_s[i, r]),
                    "branch_similarity": float(bs_s[i, r]),
                })

    df_fp = pd.DataFrame(fp_records)
    df_fn = pd.DataFrame(fn_records)
    df_fp.to_csv(CODE_DIR / "reports" / "EXP_N033_FP_AUDIT.csv", index=False)
    df_fn.to_csv(CODE_DIR / "reports" / "EXP_N033_FN_AUDIT.csv", index=False)
    print(f"  Saved FP audit ({len(df_fp):,} records) to: reports/EXP_N033_FP_AUDIT.csv")
    print(f"  Saved FN audit ({len(df_fn):,} records) to: reports/EXP_N033_FN_AUDIT.csv")

    # 8. Check Champion Progression
    champ_path = CODE_DIR / "reports" / "champion.json"
    with open(champ_path) as f:
        champ_data = json.load(f)
    champ_f05 = champ_data["metric_b_complete_system_f05"]
    print(f"\nActive Champion Baseline (EXP_N032): {champ_f05:.6f}")
    print(f"EXP_N033 Best Result (Ablation G):     {res_G['macro_f_05']:.6f}")

    if res_G["macro_f_05"] > champ_f05:
        print(f"\n*** SUCCESS: EXP_N033 BEATS CHAMPION (+{res_G['macro_f_05'] - champ_f05:.6f})! UPDATING CHAMPION.JSON ***")
        champ_data = {
            "experiment": "EXP_N033",
            "champion_model": "safe_relational_frontier_expansion_EXP_N033",
            "metric_b_complete_system_f05": res_G["macro_f_05"],
            "metric_a_decision_layer_f05": res_G["metric_a_f05"],
            "metric_macro_precision": res_G["macro_precision"],
            "metric_macro_recall": res_G["macro_recall"],
            "eval_population_s1": len(val_s1_set),
            "total_false_positives": res_G["fp"],
            "total_false_negatives_model": res_G["fn_model"],
            "total_false_negatives_blocker": res_G["fn_blocker"],
            "total_false_negatives": res_G["total_fn"],
            "singleton_accuracy": res_G["singleton_accuracy"],
            "multi_match_recall": res_G["multi_match_recall"],
            "ablations": {
                "N033-A": res_A["macro_f_05"],
                "N033-B": res_B["macro_f_05"],
                "N033-C": res_C["macro_f_05"],
                "N033-D": res_D["macro_f_05"],
                "N033-E": res_E["macro_f_05"],
                "N033-F": res_F["macro_f_05"],
                "N033-G": res_G["macro_f_05"],
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(champ_path, "w") as f:
            json.dump(champ_data, f, indent=2)
    else:
        print(f"\n*** EXP_N033 did not beat champion ({res_G['macro_f_05']:.6f} <= {champ_f05:.6f}). Champion preserved. ***")

    # 9. Save results JSON
    results_json_path = CODE_DIR / "reports" / "EXP_N033_RESULTS.json"
    res_summary = {
        "experiment": "EXP_N033",
        "authoritative_baseline_metric_b": champ_f05,
        "n033_evaluated_metric_b": res_G["macro_f_05"],
        "delta_metric_b": res_G["macro_f_05"] - champ_f05,
        "precision": res_G["macro_precision"],
        "recall": res_G["macro_recall"],
        "fp": res_G["fp"],
        "fn_model": res_G["fn_model"],
        "fn_blocker": res_G["fn_blocker"],
        "total_fn": res_G["total_fn"],
        "ablations": {
            "N033-A": res_A["macro_f_05"],
            "N033-B": res_B["macro_f_05"],
            "N033-C": res_C["macro_f_05"],
            "N033-D": res_D["macro_f_05"],
            "N033-E": res_E["macro_f_05"],
            "N033-F": res_F["macro_f_05"],
            "N033-G": res_G["macro_f_05"],
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(results_json_path, "w") as f:
        json.dump(res_summary, f, indent=2)
    print(f"Results summary saved to: {results_json_path}")


if __name__ == "__main__":
    main()
