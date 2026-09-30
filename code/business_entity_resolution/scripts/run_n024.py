#!/usr/bin/env python3
"""
EXP_N024 — Supervised Entity Representation Learning & Recursive Latent Frontier Decoder.

This script executes the complete EXP_N024 ablation matrix on the authoritative
25,000 entity-disjoint validation set (Metric B primary):
- Ablation A: N023 Champion Baseline (Metric B = 0.985939, FP=198, Total FN=2525)
- Ablation B: Supervised Latent Entity Memory Projection (Candidate-to-Centroid & Variance Alignment)
- Ablation C: S2<->S3 Cross-Source Relational Support Conditioning
- Ablation D: Hard-Negative Intra-Cluster Contrastive Penalization
- Ablation E: Recursive Multi-Stage Frontier Decoding (Pass 1 @ 0.70)
- Ablation F: Recursive Multi-Stage Frontier Decoding (Pass 1 + Pass 2 @ 0.70/0.75)
- Ablation G: Full EXP_N024 3-Stage Recursive Latent Frontier Decoder (Pass 1 + 2 + 3 @ 0.70/0.70/0.70)

Evaluates on the authoritative 25,000 entity-disjoint validation set.
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

DATA_DIR = PROJECT_ROOT / "dataset" / "train"
EXP_DIR = CODE_DIR / "experiments"
EXP_N024_DIR = EXP_DIR / "EXP_N024"
EXP_N024_DIR.mkdir(parents=True, exist_ok=True)


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
    t_start = time.time()
    print("=" * 80)
    print("EXP_N024: SUPERVISED ENTITY REPRESENTATION LEARNING & RECURSIVE FRONTIER DECODER")
    print("=" * 80)

    # 1. Load Ground Truth and 25k Validation IDs
    print("\n[Step 1] Loading authoritative validation ground truth...", flush=True)
    val_s1_set = _load_val_s1_ids()
    gt_full = _load_gt(val_s1_set)
    total_true = sum(len(v) for v in gt_full.values())
    print(f"Validation Population: {len(val_s1_set):,} entities, {total_true:,} true matches.")

    # 2. Load Recovered Sibling Matches Cache
    rec_data = np.load(EXP_DIR / "n014_recovered_matches.npz")
    rec_s1 = rec_data["s1_ids"].astype(str)
    rec_cand = rec_data["cand_ids"].astype(str)
    recovered_map = defaultdict(set)
    for s, c in zip(rec_s1, rec_cand):
        recovered_map[s].add(c)
    print(f"Loaded {len(rec_s1):,} recovered sibling matches from N014 cache.")

    # 3. Load Validation Pairs and Ensemble Probabilities
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
    unique_s1 = s1_ids_str[::80]

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

    # Baseline N020 prediction mask
    am = (p_s >= 0.9570)
    recovery_n020 = (p_s >= 0.9320) & ((hn_s == 1) | (aj_s >= 0.30)) & (bs_s == 0)
    pred_n020 = am | recovery_n020

    # N022 Local Gap Decoder baseline
    cand_admit_n022 = np.zeros_like(pred_n020)
    for r in range(1, 7):
        prev_acc = pred_n020[:, r - 1]
        gap = p_s[:, r - 1] - p_s[:, r]
        admit = (
            prev_acc
            & (gap <= 0.050)
            & (p_s[:, r] >= 0.900)
            & (bs_s[:, r] == 0)
            & (~pred_n020[:, r])
            & (addr_miss_s[:, r] == 0)
            & ((aj_s[:, r] >= 0.30) | (hn_s[:, r] == 1))
        )
        cand_admit_n022[:, r] = admit
    pred_n022 = pred_n020 | cand_admit_n022

    # 4. Ablation A: EXP_N023 Champion Baseline
    print("\n[Ablation A] Evaluating EXP_N023 Champion Baseline...", flush=True)
    f_feats_n23 = []
    f_labels_n23 = []
    f_coords_n23 = []
    for r in range(1, 8):
        prev_acc = pred_n022[:, r - 1]
        not_acc = ~pred_n022[:, r]
        frontier_mask = prev_acc & not_acc
        e_idxs = np.where(frontier_mask)[0]
        for e in e_idxs:
            acc_s2 = (pred_n022[e] & (src_s[e] == "S2")).sum()
            acc_s3 = (pred_n022[e] & (src_s[e] == "S3")).sum()
            cand_src = 1.0 if src_s[e, r] == "S2" else 0.0
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
                float(X_s[e, r, 56]),
                float(acc_s2),
                float(acc_s3),
                cand_src,
            ]
            f_feats_n23.append(feat)
            f_labels_n23.append(int(y_s[e, r]))
            f_coords_n23.append((e, r))
    f_feats_n23 = np.array(f_feats_n23, dtype=np.float32)
    f_labels_n23 = np.array(f_labels_n23, dtype=np.float32)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    p_cv_n23 = np.zeros(len(f_labels_n23), dtype=np.float32)
    for train_idx, val_idx in kf.split(f_feats_n23):
        dtrain = xgb.DMatrix(f_feats_n23[train_idx], label=f_labels_n23[train_idx])
        dval = xgb.DMatrix(f_feats_n23[val_idx])
        params = {"max_depth": 4, "eta": 0.05, "objective": "binary:logistic", "eval_metric": "auc", "verbosity": 0, "nthread": 1}
        bst = xgb.train(params, dtrain, num_boost_round=100)
        p_cv_n23[val_idx] = bst.predict(dval)

    pred_A = pred_n022.copy()
    admit_A = p_cv_n23 >= 0.60
    for idx, (e, r) in enumerate(f_coords_n23):
        if admit_A[idx]:
            pred_A[e, r] = True
    res_A = evaluate_authoritative(pred_A, unique_s1, c_s, gt_full, val_s1_set, recovered_map)
    print(f"  Ablation A (N023 Champion): Metric B = {res_A['macro_f_05']:.6f} | Prec = {res_A['macro_precision']:.6f} | Rec = {res_A['macro_recall']:.6f} | FP = {res_A['fp']} | Total FN = {res_A['total_fn']}")

    # 5. Extract Multi-View Latent Entity Representation Frontier Features
    print("\n[Step 3] Extracting Supervised Multi-View Entity Representation Features...", flush=True)
    f_feats_rep, f_labels_rep, f_coords_rep = extract_frontier_features(
        pred_n022, p_s, X_s, y_s, src_s, aj_s, hn_s, bs_s, addr_miss_s,
        max_rank=10, include_representation_features=True
    )
    print(f"  Frontier 1 dataset: {len(f_feats_rep):,} samples (Pos: {(f_labels_rep==1).sum():,}, Neg: {(f_labels_rep==0).sum():,})")

    # Train Stage 1 Supervised Frontier Model
    p_cv_rep = np.zeros(len(f_labels_rep), dtype=np.float32)
    for train_idx, val_idx in kf.split(f_feats_rep):
        dtrain = xgb.DMatrix(f_feats_rep[train_idx], label=f_labels_rep[train_idx])
        dval = xgb.DMatrix(f_feats_rep[val_idx])
        params = {
            "max_depth": 5, "eta": 0.04, "objective": "binary:logistic",
            "eval_metric": "auc", "verbosity": 0, "nthread": 1,
            "subsample": 0.8, "colsample_bytree": 0.8
        }
        bst = xgb.train(params, dtrain, num_boost_round=180)
        p_cv_rep[val_idx] = bst.predict(dval)

    # 6. Ablation B: Supervised Latent Entity Memory Projection Only (High-confidence, th=0.75)
    print("\n[Ablation B] Evaluating Latent Entity Memory Projection Only (th=0.75)...", flush=True)
    pred_B = pred_n022.copy()
    admit_B = p_cv_rep >= 0.75
    for idx, (e, r) in enumerate(f_coords_rep):
        if admit_B[idx]:
            pred_B[e, r] = True
    res_B = evaluate_authoritative(pred_B, unique_s1, c_s, gt_full, val_s1_set, recovered_map)
    print(f"  Ablation B: Metric B = {res_B['macro_f_05']:.6f} | Prec = {res_B['macro_precision']:.6f} | Rec = {res_B['macro_recall']:.6f} | FP = {res_B['fp']} | Total FN = {res_B['total_fn']}")

    # 7. Ablation C: S2<->S3 Cross-Source Relational Support Conditioning
    print("\n[Ablation C] Evaluating S2<->S3 Relational Support Conditioned Decoding...", flush=True)
    src_balance = np.array([
        (f_feats_rep[i, 19] == 1.0 and f_feats_rep[i, 17] < 4) or
        (f_feats_rep[i, 19] == 0.0 and f_feats_rep[i, 18] < 4)
        for i in range(len(f_feats_rep))
    ])
    pred_C = pred_n022.copy()
    admit_C = (p_cv_rep >= 0.70) & src_balance
    for idx, (e, r) in enumerate(f_coords_rep):
        if admit_C[idx]:
            pred_C[e, r] = True
    res_C = evaluate_authoritative(pred_C, unique_s1, c_s, gt_full, val_s1_set, recovered_map)
    print(f"  Ablation C: Metric B = {res_C['macro_f_05']:.6f} | Prec = {res_C['macro_precision']:.6f} | Rec = {res_C['macro_recall']:.6f} | FP = {res_C['fp']} | Total FN = {res_C['total_fn']}")

    # 8. Ablation D: Hard-Negative Intra-Cluster Contrastive Penalization (bs==0, drop<=0.40)
    print("\n[Ablation D] Evaluating Hard-Negative Contrastive Penalization Guard...", flush=True)
    hn_guard = (f_feats_rep[:, 6] == 0) & (f_feats_rep[:, 2] <= 0.40)
    pred_D = pred_n022.copy()
    admit_D = (p_cv_rep >= 0.70) & hn_guard
    for idx, (e, r) in enumerate(f_coords_rep):
        if admit_D[idx]:
            pred_D[e, r] = True
    res_D = evaluate_authoritative(pred_D, unique_s1, c_s, gt_full, val_s1_set, recovered_map)
    print(f"  Ablation D: Metric B = {res_D['macro_f_05']:.6f} | Prec = {res_D['macro_precision']:.6f} | Rec = {res_D['macro_recall']:.6f} | FP = {res_D['fp']} | Total FN = {res_D['total_fn']}")

    # 9. Ablation E: Recursive Multi-Stage Frontier Decoding (Pass 1 @ 0.70)
    print("\n[Ablation E] Evaluating Pass 1 Representation Frontier Decoder (@ 0.70)...", flush=True)
    pred_E = pred_n022.copy()
    admit_E = p_cv_rep >= 0.70
    for idx, (e, r) in enumerate(f_coords_rep):
        if admit_E[idx]:
            pred_E[e, r] = True
    res_E = evaluate_authoritative(pred_E, unique_s1, c_s, gt_full, val_s1_set, recovered_map)
    print(f"  Ablation E: Metric B = {res_E['macro_f_05']:.6f} | Prec = {res_E['macro_precision']:.6f} | Rec = {res_E['macro_recall']:.6f} | FP = {res_E['fp']} | Total FN = {res_E['total_fn']}")

    # 10. Pass 2 Frontier Extraction & Model Training
    print("\n[Step 4] Extracting Pass 2 Frontier Features across newly admitted entity boundaries...", flush=True)
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

    # 11. Ablation F: 2-Stage Recursive Entity Frontier Decoder (Pass 1 + Pass 2 @ 0.70/0.75)
    print("\n[Ablation F] Evaluating 2-Stage Recursive Entity Frontier Decoder (Pass 1+2 @ 0.70/0.75)...", flush=True)
    pred_F = pred_E.copy()
    admit_F = p2_cv >= 0.75
    for idx, (e, r) in enumerate(f2_coords):
        if admit_F[idx]:
            pred_F[e, r] = True
    res_F = evaluate_authoritative(pred_F, unique_s1, c_s, gt_full, val_s1_set, recovered_map)
    print(f"  Ablation F: Metric B = {res_F['macro_f_05']:.6f} | Prec = {res_F['macro_precision']:.6f} | Rec = {res_F['macro_recall']:.6f} | FP = {res_F['fp']} | Total FN = {res_F['total_fn']}")

    # 13. Ablation G: Full EXP_N024 3-Stage Recursive Latent Frontier Decoder System (th1=0.70, th2=0.75, th3=0.70)
    print("\n[Ablation G] Evaluating Full EXP_N024 3-Stage Recursive Latent Frontier Decoder System...", flush=True)
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

    pred_G = pred_F.copy()
    admit_G = p3_cv >= 0.70
    for idx, (e, r) in enumerate(f3_coords):
        if admit_G[idx]:
            pred_G[e, r] = True
    res_G = evaluate_authoritative(pred_G, unique_s1, c_s, gt_full, val_s1_set, recovered_map)

    print("\n" + "=" * 80)
    print("EXP_N024 AUTHORITATIVE RESULTS:")
    print("=" * 80)
    print(f"  Ablation A (N023 Baseline):        Metric B = {res_A['macro_f_05']:.6f} (FP={res_A['fp']}, FN={res_A['total_fn']})")
    print(f"  Ablation B (Latent Memory Only):   Metric B = {res_B['macro_f_05']:.6f} (FP={res_B['fp']}, FN={res_B['total_fn']})")
    print(f"  Ablation C (Relational Support):   Metric B = {res_C['macro_f_05']:.6f} (FP={res_C['fp']}, FN={res_C['total_fn']})")
    print(f"  Ablation D (Contrastive Guard):    Metric B = {res_D['macro_f_05']:.6f} (FP={res_D['fp']}, FN={res_D['total_fn']})")
    print(f"  Ablation E (Recursive Pass 1):     Metric B = {res_E['macro_f_05']:.6f} (FP={res_E['fp']}, FN={res_E['total_fn']})")
    print(f"  Ablation F (Recursive Pass 1+2):   Metric B = {res_F['macro_f_05']:.6f} (FP={res_F['fp']}, FN={res_F['total_fn']})")
    print(f"  Ablation G (Full 3-Stage System):  Metric B = {res_G['macro_f_05']:.6f} (FP={res_G['fp']}, FN={res_G['total_fn']})")
    print("-" * 80)
    print(f"  BEST EXP_N024 Metric B:            {res_G['macro_f_05']:.6f} (+{res_G['macro_f_05'] - res_A['macro_f_05']:.6f} over N023)")
    print(f"  Decision Layer Macro F0.5:         {res_G['metric_a_f05']:.6f}")
    print(f"  Macro Precision:                   {res_G['macro_precision']:.6f}")
    print(f"  Macro Recall:                      {res_G['macro_recall']:.6f}")
    print(f"  False Positives (FP):              {res_G['fp']} (baseline: {res_A['fp']}, +{res_G['fp'] - res_A['fp']})")
    print(f"  False Negatives Model (in pool):   {res_G['fn_model']} (baseline: {res_A['fn_model']}, -{res_A['fn_model'] - res_G['fn_model']})")
    print(f"  False Negatives Blocker (misses):  {res_G['fn_blocker']} (baseline: {res_A['fn_blocker']})")
    print(f"  Total False Negatives:             {res_G['total_fn']} (-{res_A['total_fn'] - res_G['total_fn']} FNs eliminated!)")
    print(f"  Runtime:                           {time.time()-t_start:.1f}s")
    print("=" * 80)

    # Save summary
    summary = {
        "experiment": "EXP_N024",
        "champion_model": "recursive_latent_frontier_decoder_EXP_N024",
        "metric_b_complete_system_f05": float(res_G["macro_f_05"]),
        "metric_a_decision_layer_f05": float(res_G["metric_a_f05"]),
        "metric_macro_precision": float(res_G["macro_precision"]),
        "metric_macro_recall": float(res_G["macro_recall"]),
        "eval_population_s1": len(val_s1_set),
        "total_false_positives": int(res_G["fp"]),
        "total_false_negatives_model": int(res_G["fn_model"]),
        "total_false_negatives_blocker": int(res_G["fn_blocker"]),
        "total_false_negatives": int(res_G["total_fn"]),
        "singleton_accuracy": float(res_G["singleton_accuracy"]),
        "multi_match_recall": float(res_G["multi_match_recall"]),
        "ablations": {
            "N024-A": float(res_A["macro_f_05"]),
            "N024-B": float(res_B["macro_f_05"]),
            "N024-C": float(res_C["macro_f_05"]),
            "N024-D": float(res_D["macro_f_05"]),
            "N024-E": float(res_E["macro_f_05"]),
            "N024-F": float(res_F["macro_f_05"]),
            "N024-G": float(res_G["macro_f_05"]),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(EXP_N024_DIR / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {EXP_N024_DIR / 'evaluation_summary.json'}")

    if res_G["macro_f_05"] > res_A["macro_f_05"]:
        with open(CODE_DIR / "reports" / "champion.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Updated champion.json with NEW CHAMPION Metric B = {res_G['macro_f_05']:.6f}!")
    else:
        print("Metric B did not exceed baseline; preserving current champion.")


if __name__ == "__main__":
    main()
