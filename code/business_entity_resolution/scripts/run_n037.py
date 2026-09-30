#!/usr/bin/env python3
"""
EXP_N037 — Reciprocal Cross-Source Entity Induction.

Executes the complete 10-variant authoritative ablation matrix (A through J)
on the canonical 25,000 entity-disjoint validation set (86,232 ground truth matches):
- Ablation A: N036 Exact Baseline Reproduction (Metric B = 0.992932, FP=191, In-pool FN=193)
- Ablation B: S2<->S3 compatibility only
- Ablation C: Dual S1 support
- Ablation D: Reciprocal top-k
- Ablation E: Reciprocal + address
- Ablation F: Reciprocal + neural
- Ablation G: Reciprocal + house/postal
- Ablation H: Reciprocal + contradiction guard
- Ablation I: Full reciprocal entity induction (N037 Unified Engine)
- Ablation J: FINAL EXP_N037 Unified System

Mandatory Deliverables:
- experiments/EXP_N037/
- reports/EXP_N037_REPORT.md
- reports/EXP_N037_RESULTS.json
- reports/EXP_N037_FN_AUDIT.csv
- reports/EXP_N037_FP_AUDIT.csv
- reports/EXP_N037_RECIPROCAL_AUDIT.csv
- reports/champion.json (ONLY if Metric B strictly beats 0.992932)
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
from run_n032 import evaluate_authoritative
from test_time_reciprocal_entity import TestTimeReciprocalEntityInducer

DATA_DIR = PROJECT_ROOT / "dataset" / "train"
EXP_DIR = CODE_DIR / "experiments"
EXP_N037_DIR = EXP_DIR / "EXP_N037"
EXP_N037_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80)
    print("EXP_N037: RECIPROCAL CROSS-SOURCE ENTITY INDUCTION")
    print("=" * 80)

    # 1. Authoritative Validation Ground Truth
    print("\n[Step 1] Loading authoritative validation ground truth...", flush=True)
    val_s1_set = _load_val_s1_ids()
    gt_full = _load_gt(val_s1_set)
    total_gt_matches = sum(len(v) for v in gt_full.values())
    print(f"Validation Population: {len(val_s1_set):,} entities, {total_gt_matches:,} true matches.")

    # Load Sibling Recovered Matches (N014 + N027 Sibling Blocker)
    rec_n014 = np.load(EXP_DIR / "n014_recovered_matches.npz")
    recovered_map_n037 = defaultdict(set)
    for s, c in zip(rec_n014["s1_ids"].astype(str), rec_n014["cand_ids"].astype(str)):
        recovered_map_n037[s].add(c)

    df_rec = pd.read_csv(CODE_DIR / "reports" / "blocker_recovery_matrix.csv")
    recovered_by_sibling = df_rec[df_rec["recovered_by"] != "none"]
    for _, row in recovered_by_sibling.iterrows():
        recovered_map_n037[str(row["s1_id"])].add(str(row["candidate_id"]))
    print(f"Loaded {sum(len(v) for v in recovered_map_n037.values()):,} total recovered blocker matches.")

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
    postal_s = X_s[:, :, 41]
    neural_cos_s = X_s[:, :, 56]

    # 3. Construct Sibling Graph from Training Dataset (Strictly Zero Leakage)
    print("\n[Step 3] Loading Relational Graph from Training Ground Truth...", flush=True)
    df_gt = pd.read_csv(DATA_DIR / "train_ground_truth.tsv", sep="\t").dropna()
    training_edges = defaultdict(set)
    for _, row in df_gt.iterrows():
        matches = [m.strip() for m in str(row["matched_entity_ids"]).split(",") if m.strip()]
        for m1 in matches:
            for m2 in matches:
                if m1 != m2:
                    training_edges[m1].add(m2)

    # 4. Phase 0: Freeze and Verify N036 Champion Baseline
    print("\n[Phase 0] Freezing & Verifying N036 Champion Baseline...", flush=True)
    pred_data = np.load(EXP_DIR / "pred_n036_matrix.npz")
    pred_N036 = pred_data["pred"]

    res_A = evaluate_authoritative(pred_N036, unique_s1, c_s, gt_full, val_s1_set, recovered_map_n037)
    print(f"  N036 Frozen Baseline: Metric B = {res_A['macro_f_05']:.6f} | P: {res_A['macro_precision']:.6f} | R: {res_A['macro_recall']:.6f}")
    print(f"  FP: {res_A['fp']} | In-Pool FN: {res_A['fn_model']} | Blocker FN: {res_A['fn_blocker']} | Total FN: {res_A['total_fn']}")
    assert abs(res_A["macro_f_05"] - 0.992932) < 1e-5, f"Phase 0 Baseline check failed! Got {res_A['macro_f_05']:.6f}"
    print("  [SUCCESS] N036 exactly reproduced and frozen.")

    unanchored_mask = (np.sum(pred_N036, axis=1) == 0)
    unanchored_indices = np.where(unanchored_mask)[0]
    print(f"  Unanchored S1 Entities in N036: {len(unanchored_indices):,}")

    # 5. Authoritative Ablation Matrix Ladder (A through J)
    print("\n[Step 4] Executing EXP_N037 Authoritative Ablation Matrix Ladder (A-J)...", flush=True)

    # Ablation B: S2<->S3 compatibility only (no multi-modal guards)
    pred_B = pred_N036.copy()
    for i in unanchored_indices:
        cands_i = c_s[i, :5]
        srcs_i = src_s[i, :5]
        s2_list = [(r, cands_i[r]) for r in range(len(cands_i)) if srcs_i[r] == "S2"]
        s3_list = [(r, cands_i[r]) for r in range(len(cands_i)) if srcs_i[r] == "S3"]
        for r2, c2 in s2_list:
            for r3, c3 in s3_list:
                if c3 in training_edges.get(c2, set()):
                    pred_B[i, r2] = True
                    pred_B[i, r3] = True
    res_B = evaluate_authoritative(pred_B, unique_s1, c_s, gt_full, val_s1_set, recovered_map_n037)
    changed_B = int(np.sum(pred_B != pred_N036))
    print(f"\n[Ablation B: S2<->S3 Compatibility Only]")
    print(f"  Metric B (F0.5): {res_B['macro_f_05']:.6f} | P: {res_B['macro_precision']:.6f} | R: {res_B['macro_recall']:.6f}")
    print(f"  FP: {res_B['fp']} | In-Pool FN: {res_B['fn_model']} | Total FN: {res_B['total_fn']}")
    print(f"  Changed bits: {changed_B} | Delta TP: {res_A['total_fn'] - res_B['total_fn']} | Delta FP: {res_B['fp'] - res_A['fp']}")

    # Ablation C: Dual S1 support (p_min >= 0.05)
    pred_C = pred_N036.copy()
    for i in unanchored_indices:
        cands_i = c_s[i, :5]
        srcs_i = src_s[i, :5]
        s2_list = [(r, cands_i[r]) for r in range(len(cands_i)) if srcs_i[r] == "S2"]
        s3_list = [(r, cands_i[r]) for r in range(len(cands_i)) if srcs_i[r] == "S3"]
        for r2, c2 in s2_list:
            for r3, c3 in s3_list:
                if c3 in training_edges.get(c2, set()):
                    if p_s[i, r2] >= 0.05 and p_s[i, r3] >= 0.05:
                        pred_C[i, r2] = True
                        pred_C[i, r3] = True
    res_C = evaluate_authoritative(pred_C, unique_s1, c_s, gt_full, val_s1_set, recovered_map_n037)
    changed_C = int(np.sum(pred_C != pred_N036))
    print(f"\n[Ablation C: Dual S1 Support (p_min >= 0.05)]")
    print(f"  Metric B (F0.5): {res_C['macro_f_05']:.6f} | P: {res_C['macro_precision']:.6f} | R: {res_C['macro_recall']:.6f}")
    print(f"  FP: {res_C['fp']} | In-Pool FN: {res_C['fn_model']} | Total FN: {res_C['total_fn']}")
    print(f"  Changed bits: {changed_C} | Delta TP: {res_A['total_fn'] - res_C['total_fn']} | Delta FP: {res_C['fp'] - res_A['fp']}")

    # Ablation D: Reciprocal top-k (K=3)
    pred_D = pred_N036.copy()
    for i in unanchored_indices:
        cands_i = c_s[i, :3]
        srcs_i = src_s[i, :3]
        s2_list = [(r, cands_i[r]) for r in range(len(cands_i)) if srcs_i[r] == "S2"]
        s3_list = [(r, cands_i[r]) for r in range(len(cands_i)) if srcs_i[r] == "S3"]
        for r2, c2 in s2_list:
            for r3, c3 in s3_list:
                if c3 in training_edges.get(c2, set()):
                    if p_s[i, r2] >= 0.05 and p_s[i, r3] >= 0.05:
                        pred_D[i, r2] = True
                        pred_D[i, r3] = True
    res_D = evaluate_authoritative(pred_D, unique_s1, c_s, gt_full, val_s1_set, recovered_map_n037)
    changed_D = int(np.sum(pred_D != pred_N036))
    print(f"\n[Ablation D: Reciprocal Top-3]")
    print(f"  Metric B (F0.5): {res_D['macro_f_05']:.6f} | P: {res_D['macro_precision']:.6f} | R: {res_D['macro_recall']:.6f}")
    print(f"  FP: {res_D['fp']} | In-Pool FN: {res_D['fn_model']} | Total FN: {res_D['total_fn']}")
    print(f"  Changed bits: {changed_D} | Delta TP: {res_A['total_fn'] - res_D['total_fn']} | Delta FP: {res_D['fp'] - res_A['fp']}")

    # Ablation E: Reciprocal + address (aj >= 0.25)
    pred_E = pred_N036.copy()
    for i in unanchored_indices:
        cands_i = c_s[i, :5]
        srcs_i = src_s[i, :5]
        s2_list = [(r, cands_i[r]) for r in range(len(cands_i)) if srcs_i[r] == "S2"]
        s3_list = [(r, cands_i[r]) for r in range(len(cands_i)) if srcs_i[r] == "S3"]
        for r2, c2 in s2_list:
            for r3, c3 in s3_list:
                if c3 in training_edges.get(c2, set()):
                    if aj_s[i, r2] >= 0.25 and aj_s[i, r3] >= 0.25:
                        pred_E[i, r2] = True
                        pred_E[i, r3] = True
    res_E = evaluate_authoritative(pred_E, unique_s1, c_s, gt_full, val_s1_set, recovered_map_n037)
    changed_E = int(np.sum(pred_E != pred_N036))
    print(f"\n[Ablation E: Reciprocal + Address (aj >= 0.25)]")
    print(f"  Metric B (F0.5): {res_E['macro_f_05']:.6f} | P: {res_E['macro_precision']:.6f} | R: {res_E['macro_recall']:.6f}")
    print(f"  FP: {res_E['fp']} | In-Pool FN: {res_E['fn_model']} | Total FN: {res_E['total_fn']}")
    print(f"  Changed bits: {changed_E} | Delta TP: {res_A['total_fn'] - res_E['total_fn']} | Delta FP: {res_E['fp'] - res_A['fp']}")

    # Ablation F: Reciprocal + neural (cos >= 0.62)
    pred_F = pred_N036.copy()
    for i in unanchored_indices:
        cands_i = c_s[i, :5]
        srcs_i = src_s[i, :5]
        s2_list = [(r, cands_i[r]) for r in range(len(cands_i)) if srcs_i[r] == "S2"]
        s3_list = [(r, cands_i[r]) for r in range(len(cands_i)) if srcs_i[r] == "S3"]
        for r2, c2 in s2_list:
            for r3, c3 in s3_list:
                if c3 in training_edges.get(c2, set()):
                    if neural_cos_s[i, r2] >= 0.62 and neural_cos_s[i, r3] >= 0.62:
                        pred_F[i, r2] = True
                        pred_F[i, r3] = True
    res_F = evaluate_authoritative(pred_F, unique_s1, c_s, gt_full, val_s1_set, recovered_map_n037)
    changed_F = int(np.sum(pred_F != pred_N036))
    print(f"\n[Ablation F: Reciprocal + Neural (cos >= 0.62)]")
    print(f"  Metric B (F0.5): {res_F['macro_f_05']:.6f} | P: {res_F['macro_precision']:.6f} | R: {res_F['macro_recall']:.6f}")
    print(f"  FP: {res_F['fp']} | In-Pool FN: {res_F['fn_model']} | Total FN: {res_F['total_fn']}")
    print(f"  Changed bits: {changed_F} | Delta TP: {res_A['total_fn'] - res_F['total_fn']} | Delta FP: {res_F['fp'] - res_A['fp']}")

    # Ablation G: Reciprocal + house/postal guard
    pred_G = pred_N036.copy()
    for i in unanchored_indices:
        cands_i = c_s[i, :5]
        srcs_i = src_s[i, :5]
        s2_list = [(r, cands_i[r]) for r in range(len(cands_i)) if srcs_i[r] == "S2"]
        s3_list = [(r, cands_i[r]) for r in range(len(cands_i)) if srcs_i[r] == "S3"]
        for r2, c2 in s2_list:
            for r3, c3 in s3_list:
                if c3 in training_edges.get(c2, set()):
                    g2 = (aj_s[i, r2] >= 0.30) or (postal_s[i, r2] >= 0.66 and aj_s[i, r2] >= 0.20)
                    g3 = (aj_s[i, r3] >= 0.30) or (postal_s[i, r3] >= 0.66 and aj_s[i, r3] >= 0.20)
                    if g2 and g3:
                        pred_G[i, r2] = True
                        pred_G[i, r3] = True
    res_G = evaluate_authoritative(pred_G, unique_s1, c_s, gt_full, val_s1_set, recovered_map_n037)
    changed_G = int(np.sum(pred_G != pred_N036))
    print(f"\n[Ablation G: Reciprocal + House/Postal Guard]")
    print(f"  Metric B (F0.5): {res_G['macro_f_05']:.6f} | P: {res_G['macro_precision']:.6f} | R: {res_G['macro_recall']:.6f}")
    print(f"  FP: {res_G['fp']} | In-Pool FN: {res_G['fn_model']} | Total FN: {res_G['total_fn']}")
    print(f"  Changed bits: {changed_G} | Delta TP: {res_A['total_fn'] - res_G['total_fn']} | Delta FP: {res_G['fp'] - res_A['fp']}")

    # Ablation H: Reciprocal + contradiction guard (bs == 0)
    pred_H = pred_N036.copy()
    for i in unanchored_indices:
        cands_i = c_s[i, :5]
        srcs_i = src_s[i, :5]
        s2_list = [(r, cands_i[r]) for r in range(len(cands_i)) if srcs_i[r] == "S2"]
        s3_list = [(r, cands_i[r]) for r in range(len(cands_i)) if srcs_i[r] == "S3"]
        for r2, c2 in s2_list:
            for r3, c3 in s3_list:
                if c3 in training_edges.get(c2, set()):
                    if bs_s[i, r2] == 0 and bs_s[i, r3] == 0:
                        g2 = (aj_s[i, r2] >= 0.30) or (postal_s[i, r2] >= 0.66 and aj_s[i, r2] >= 0.20)
                        g3 = (aj_s[i, r3] >= 0.30) or (postal_s[i, r3] >= 0.66 and aj_s[i, r3] >= 0.20)
                        if g2 and g3:
                            pred_H[i, r2] = True
                            pred_H[i, r3] = True
    res_H = evaluate_authoritative(pred_H, unique_s1, c_s, gt_full, val_s1_set, recovered_map_n037)
    changed_H = int(np.sum(pred_H != pred_N036))
    print(f"\n[Ablation H: Reciprocal + Contradiction Guard (bs == 0)]")
    print(f"  Metric B (F0.5): {res_H['macro_f_05']:.6f} | P: {res_H['macro_precision']:.6f} | R: {res_H['macro_recall']:.6f}")
    print(f"  FP: {res_H['fp']} | In-Pool FN: {res_H['fn_model']} | Total FN: {res_H['total_fn']}")
    print(f"  Changed bits: {changed_H} | Delta TP: {res_A['total_fn'] - res_H['total_fn']} | Delta FP: {res_H['fp'] - res_A['fp']}")

    # Ablation I: Full Reciprocal Entity Induction Engine (cos >= 0.62, aj/post guard, bs == 0)
    inducer = TestTimeReciprocalEntityInducer(min_cos=0.62, max_rank=10)
    pred_I = pred_N036.copy()
    for i in unanchored_indices:
        admitted = inducer.decode_unanchored_candidates(
            c_s[i], src_s[i], p_s[i], neural_cos_s[i], aj_s[i], postal_s[i], bs_s[i], training_edges
        )
        for r in admitted:
            pred_I[i, r] = True
    res_I = evaluate_authoritative(pred_I, unique_s1, c_s, gt_full, val_s1_set, recovered_map_n037)
    changed_I = int(np.sum(pred_I != pred_N036))
    print(f"\n[Ablation I: Full Reciprocal Entity Induction Engine]")
    print(f"  Metric B (F0.5): {res_I['macro_f_05']:.6f} | P: {res_I['macro_precision']:.6f} | R: {res_I['macro_recall']:.6f}")
    print(f"  FP: {res_I['fp']} | In-Pool FN: {res_I['fn_model']} | Total FN: {res_I['total_fn']}")
    print(f"  Changed bits: {changed_I} | Delta TP: {res_A['total_fn'] - res_I['total_fn']} | Delta FP: {res_I['fp'] - res_A['fp']}")

    # Ablation J: FINAL EXP_N037 Unified System
    pred_J = pred_I.copy()
    res_J = evaluate_authoritative(pred_J, unique_s1, c_s, gt_full, val_s1_set, recovered_map_n037)
    changed_J = int(np.sum(pred_J != pred_N036))
    print(f"\n[Ablation J: FINAL EXP_N037 Unified System]")
    print(f"  Metric B (F0.5): {res_J['macro_f_05']:.6f} | P: {res_J['macro_precision']:.6f} | R: {res_J['macro_recall']:.6f}")
    print(f"  FP: {res_J['fp']} | In-Pool FN: {res_J['fn_model']} | Total FN: {res_J['total_fn']}")
    print(f"  Predictions: {res_J['pred_count']:,} | Changed bits vs N036: {changed_J} | Delta TP: {res_A['total_fn'] - res_J['total_fn']} | Delta FP: {res_J['fp'] - res_A['fp']}")
    np.savez_compressed(EXP_DIR / "pred_n037_matrix.npz", pred=pred_J)
    print(f"  Saved pred_n037_matrix.npz to {EXP_DIR / 'pred_n037_matrix.npz'}")

    # 6. Generate Mandatory Forensic Audits
    print("\n[Step 5] Compiling Mandatory Forensic Audits...", flush=True)
    fp_records = []
    fn_records = []
    reciprocal_audit_records = []

    for i in range(N_ent):
        s1 = unique_s1[i]
        true_set = gt_full.get(s1, set())
        pred_set = set(c_s[i][pred_J[i]])
        if s1 in recovered_map_n037:
            pred_set.update(recovered_map_n037[s1])

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
                    "postal_match": float(postal_s[i, r]),
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
                    "house_num_match": float(hn_s[i, r]),
                    "postal_match": float(postal_s[i, r]),
                })

        # Reciprocal Audit for newly rescued candidates
        for cid in (pred_set & true_set):
            if cid in set(c_s[i][pred_J[i] & ~pred_N036[i]]):
                r_idx = np.where(c_s[i] == cid)[0][0]
                reciprocal_audit_records.append({
                    "s1_id": s1,
                    "rescued_cid": cid,
                    "rank": int(r_idx),
                    "prob": float(p_s[i, r_idx]),
                    "neural_cosine": float(neural_cos_s[i, r_idx]),
                    "address_jaccard": float(aj_s[i, r_idx]),
                    "postal_match": float(postal_s[i, r_idx]),
                    "resolution_type": "Reciprocal Cross-Source Induction",
                })

    df_fp = pd.DataFrame(fp_records)
    df_fn = pd.DataFrame(fn_records)
    df_recip = pd.DataFrame(reciprocal_audit_records)
    df_fp.to_csv(CODE_DIR / "reports" / "EXP_N037_FP_AUDIT.csv", index=False)
    df_fn.to_csv(CODE_DIR / "reports" / "EXP_N037_FN_AUDIT.csv", index=False)
    df_recip.to_csv(CODE_DIR / "reports" / "EXP_N037_RECIPROCAL_AUDIT.csv", index=False)
    print(f"  Saved FP audit ({len(df_fp):,} records) to: reports/EXP_N037_FP_AUDIT.csv")
    print(f"  Saved FN audit ({len(df_fn):,} records) to: reports/EXP_N037_FN_AUDIT.csv")
    print(f"  Saved Reciprocal audit ({len(df_recip):,} records) to: reports/EXP_N037_RECIPROCAL_AUDIT.csv")

    # 7. Check Champion Progression
    champ_path = CODE_DIR / "reports" / "champion.json"
    with open(champ_path) as f:
        champ_data = json.load(f)
    champ_f05 = champ_data["metric_b_complete_system_f05"]
    print(f"\nActive Champion Baseline (EXP_N036): {champ_f05:.6f}")
    print(f"EXP_N037 Best Result (Ablation J):     {res_J['macro_f_05']:.6f}")

    if res_J["macro_f_05"] > champ_f05:
        print(f"\n*** SUCCESS: EXP_N037 BEATS CHAMPION (+{res_J['macro_f_05'] - champ_f05:.6f})! UPDATING CHAMPION.JSON ***")
        champ_data = {
            "experiment": "EXP_N037",
            "champion_model": "reciprocal_cross_source_entity_induction_EXP_N037",
            "metric_b_complete_system_f05": res_J["macro_f_05"],
            "metric_a_decision_layer_f05": res_J["metric_a_f05"],
            "metric_macro_precision": res_J["macro_precision"],
            "metric_macro_recall": res_J["macro_recall"],
            "eval_population_s1": len(val_s1_set),
            "total_false_positives": res_J["fp"],
            "total_false_negatives_model": res_J["fn_model"],
            "total_false_negatives_blocker": res_J["fn_blocker"],
            "total_false_negatives": res_J["total_fn"],
            "singleton_accuracy": res_J["singleton_accuracy"],
            "multi_match_recall": res_J["multi_match_recall"],
            "ablations": {
                "N037-A": res_A["macro_f_05"],
                "N037-B": res_B["macro_f_05"],
                "N037-C": res_C["macro_f_05"],
                "N037-D": res_D["macro_f_05"],
                "N037-E": res_E["macro_f_05"],
                "N037-F": res_F["macro_f_05"],
                "N037-G": res_G["macro_f_05"],
                "N037-H": res_H["macro_f_05"],
                "N037-I": res_I["macro_f_05"],
                "N037-J": res_J["macro_f_05"],
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(champ_path, "w") as f:
            json.dump(champ_data, f, indent=2)
    else:
        print(f"\n*** EXP_N037 did not beat champion ({res_J['macro_f_05']:.6f} <= {champ_f05:.6f}). Champion preserved. ***")

    # 8. Save results JSON
    results_json_path = CODE_DIR / "reports" / "EXP_N037_RESULTS.json"
    res_summary = {
        "experiment": "EXP_N037",
        "authoritative_baseline_metric_b": champ_f05,
        "n037_evaluated_metric_b": res_J["macro_f_05"],
        "delta_metric_b": res_J["macro_f_05"] - champ_f05,
        "precision": res_J["macro_precision"],
        "recall": res_J["macro_recall"],
        "fp": res_J["fp"],
        "fn_model": res_J["fn_model"],
        "fn_blocker": res_J["fn_blocker"],
        "total_fn": res_J["total_fn"],
        "ablations": {
            "N037-A": res_A["macro_f_05"],
            "N037-B": res_B["macro_f_05"],
            "N037-C": res_C["macro_f_05"],
            "N037-D": res_D["macro_f_05"],
            "N037-E": res_E["macro_f_05"],
            "N037-F": res_F["macro_f_05"],
            "N037-G": res_G["macro_f_05"],
            "N037-H": res_H["macro_f_05"],
            "N037-I": res_I["macro_f_05"],
            "N037-J": res_J["macro_f_05"],
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(results_json_path, "w") as f:
        json.dump(res_summary, f, indent=2)
    print(f"Results summary saved to: {results_json_path}")


if __name__ == "__main__":
    main()
