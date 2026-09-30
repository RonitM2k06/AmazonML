#!/usr/bin/env python3
"""
EXP_N012 — Phase 4 & 5: Train Union-Distribution Ranker with Provenance & Path Features.

Combines:
- 49,735 Lexical Positives (from train_pairs_exp008.npz)
- 204,930 Lexical Negatives (from train_pairs_exp008.npz)
- 60,000 Mined High-Cosine ANN Hard Negatives (from hard_neg_manifest.csv)

Features:
- 56 Base Lexical / Address / Interaction features
- 1 Two-Tower Neural Cosine Similarity
- 3 Provenance features (is_lexical, is_ann_only, is_s3)
- 7 Group & Cross-Source Path features (group_max_cos, cos_margin_to_max, is_top_cos,
                                        cos_x_name_jac, cos_x_addr_jac,
                                        cos_x_house_num_exact, cos_x_addr_norm_exact)
Total: 67 features

Training:
- Scale pos weight = 1.0 (proper unweighted calibration)
- LightGBM + GPU XGBoost
- Generates experiments/probs_n012_union_lgb.npy & probs_n012_union_xgb.npy
- Evaluates on authoritative 25k frozen validation set (Metric B)
"""

from __future__ import annotations

import os
import sys
import time
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONUNBUFFERED"] = "1"

# Paths
CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_DIR.parents[1]
sys.path.insert(0, str(CODE_DIR / "src"))
sys.path.insert(0, str(CODE_DIR / "scripts"))

from authoritative_eval_fn import run_authoritative_eval
from pair_features import extract_pair_features, FEATURE_NAMES
from two_tower_encoder import load_two_tower, RecordTokenizer, encode_records_fused

DATA_DIR = PROJECT_ROOT / "dataset" / "train"
EXP_DIR = CODE_DIR / "experiments"
MODEL_DIR = CODE_DIR / "models"


def main():
    t_start = time.time()
    print("=" * 75, flush=True)
    print("[EXP_N012] Phase 4 & 5: Training Union-Distribution Ranker", flush=True)
    print("=" * 75, flush=True)

    # 1. Load Base Training Pairs
    print("\n[Step 1] Loading base lexical train pairs (train_pairs_exp008.npz)...", flush=True)
    tp = np.load(EXP_DIR / "train_pairs_exp008.npz", allow_pickle=True)
    X_base = tp["X"].astype(np.float32)
    y_base = tp["y"].astype(np.int8)
    s1_base = tp["s1_ids"].astype(str)
    cands_base = tp["cand_ids"].astype(str)
    sources_base = tp["sources"].astype(str)
    N_base = len(y_base)
    print(f"Loaded {N_base:,} base pairs (Positives: {np.sum(y_base==1):,}, Negatives: {np.sum(y_base==0):,})", flush=True)

    # 2. Sample Mined ANN Negatives from hard_neg_manifest.csv
    print("\n[Step 2] Sampling ANN hard negatives from hard_neg_manifest.csv...", flush=True)
    df_man = pd.read_csv(EXP_DIR / "hard_neg_manifest.csv")
    # Take top 60,000 hardest negatives
    df_man_sample = df_man.sort_values(by="score", ascending=False).head(60000)
    print(f"Sampled {len(df_man_sample):,} hardest ANN negatives (score range: {df_man_sample['score'].min():.4f} to {df_man_sample['score'].max():.4f})", flush=True)

    ann_s1 = df_man_sample["s1_id"].values.astype(str)
    ann_cand = df_man_sample["neg_id"].values.astype(str)
    ann_cos = df_man_sample["score"].values.astype(np.float32)
    ann_y = np.zeros(len(ann_s1), dtype=np.int8)
    ann_sources = np.array(["S3" if c.startswith("S3") else "S2" for c in ann_cand])

    # 3. Stream raw text for ANN pairs to extract features
    print("\n[Step 3] Extracting features for 60,000 ANN negative pairs...", flush=True)
    needed_s1 = set(ann_s1)
    needed_targets = set(ann_cand)

    s1_recs = {}
    with open(DATA_DIR / "train_source1.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in needed_s1:
                s1_recs[p[0]] = (p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "", p[3] if len(p) > 3 else "")

    target_recs = {}
    for src_file in ["train_source2.tsv", "train_source3.tsv"]:
        with open(DATA_DIR / src_file, encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if p[0] in needed_targets:
                    target_recs[p[0]] = (p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "", p[3] if len(p) > 3 else "")

    X_ann_56 = np.zeros((len(ann_s1), 56), dtype=np.float32)
    t0 = time.time()
    for i in range(len(ann_s1)):
        s1 = s1_recs.get(ann_s1[i], ("", "", ""))
        tgt = target_recs.get(ann_cand[i], ("", "", ""))
        X_ann_56[i] = extract_pair_features(s1[0], s1[1], s1[2], tgt[0], tgt[1], tgt[2])
    print(f"Extracted 56 features for {len(ann_s1):,} ANN pairs in {time.time()-t0:.1f}s", flush=True)

    # 4. Neural Cosine for Base Pairs using Two-Tower Model on GPU
    print("\n[Step 4] Computing fine-tuned Two-Tower cosine for base train pairs on GPU...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    two_tower = load_two_tower(str(MODEL_DIR / "two_tower_stageC.pt"), device=device)
    two_tower.eval()
    tok = RecordTokenizer()

    # Collect unique entities in base pairs
    needed_base_s1 = set(s1_base)
    needed_base_cand = set(cands_base)

    base_s1_dict = {}
    with open(DATA_DIR / "train_source1.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in needed_base_s1:
                base_s1_dict[p[0]] = (p[0], p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "", p[3] if len(p) > 3 else "")

    base_tgt_dict = {}
    for src_file in ["train_source2.tsv", "train_source3.tsv"]:
        with open(DATA_DIR / src_file, encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if p[0] in needed_base_cand:
                    base_tgt_dict[p[0]] = (p[0], p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "", p[3] if len(p) > 3 else "")

    print(f"Encoding {len(base_s1_dict):,} S1 and {len(base_tgt_dict):,} Target records on GPU...", flush=True)
    t0 = time.time()
    s1_ids_list, s1_vecs = encode_records_fused(two_tower, tok, base_s1_dict, batch_size=512, device=device)
    tgt_ids_list, tgt_vecs = encode_records_fused(two_tower, tok, base_tgt_dict, batch_size=512, device=device)
    print(f"GPU encoding finished in {time.time()-t0:.1f}s", flush=True)

    s1_vmap = {eid: s1_vecs[i] for i, eid in enumerate(s1_ids_list)}
    tgt_vmap = {eid: tgt_vecs[i] for i, eid in enumerate(tgt_ids_list)}

    base_cos = np.zeros(N_base, dtype=np.float32)
    for i in range(N_base):
        v1 = s1_vmap.get(s1_base[i])
        v2 = tgt_vmap.get(cands_base[i])
        if v1 is not None and v2 is not None:
            base_cos[i] = float(np.dot(v1, v2))

    # Free vector memory
    del s1_vecs, tgt_vecs, s1_vmap, tgt_vmap, base_s1_dict, base_tgt_dict, s1_recs, target_recs
    import gc
    gc.collect()

    # 5. Assemble Combined Training Matrix with Provenance & Path Features
    print("\n[Step 5] Assembling 67 features for Union Training Set...", flush=True)
    # Combine arrays
    all_s1 = np.concatenate([s1_base, ann_s1])
    all_cand = np.concatenate([cands_base, ann_cand])
    all_y = np.concatenate([y_base, ann_y])
    all_sources = np.concatenate([sources_base, ann_sources])
    all_X_56 = np.vstack([X_base, X_ann_56])
    all_cos = np.concatenate([base_cos, ann_cos])
    N_total = len(all_y)

    # Provenance
    is_lexical = np.concatenate([np.ones(N_base, dtype=np.float32), np.zeros(len(ann_s1), dtype=np.float32)])
    is_ann_only = 1.0 - is_lexical
    is_s3 = (all_sources == "S3").astype(np.float32)

    # Group Features per S1
    print("Computing group and path features on training set...", flush=True)
    unique_s1, s1_codes = np.unique(all_s1, return_inverse=True)
    
    # Vectorized group max cosine
    import scipy.ndimage as ndi
    max_cos_per_group = ndi.maximum(all_cos, labels=s1_codes, index=np.arange(len(unique_s1)))
    group_max_cos = max_cos_per_group[s1_codes].astype(np.float32)
    cos_margin_to_max = (group_max_cos - all_cos).astype(np.float32)
    is_top_cos = (cos_margin_to_max <= 1e-4).astype(np.float32)

    # Cross features
    name_jac = all_X_56[:, 11]
    addr_jac = all_X_56[:, 30]
    house_num_exact = all_X_56[:, 40]
    addr_norm_exact = all_X_56[:, 29]

    cos_x_name_jac = (all_cos * name_jac).astype(np.float32)
    cos_x_addr_jac = (all_cos * addr_jac).astype(np.float32)
    cos_x_house_exact = (all_cos * house_num_exact).astype(np.float32)
    cos_x_addr_exact = (all_cos * addr_norm_exact).astype(np.float32)

    X_train_union = np.column_stack([
        all_X_56,
        all_cos,
        is_lexical,
        is_ann_only,
        is_s3,
        group_max_cos,
        cos_margin_to_max,
        is_top_cos,
        cos_x_name_jac,
        cos_x_addr_jac,
        cos_x_house_exact,
        cos_x_addr_exact,
    ]).astype(np.float32)

    print(f"Union Train Matrix: {X_train_union.shape} (Positives: {np.sum(all_y==1):,}, Negatives: {np.sum(all_y==0):,})", flush=True)

    # 6. Build Matching Features for Validation Set (val_pairs_n010.npz)
    print("\n[Step 6] Constructing matching 67 features for Validation Set (val_pairs_n010.npz)...", flush=True)
    va = np.load(EXP_DIR / "val_pairs_n010.npz", allow_pickle=True)
    va_s1 = va["s1_ids"].astype(str)
    va_cand = va["cand_ids"].astype(str)
    va_sources = va["sources"].astype(str)
    va_X_57 = va["X"].astype(np.float32)
    va_cos = va_X_57[:, 56]
    va_X_56 = va_X_57[:, :56]
    N_va = len(va_s1)

    # Provenance on validation
    v9 = np.load(EXP_DIR / "val_pairs_n009.npz", allow_pickle=True)
    set_v9 = set(zip(v9["s1_ids"].astype(str), v9["cand_ids"].astype(str)))
    del v9
    gc.collect()

    va_is_lexical = np.array([(s, c) in set_v9 for s, c in zip(va_s1, va_cand)], dtype=np.float32)
    va_is_ann_only = 1.0 - va_is_lexical
    va_is_s3 = (va_sources == "S3").astype(np.float32)

    # Group features on validation
    va_unique_s1, va_s1_codes = np.unique(va_s1, return_inverse=True)
    va_max_cos_per_group = ndi.maximum(va_cos, labels=va_s1_codes, index=np.arange(len(va_unique_s1)))
    va_group_max_cos = va_max_cos_per_group[va_s1_codes].astype(np.float32)
    va_cos_margin = (va_group_max_cos - va_cos).astype(np.float32)
    va_is_top_cos = (va_cos_margin <= 1e-4).astype(np.float32)

    va_cos_x_name = (va_cos * va_X_56[:, 11]).astype(np.float32)
    va_cos_x_addr = (va_cos * va_X_56[:, 30]).astype(np.float32)
    va_cos_x_house = (va_cos * va_X_56[:, 40]).astype(np.float32)
    va_cos_x_addr_ex = (va_cos * va_X_56[:, 29]).astype(np.float32)

    X_val_union = np.column_stack([
        va_X_56,
        va_cos,
        va_is_lexical,
        va_is_ann_only,
        va_is_s3,
        va_group_max_cos,
        va_cos_margin,
        va_is_top_cos,
        va_cos_x_name,
        va_cos_x_addr,
        va_cos_x_house,
        va_cos_x_addr_ex,
    ]).astype(np.float32)
    print(f"Union Val Matrix: {X_val_union.shape}", flush=True)

    # 7. Train LightGBM Ranker (Properly Calibrated, unweighted)
    import lightgbm as lgb
    print("\n[Step 7] Training Calibrated LightGBM Classifier (scale_pos_weight=1.0)...", flush=True)
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": 750,
        "max_depth": 8,
        "num_leaves": 95,
        "learning_rate": 0.04,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_samples": 25,
        "scale_pos_weight": 1.0,  # Unweighted calibration
        "n_jobs": 4,
        "random_state": 42,
        "verbose": -1,
    }
    t0 = time.time()
    lgb_clf = lgb.LGBMClassifier(**lgb_params)
    lgb_clf.fit(X_train_union, all_y)
    print(f"LightGBM trained in {time.time()-t0:.1f}s", flush=True)

    print("Predicting LightGBM probabilities on 2,000,000 validation pairs...", flush=True)
    t0 = time.time()
    val_probs_lgb = lgb_clf.predict_proba(X_val_union)[:, 1].astype(np.float32)
    print(f"LightGBM inference complete in {time.time()-t0:.1f}s", flush=True)
    np.save(EXP_DIR / "probs_n012_union_lgb.npy", val_probs_lgb)

    # 8. Train XGBoost on GPU
    import xgboost as xgb
    print("\n[Step 8] Training Calibrated XGBoost Classifier on GPU (scale_pos_weight=1.0)...", flush=True)
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 750,
        "max_depth": 8,
        "learning_rate": 0.04,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "scale_pos_weight": 1.0,
        "tree_method": "hist",
        "device": "cuda",
        "random_state": 42,
    }
    t0 = time.time()
    xgb_clf = xgb.XGBClassifier(**xgb_params)
    xgb_clf.fit(X_train_union, all_y)
    print(f"XGBoost trained in {time.time()-t0:.1f}s", flush=True)

    print("Predicting XGBoost probabilities on 2,000,000 validation pairs...", flush=True)
    t0 = time.time()
    val_probs_xgb = xgb_clf.predict_proba(X_val_union)[:, 1].astype(np.float32)
    print(f"XGBoost inference complete in {time.time()-t0:.1f}s", flush=True)
    np.save(EXP_DIR / "probs_n012_union_xgb.npy", val_probs_xgb)

    # 9. Authoritative Complete-System Evaluation
    print("\n" + "=" * 75, flush=True)
    print("[EXP_N012] Running Authoritative Complete-System Evaluation (Metric B)...", flush=True)
    print("=" * 75, flush=True)

    # Let's test a couple of threshold configurations with run_authoritative_eval
    best_res = None
    best_mB = 0.0

    for base_th in [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92]:
        for margin in [0.03, 0.05, 0.08]:
            res = run_authoritative_eval(
                val_npz_path=str(EXP_DIR / "val_pairs_n010.npz"),
                probs_lgb_path=str(EXP_DIR / "probs_n012_union_lgb.npy"),
                probs_xgb_path=str(EXP_DIR / "probs_n012_union_xgb.npy"),
                w_lgb=0.50,
                label=f"EXP_N012_th{base_th:.2f}_m{margin:.2f}",
                base_th=base_th,
                dense_th=base_th + 0.02,
                margin=margin,
                anchor=base_th + 0.04,
                bridge=base_th - 0.04,
                density_cutoff=75,
                verbose=False
            )
            mB = res["B"]["macro_f_05"]
            mA = res["A"]["macro_f_05"]
            pB = res["B"]["macro_precision"]
            rB = res["B"]["macro_recall"]
            errs = res["errors"]
            print(f"th={base_th:.2f}, margin={margin:.2f} -> Metric B: {mB:.6f} (A: {mA:.6f}, P: {pB:.4f}, R: {rB:.4f}, FP: {errs['fp']}, FN-mod: {errs['fn_model']}, FN-blk: {errs['fn_blocker']})", flush=True)
            if mB > best_mB:
                best_mB = mB
                best_res = (base_th, margin, res)

    print("\n" + "=" * 75, flush=True)
    b_th, b_m, b_r = best_res
    print(f"EXP_N012 UNION-RANKER CHAMPION RESULTS:")
    print(f"  Best Config: base_th={b_th:.2f}, margin={b_m:.2f}")
    print(f"  Complete System F0.5 (Metric B): {b_r['B']['macro_f_05']:.6f}")
    print(f"  Decision Layer F0.5 (Metric A):  {b_r['A']['macro_f_05']:.6f}")
    print(f"  Macro Precision:                {b_r['B']['macro_precision']:.6f}")
    print(f"  Macro Recall:                   {b_r['B']['macro_recall']:.6f}")
    print(f"  False Positives (FP):           {b_r['errors']['fp']}")
    print(f"  False Negatives Model:          {b_r['errors']['fn_model']}")
    print(f"  False Negatives Blocker:        {b_r['errors']['fn_blocker']}")
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
