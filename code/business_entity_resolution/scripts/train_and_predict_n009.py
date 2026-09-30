#!/usr/bin/env python3
"""
EXP_N009 / EXP_N010 Training and Scoring Pipeline:
1. Load train pairs (train_pairs_exp008.npz) augmented with cosine sim (feature #57)
2. Train LightGBM + XGBoost with hard-negative weighting
3. Predict on val_pairs_n009.npz (1,912,769 pairs)
4. Save probs_n009_lgb.npy, probs_n009_xgb.npy
"""
import os
import sys
import time
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["PYTHONUNBUFFERED"] = "1"

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
code_dir = os.path.join(project_root, "code", "business_entity_resolution")
sys.path.insert(0, os.path.join(code_dir, "src"))

from neural_encoder import load_embeddings

EXPERIMENTS_DIR = os.path.join(code_dir, "experiments")
TRAIN_NPZ = os.path.join(EXPERIMENTS_DIR, "train_pairs_exp008.npz")
VAL_NPZ = os.path.join(EXPERIMENTS_DIR, "val_pairs_n009.npz")
S1_EMB_PATH = os.path.join(EXPERIMENTS_DIR, "s1_embeddings.npz")
TARGET_EMB_PATH = os.path.join(EXPERIMENTS_DIR, "target_embeddings.npz")

PROBS_LGB_PATH = os.path.join(EXPERIMENTS_DIR, "probs_n009_lgb.npy")
PROBS_XGB_PATH = os.path.join(EXPERIMENTS_DIR, "probs_n009_xgb.npy")

def main():
    t_start = time.time()
    print("=" * 70, flush=True)
    print("[EXP_N010] Training Matching Models on 57 Features...", flush=True)
    print("=" * 70, flush=True)

    # 1. Load train dataset (56 features)
    print("Loading train dataset...", flush=True)
    tr_data = np.load(TRAIN_NPZ, allow_pickle=True)
    X_train_56 = tr_data["X"].astype(np.float32)
    y_train = tr_data["y"].astype(np.int8)
    train_s1_ids = tr_data["s1_ids"]
    train_cand_ids = tr_data["cand_ids"]
    N_tr = len(y_train)

    # Load embeddings to compute cosine feature on train
    print("Appending neural_cosine_sim to train pairs...", flush=True)
    s1_ids_emb, s1_vecs = load_embeddings(S1_EMB_PATH)
    target_ids_emb, tgt_vecs = load_embeddings(TARGET_EMB_PATH)
    s1_emb_map = {id_: i for i, id_ in enumerate(s1_ids_emb)}
    tgt_emb_map = {id_: i for i, id_ in enumerate(target_ids_emb)}

    train_cosine = np.zeros(N_tr, dtype=np.float32)
    for i in range(N_tr):
        idx1 = s1_emb_map.get(train_s1_ids[i])
        idx2 = tgt_emb_map.get(train_cand_ids[i])
        if idx1 is not None and idx2 is not None:
            train_cosine[i] = float(np.dot(s1_vecs[idx1], tgt_vecs[idx2]))

    X_train_57 = np.hstack([X_train_56, train_cosine.reshape(-1, 1)]).astype(np.float32)
    print(f"Train feature matrix: {X_train_57.shape} (pos={np.sum(y_train==1):,}, neg={np.sum(y_train==0):,})", flush=True)

    # 2. Train LightGBM
    import lightgbm as lgb
    print("\n--- Training LightGBM ---", flush=True)
    pos = int(np.sum(y_train == 1))
    neg = int(np.sum(y_train == 0))
    scale = neg / max(1, pos)

    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": 600,
        "max_depth": 7,
        "num_leaves": 63,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "scale_pos_weight": scale,
        "n_jobs": 4,
        "random_state": 42,
        "verbose": -1,
    }
    t0 = time.time()
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_train_57, y_train)
    print(f"LightGBM trained in {time.time()-t0:.1f}s", flush=True)

    # 3. Train XGBoost
    import xgboost as xgb
    print("\n--- Training XGBoost ---", flush=True)
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 600,
        "max_depth": 7,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "scale_pos_weight": scale,
        "n_jobs": 4,
        "random_state": 42,
        "tree_method": "hist",
        "device": "cuda",
        "verbosity": 0,
    }
    t0 = time.time()
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train_57, y_train)
    print(f"XGBoost trained in {time.time()-t0:.1f}s", flush=True)

    # 4. Load val_pairs_n009.npz (already has 57 features)
    print(f"\nLoading validation pairs from {VAL_NPZ}...", flush=True)
    va_data = np.load(VAL_NPZ, allow_pickle=True)
    X_val_57 = va_data["X"].astype(np.float32)
    print(f"Val feature matrix: {X_val_57.shape}", flush=True)

    # 5. Predict probabilities
    print("Generating LightGBM validation probabilities...", flush=True)
    t0 = time.time()
    probs_lgb = lgb_model.predict_proba(X_val_57)[:, 1].astype(np.float32)
    np.save(PROBS_LGB_PATH, probs_lgb)
    print(f"Saved {PROBS_LGB_PATH} in {time.time()-t0:.1f}s (mean={probs_lgb.mean():.4f})", flush=True)

    print("Generating XGBoost validation probabilities...", flush=True)
    t0 = time.time()
    probs_xgb = xgb_model.predict_proba(X_val_57)[:, 1].astype(np.float32)
    np.save(PROBS_XGB_PATH, probs_xgb)
    print(f"Saved {PROBS_XGB_PATH} in {time.time()-t0:.1f}s (mean={probs_xgb.mean():.4f})", flush=True)

    print(f"\nTotal pipeline completed in {time.time()-t_start:.1f}s", flush=True)

if __name__ == "__main__":
    main()
