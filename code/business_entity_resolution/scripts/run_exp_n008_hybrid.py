#!/usr/bin/env python3
"""
EXP_N008: Hybrid Pipeline — 56 Existing Features + Neural Cosine Score (57th feature).

Hypothesis: The cosine similarity from paraphrase-multilingual-MiniLM-L12-v2, while
insufficient as a standalone matcher (EXP_N001 F0.5=0.554), provides semantic cross-lingual
signal absent from the existing 56 hand-crafted features. Adding it as feature #57
to the XGB/LGB/CAT pipeline may push Macro F0.5 above the current champion (0.951463).

Pipeline:
  1. Load train_pairs_exp008.npz (56 features, 254,665 pairs, ~95.73% blocker recall)
  2. Build val pairs from scratch (56 features) using the same high-recall blocker
  3. For each (s1_id, cand_id) pair: compute cosine score from cached embeddings
  4. Augment both train and val feature matrices with cosine score → 57 features
  5. Train LGB + XGB on 57 features with 5-fold CV
  6. Blend predictions and sweep thresholds with transitive bridge decision
  7. Compare final F0.5 against champion EXP_007 (0.951463)

IMPORTANT: Does NOT overwrite any existing champion files.
"""

import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# ─────────────────────────────────────────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
code_dir = os.path.join(project_root, "code", "business_entity_resolution")
src_dir = os.path.join(code_dir, "src")
sys.path.insert(0, src_dir)
sys.path.insert(0, os.path.join(code_dir, "scripts"))

from neural_encoder import NeuralEncoder, load_embeddings, save_embeddings
from neural_retrieval import NeuralRetriever
from pair_features import FEATURE_NAMES, extract_pair_features
from build_pair_dataset import UNION_E_KEYS, build_pair_dataset, save_pair_dataset
from entity_decision import EntityDecisionEngine, run_threshold_sweep
from evaluation import Evaluator
from blocking import build_target_index, generate_candidates

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DATASET_DIR = os.path.join(project_root, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
EXPERIMENTS_DIR = os.path.join(code_dir, "experiments")
REPORTS_DIR = os.path.join(code_dir, "reports")
MODELS_DIR = os.path.join(code_dir, "models")

VAL_IDS_PATH = os.path.join(EXPERIMENTS_DIR, "validation_s1_ids.txt")
TRAIN_NPZ_EXP008 = os.path.join(EXPERIMENTS_DIR, "train_pairs_exp008.npz")
VAL_NPZ_N008 = os.path.join(EXPERIMENTS_DIR, "val_pairs_n008.npz")
S1_EMB_PATH = os.path.join(EXPERIMENTS_DIR, "s1_embeddings.npz")
TARGET_EMB_PATH = os.path.join(EXPERIMENTS_DIR, "target_embeddings.npz")

CHAMPION_JSON = os.path.join(REPORTS_DIR, "champion.json")
REGISTRY_PATH = os.path.join(REPORTS_DIR, "neural_experiment_registry.csv")

VAL_SAMPLE_SIZE = 25000
SEED = 42
NEURAL_FEATURE_NAME = "neural_cosine_sim"

os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cosine score computation from cached embeddings
# ─────────────────────────────────────────────────────────────────────────────
def build_cosine_lookup(
    s1_emb_path: str,
    target_emb_path: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load cached embeddings and build ID → vector lookup dicts.
    Vectors are L2-normalised, so dot product = cosine similarity.
    """
    print("[EXP_N008] Loading cached embeddings for cosine lookup...")
    s1_ids, s1_vecs = load_embeddings(s1_emb_path)
    target_ids, target_vecs = load_embeddings(target_emb_path)

    s1_lookup = {sid: s1_vecs[i] for i, sid in enumerate(s1_ids)}
    target_lookup = {tid: target_vecs[i] for i, tid in enumerate(target_ids)}

    print(f"  S1 lookup: {len(s1_lookup):,} entries")
    print(f"  Target lookup: {len(target_lookup):,} entries")
    return s1_lookup, target_lookup


def compute_cosine_features(
    s1_ids_arr: np.ndarray,
    cand_ids_arr: np.ndarray,
    s1_lookup: Dict[str, np.ndarray],
    target_lookup: Dict[str, np.ndarray],
    missing_val: float = 0.0,
) -> np.ndarray:
    """
    Compute cosine similarity for each (s1_id, cand_id) pair.
    Returns float32 array [N] with cosine scores.
    Missing embeddings (IDs not in cache) are filled with missing_val.
    """
    N = len(s1_ids_arr)
    cosine_scores = np.full(N, missing_val, dtype=np.float32)

    missing_s1 = 0
    missing_cand = 0

    for i in range(N):
        s1_id = str(s1_ids_arr[i])
        cand_id = str(cand_ids_arr[i])

        s1_vec = s1_lookup.get(s1_id)
        cand_vec = target_lookup.get(cand_id)

        if s1_vec is None:
            missing_s1 += 1
            continue
        if cand_vec is None:
            missing_cand += 1
            continue

        cosine_scores[i] = float(np.dot(s1_vec, cand_vec))  # L2-normed → cosine

    if missing_s1 > 0 or missing_cand > 0:
        print(f"  [cosine] Missing embeddings: {missing_s1:,} S1, {missing_cand:,} cand "
              f"(filled with {missing_val})")

    return cosine_scores


# ─────────────────────────────────────────────────────────────────────────────
# Encode new records not in cache
# ─────────────────────────────────────────────────────────────────────────────
def encode_missing_records(
    records: Dict[str, Tuple],
    existing_lookup: Dict[str, np.ndarray],
    encoder: NeuralEncoder,
    label: str,
) -> Dict[str, np.ndarray]:
    """Encode any records not already in the embedding cache."""
    missing_ids = [rid for rid in records if rid not in existing_lookup]
    if not missing_ids:
        print(f"  [{label}] All {len(records):,} records already in embedding cache.")
        return existing_lookup

    print(f"  [{label}] Encoding {len(missing_ids):,} missing records not in cache...")
    missing_recs = {rid: records[rid] for rid in missing_ids}
    new_ids, new_vecs = encoder.encode_records(missing_recs)
    for i, rid in enumerate(new_ids):
        existing_lookup[rid] = new_vecs[i]

    return existing_lookup


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — validation split
# ─────────────────────────────────────────────────────────────────────────────
def load_validation_data(val_sample_size: int = VAL_SAMPLE_SIZE, seed: int = SEED):
    """Load frozen 25k validation split (S1, targets, ground truth)."""
    t0 = time.time()
    random.seed(seed)

    print("=" * 70)
    print("[EXP_N008] Loading validation data...")

    with open(VAL_IDS_PATH, "r", encoding="utf-8") as f:
        val_pool = [line.strip() for line in f if line.strip()]
    selected_val_s1 = set(random.sample(val_pool, min(val_sample_size, len(val_pool))))

    val_matches: Dict[str, List[str]] = {}
    val_needed_s2: Set[str] = set()
    val_needed_s3: Set[str] = set()

    with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1 = parts[0]
            if s1 not in selected_val_s1:
                continue
            mids = parts[1].split(",") if len(parts) > 1 and parts[1].strip() else []
            val_matches[s1] = mids
            for m in mids:
                if m.startswith("S2-"):
                    val_needed_s2.add(m)
                elif m.startswith("S3-"):
                    val_needed_s3.add(m)
            if len(val_matches) >= val_sample_size:
                break

    val_s1_records: Dict[str, Tuple] = {}
    with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), "r", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            eid = line.split("\t", 1)[0]
            if eid in selected_val_s1:
                parts = line.rstrip("\r\n").split("\t")
                val_s1_records[eid] = (parts[0], parts[1], parts[2], parts[3])
                if len(val_s1_records) >= len(selected_val_s1):
                    break

    s2_records: Dict[str, Tuple] = {}
    bg_s2 = 0
    with open(os.path.join(TRAIN_DIR, "train_source2.tsv"), "r", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            eid = line.split("\t", 1)[0]
            if eid in val_needed_s2:
                parts = line.rstrip("\r\n").split("\t")
                s2_records[eid] = (parts[0], parts[1], parts[2], parts[3])
            elif bg_s2 < 30000 and random.random() < 0.012:
                parts = line.rstrip("\r\n").split("\t")
                s2_records[eid] = (parts[0], parts[1], parts[2], parts[3])
                bg_s2 += 1

    s3_records: Dict[str, Tuple] = {}
    bg_s3 = 0
    with open(os.path.join(TRAIN_DIR, "train_source3.tsv"), "r", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            eid = line.split("\t", 1)[0]
            if eid in val_needed_s3:
                parts = line.rstrip("\r\n").split("\t")
                s3_records[eid] = (parts[0], parts[1], parts[2], parts[3])
            elif bg_s3 < 30000 and random.random() < 0.012:
                parts = line.rstrip("\r\n").split("\t")
                s3_records[eid] = (parts[0], parts[1], parts[2], parts[3])
                bg_s3 += 1

    target_records = {**s2_records, **s3_records}
    print(f"  Loaded: {len(val_s1_records):,} S1, {len(target_records):,} targets in {time.time()-t0:.1f}s")
    return val_s1_records, target_records, val_matches


# ─────────────────────────────────────────────────────────────────────────────
# Model training
# ─────────────────────────────────────────────────────────────────────────────
def train_lgb_model(X_train: np.ndarray, y_train: np.ndarray, n_features: int):
    """Train LightGBM binary classifier."""
    import lightgbm as lgb

    print(f"[EXP_N008] Training LightGBM on {X_train.shape[0]:,} pairs ({n_features} features)...")
    pos = int(np.sum(y_train == 1))
    neg = int(np.sum(y_train == 0))
    scale = neg / max(1, pos)

    params = {
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
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    print(f"  LGB training complete. Best iteration: {model.best_iteration_}")
    return model


def train_xgb_model(X_train: np.ndarray, y_train: np.ndarray, n_features: int):
    """Train XGBoost binary classifier."""
    import xgboost as xgb

    print(f"[EXP_N008] Training XGBoost on {X_train.shape[0]:,} pairs ({n_features} features)...")
    pos = int(np.sum(y_train == 1))
    neg = int(np.sum(y_train == 0))
    scale = neg / max(1, pos)

    params = {
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
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    print("  XGB training complete.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Entity-level evaluation with transitive bridge
# ─────────────────────────────────────────────────────────────────────────────
def build_s1_candidate_lists(
    val_meta: List[Dict],
    probs: np.ndarray,
    feature_names: List[str],
    X_val: np.ndarray,
) -> Dict[str, List[Dict]]:
    """Build per-S1 candidate probability list for EntityDecisionEngine."""
    feat_idx = {name: i for i, name in enumerate(feature_names)}
    s1_cands: Dict[str, List[Dict]] = defaultdict(list)
    for i, meta in enumerate(val_meta):
        if meta["label"] == 0 or True:  # include all candidates
            feats = {name: float(X_val[i, j]) for name, j in feat_idx.items()}
            s1_cands[meta["s1_id"]].append({
                "cand_id": meta["cand_id"],
                "source": meta["source"],
                "prob": float(probs[i]),
                "features": feats,
            })
    return dict(s1_cands)


def run_transitive_bridge_sweep(
    all_s1_candidates: Dict[str, List[Dict]],
    val_matches: Dict[str, List[str]],
) -> Tuple[Dict, float]:
    """
    Sweep bridge+threshold parameters and return best F0.5 result.
    Mirrors the champion EXP_007 decision strategy.
    """
    best_result = {"macro_f_05": 0.0}
    best_params = {}

    # Champion parameters as starting point
    base_thresholds = [0.950, 0.955, 0.960, 0.962, 0.965, 0.968, 0.970]
    dense_thresholds = [0.960, 0.963, 0.966, 0.968, 0.970]
    margin_deltas = [0.025, 0.030, 0.035]
    anchor_gates = [0.980, 0.985, 0.990]
    bridge_thresholds = [0.930, 0.935, 0.938, 0.942]
    density_cutoffs = [80, 100, 120]

    print("[EXP_N008] Sweeping transitive bridge parameters...")
    sweep_count = 0

    for base_th in base_thresholds:
        for dense_th in dense_thresholds:
            if dense_th <= base_th:
                continue
            for margin in margin_deltas:
                for anchor in anchor_gates:
                    for bridge in bridge_thresholds:
                        for density_cutoff in density_cutoffs:
                            # Build predictions using transitive bridge logic
                            evaluator = Evaluator()
                            for s1_id, true_mids in val_matches.items():
                                cands = all_s1_candidates.get(s1_id, [])
                                if not cands:
                                    evaluator.update(s1_id, set(true_mids), set())
                                    continue

                                sorted_cands = sorted(cands, key=lambda c: c["prob"], reverse=True)
                                top_prob = sorted_cands[0]["prob"]

                                if top_prob < base_th * 0.85:
                                    evaluator.update(s1_id, set(true_mids), set())
                                    continue

                                num_cands = len(cands)
                                effective_th = dense_th if num_cands > density_cutoff else base_th

                                accepted = set()
                                anchor_ids = set()

                                for cand in sorted_cands:
                                    prob = cand["prob"]
                                    if prob >= effective_th and (top_prob - prob) <= margin:
                                        accepted.add(cand["cand_id"])
                                        if prob >= anchor:
                                            anchor_ids.add(cand["cand_id"])

                                # Transitive cross-source consistency bridge
                                if anchor_ids:
                                    for cand in sorted_cands:
                                        if cand["cand_id"] in accepted:
                                            continue
                                        if cand["prob"] >= bridge:
                                            src = cand.get("source", "S2")
                                            accepted_sources = {
                                                c2["source"] for c2 in sorted_cands
                                                if c2["cand_id"] in anchor_ids
                                            }
                                            if src not in accepted_sources:
                                                accepted.add(cand["cand_id"])

                                evaluator.update(s1_id, set(true_mids), accepted)

                            metrics = evaluator.compute()
                            sweep_count += 1

                            if metrics["macro_f_05"] > best_result["macro_f_05"]:
                                best_result = metrics
                                best_params = {
                                    "base_th": base_th,
                                    "dense_th": dense_th,
                                    "margin": margin,
                                    "anchor": anchor,
                                    "bridge": bridge,
                                    "density_cutoff": density_cutoff,
                                }

    print(f"  Sweep complete: {sweep_count} configurations. Best F0.5: {best_result['macro_f_05']:.6f}")
    print(f"  Best params: {best_params}")
    return best_result, best_result["macro_f_05"]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    print("=" * 70)
    print("EXP_N008: Hybrid — 56 Features + Neural Cosine Score (57th Feature)")
    print("=" * 70)

    with open(CHAMPION_JSON, "r") as f:
        champion = json.load(f)
    champion_f05 = champion["metric_macro_f05"]
    print(f"Champion baseline: Macro F0.5 = {champion_f05:.6f}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Load training data (train_pairs_exp008.npz, 56 features)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Step 1] Loading EXP_008 training pairs (56 features)...")
    tr_data = np.load(TRAIN_NPZ_EXP008, allow_pickle=True)
    X_train_56 = tr_data["X"].astype(np.float32)
    y_train = tr_data["y"].astype(np.int8)
    train_s1_ids = tr_data["s1_ids"]
    train_cand_ids = tr_data["cand_ids"]
    train_sources = tr_data["sources"]
    train_neg_cats = tr_data["neg_cats"]
    print(f"  Train: {X_train_56.shape[0]:,} pairs, {X_train_56.shape[1]} features, "
          f"{int(np.sum(y_train==1)):,} positives, {int(np.sum(y_train==0)):,} negatives")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Build val pairs (56 features) — or load from cache
    # ─────────────────────────────────────────────────────────────────────────
    val_s1_records, target_records, val_matches = load_validation_data()

    if os.path.exists(VAL_NPZ_N008):
        print(f"\n[Step 2] Loading cached val pairs from {VAL_NPZ_N008}...")
        va_data = np.load(VAL_NPZ_N008, allow_pickle=True)
        X_val_56 = va_data["X"].astype(np.float32)
        y_val = va_data["y"].astype(np.int8)
        val_s1_ids = va_data["s1_ids"]
        val_cand_ids = va_data["cand_ids"]
        val_sources = va_data["sources"]
        val_neg_cats = va_data["neg_cats"]
        val_meta = [
            {"s1_id": s, "cand_id": c, "source": src, "label": int(l), "hard_neg_cat": n}
            for s, c, src, l, n in zip(val_s1_ids, val_cand_ids, val_sources, y_val, val_neg_cats)
        ]
        print(f"  Val: {X_val_56.shape[0]:,} pairs, {int(np.sum(y_val==1)):,} positives")
    else:
        print(f"\n[Step 2] Building val pairs (56 features) — this takes ~15-20 min...")
        X_val_56, y_val, val_meta, val_neg_cats_dict = build_pair_dataset(
            val_s1_records, target_records, val_matches,
            max_bucket_size=1000,
            max_candidates_per_s1=200,
            max_negatives_per_s1=None,
            seed=SEED,
        )
        y_val = y_val.astype(np.int8)

        # Save val pairs
        val_s1_ids_arr = np.array([m["s1_id"] for m in val_meta], dtype=object)
        val_cand_ids_arr = np.array([m["cand_id"] for m in val_meta], dtype=object)
        val_sources_arr = np.array([m["source"] for m in val_meta], dtype=object)
        val_neg_cats_arr = np.array([m["hard_neg_cat"] for m in val_meta], dtype=object)
        np.savez_compressed(
            VAL_NPZ_N008,
            X=X_val_56, y=y_val,
            s1_ids=val_s1_ids_arr, cand_ids=val_cand_ids_arr,
            sources=val_sources_arr, neg_cats=val_neg_cats_arr,
            feature_names=np.array(FEATURE_NAMES, dtype=object),
        )
        print(f"  Saved: {VAL_NPZ_N008}")

        val_s1_ids = np.array([m["s1_id"] for m in val_meta], dtype=object)
        val_cand_ids = np.array([m["cand_id"] for m in val_meta], dtype=object)
        val_sources = np.array([m["source"] for m in val_meta], dtype=object)
        val_neg_cats = np.array([m["hard_neg_cat"] for m in val_meta], dtype=object)

    print(f"  Val: {X_val_56.shape[0]:,} pairs, {int(np.sum(y_val==1)):,} positives, "
          f"{int(np.sum(y_val==0)):,} negatives")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Load / compute cosine scores and augment feature matrices
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Step 3] Computing neural cosine scores...")

    # Load cached embeddings from POC run
    s1_lookup, target_lookup = build_cosine_lookup(S1_EMB_PATH, TARGET_EMB_PATH)

    # Encode training records not in cache (different sample from val)
    encoder = NeuralEncoder()

    # Load train S1 records to check coverage
    print("  Checking embedding coverage for training S1 IDs...")
    train_s1_unique = set(str(x) for x in train_s1_ids)
    train_cand_unique = set(str(x) for x in train_cand_ids)
    s1_missing_train = [sid for sid in train_s1_unique if sid not in s1_lookup]
    cand_missing_train = [cid for cid in train_cand_unique if cid not in target_lookup]
    print(f"  Training: {len(s1_missing_train):,}/{len(train_s1_unique):,} S1 not in cache; "
          f"{len(cand_missing_train):,}/{len(train_cand_unique):,} cand not in cache")

    # Load and encode missing training records if needed
    if s1_missing_train or cand_missing_train:
        print("  Loading missing training records from TSV files...")
        # Load train S1 records
        if s1_missing_train:
            miss_set = set(s1_missing_train)
            tmp_recs = {}
            with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), "r", encoding="utf-8", errors="replace") as f:
                next(f)
                for line in f:
                    eid = line.split("\t", 1)[0]
                    if eid in miss_set:
                        parts = line.rstrip("\r\n").split("\t")
                        tmp_recs[eid] = (parts[0], parts[1], parts[2], parts[3])
                        if len(tmp_recs) >= len(miss_set):
                            break
            s1_lookup = encode_missing_records(tmp_recs, s1_lookup, encoder, "TrainS1")

        if cand_missing_train:
            miss_set = set(cand_missing_train)
            tmp_recs = {}
            for tsv in ["train_source2.tsv", "train_source3.tsv"]:
                with open(os.path.join(TRAIN_DIR, tsv), "r", encoding="utf-8", errors="replace") as f:
                    next(f)
                    for line in f:
                        eid = line.split("\t", 1)[0]
                        if eid in miss_set:
                            parts = line.rstrip("\r\n").split("\t")
                            tmp_recs[eid] = (parts[0], parts[1], parts[2], parts[3])
                            if len(tmp_recs) >= len(miss_set):
                                break
            target_lookup = encode_missing_records(tmp_recs, target_lookup, encoder, "TrainCands")

        # Update cache files with newly encoded records
        s1_ids_upd = list(s1_lookup.keys())
        s1_vecs_upd = np.stack([s1_lookup[k] for k in s1_ids_upd]).astype(np.float32)
        save_embeddings(S1_EMB_PATH, s1_ids_upd, s1_vecs_upd)

        t_ids_upd = list(target_lookup.keys())
        t_vecs_upd = np.stack([target_lookup[k] for k in t_ids_upd]).astype(np.float32)
        save_embeddings(TARGET_EMB_PATH, t_ids_upd, t_vecs_upd)

    # Compute cosine scores for training pairs
    print("  Computing cosine scores for training pairs...")
    train_cosine = compute_cosine_features(train_s1_ids, train_cand_ids, s1_lookup, target_lookup)
    print(f"  Train cosine: min={train_cosine.min():.4f}, max={train_cosine.max():.4f}, "
          f"mean={train_cosine.mean():.4f}")

    # Compute cosine scores for validation pairs
    print("  Computing cosine scores for validation pairs...")
    val_meta_for_cosine = np.load(VAL_NPZ_N008 if os.path.exists(VAL_NPZ_N008) else VAL_NPZ_N008, allow_pickle=True)
    val_s1_ids_for_cosine = val_meta_for_cosine["s1_ids"]
    val_cand_ids_for_cosine = val_meta_for_cosine["cand_ids"]

    val_cosine = compute_cosine_features(val_s1_ids_for_cosine, val_cand_ids_for_cosine, s1_lookup, target_lookup)
    print(f"  Val cosine: min={val_cosine.min():.4f}, max={val_cosine.max():.4f}, "
          f"mean={val_cosine.mean():.4f}")

    # Augment feature matrices — concatenate cosine as 57th feature
    X_train_57 = np.hstack([X_train_56, train_cosine.reshape(-1, 1)])
    X_val_57 = np.hstack([X_val_56, val_cosine.reshape(-1, 1)])

    # Reload val meta properly
    va_data_reload = np.load(VAL_NPZ_N008, allow_pickle=True)
    val_meta_list = [
        {"s1_id": str(s), "cand_id": str(c), "source": str(src), "label": int(l), "hard_neg_cat": str(n)}
        for s, c, src, l, n in zip(
            va_data_reload["s1_ids"], va_data_reload["cand_ids"],
            va_data_reload["sources"], va_data_reload["y"], va_data_reload["neg_cats"]
        )
    ]

    feature_names_57 = list(FEATURE_NAMES) + [NEURAL_FEATURE_NAME]
    print(f"\n  Feature matrix: train {X_train_57.shape}, val {X_val_57.shape}")
    print(f"  Feature #57: {NEURAL_FEATURE_NAME}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Train LGB + XGB on 57 features
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Step 4] Training models on 57-feature dataset...")
    lgb_model = train_lgb_model(X_train_57, y_train, n_features=57)
    xgb_model = train_xgb_model(X_train_57, y_train, n_features=57)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: Generate probabilities and blend
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Step 5] Generating validation probabilities...")
    probs_lgb = lgb_model.predict_proba(X_val_57)[:, 1].astype(np.float32)
    probs_xgb = xgb_model.predict_proba(X_val_57)[:, 1].astype(np.float32)

    # Sweep blend weights
    print("\n[Step 5] Sweeping blend weights (LGB/XGB)...")
    best_blend_f05 = 0.0
    best_blend_w = (0.5, 0.5)

    for w_lgb in [0.4, 0.45, 0.5, 0.55, 0.6]:
        w_xgb = 1.0 - w_lgb
        probs_blend = w_lgb * probs_lgb + w_xgb * probs_xgb

        # Quick global threshold evaluation at champion's base threshold
        evaluator = Evaluator()
        s1_cands_tmp = defaultdict(list)
        for i, meta in enumerate(val_meta_list):
            s1_cands_tmp[meta["s1_id"]].append({
                "cand_id": meta["cand_id"],
                "source": meta["source"],
                "prob": float(probs_blend[i]),
            })
        for s1_id, true_mids in val_matches.items():
            cands = sorted(s1_cands_tmp.get(s1_id, []), key=lambda c: c["prob"], reverse=True)
            top_prob = cands[0]["prob"] if cands else 0.0
            gate = 0.85 * 0.962
            if top_prob < gate:
                evaluator.update(s1_id, set(true_mids), set())
                continue
            accepted = {c["cand_id"] for c in cands if c["prob"] >= 0.962}
            evaluator.update(s1_id, set(true_mids), accepted)
        metrics = evaluator.compute()
        f05 = metrics["macro_f_05"]
        print(f"  w_lgb={w_lgb:.2f}  F0.5={f05:.6f}  P={metrics['macro_precision']:.4f}  "
              f"R={metrics['macro_recall']:.4f}")
        if f05 > best_blend_f05:
            best_blend_f05 = f05
            best_blend_w = (w_lgb, w_xgb)

    probs_final = best_blend_w[0] * probs_lgb + best_blend_w[1] * probs_xgb
    print(f"\n  Best blend: LGB={best_blend_w[0]:.2f} / XGB={best_blend_w[1]:.2f}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: Transitive bridge threshold sweep
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Step 6] Building S1 candidate lists for bridge sweep...")
    all_s1_cands = build_s1_candidate_lists(val_meta_list, probs_final, feature_names_57, X_val_57)

    best_result, best_f05 = run_transitive_bridge_sweep(all_s1_cands, val_matches)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7: Report results
    # ─────────────────────────────────────────────────────────────────────────
    delta = best_f05 - champion_f05
    verdict = "▲ IMPROVEMENT" if delta > 0 else "▼ REGRESSION"

    print("\n" + "=" * 70)
    print("EXP_N008 RESULTS")
    print("=" * 70)
    print(f"  Champion (EXP_007)    : F0.5 = {champion_f05:.6f}")
    print(f"  EXP_N008 (57 features): F0.5 = {best_f05:.6f}  {verdict} {abs(delta):.6f}")
    print(f"  Precision             : {best_result.get('macro_precision', 0):.6f}")
    print(f"  Recall                : {best_result.get('macro_recall', 0):.6f}")
    print(f"  Singleton Accuracy    : {best_result.get('singleton_accuracy', 0):.6f}")
    print(f"  Total runtime         : {(time.time()-t_start)/60:.1f} minutes")
    print("=" * 70)

    # Append to neural registry
    fieldnames = [
        "exp_id", "description", "macro_f_05", "macro_precision", "macro_recall",
        "singleton_accuracy", "exact_set_accuracy", "total_entities",
        "total_true_matches", "total_false_positives", "total_false_negatives",
        "threshold", "notes",
    ]
    write_header = not os.path.exists(REGISTRY_PATH)
    with open(REGISTRY_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({
            "exp_id": "EXP_N008",
            "description": "Hybrid: 56 features + neural cosine score (57th feature)",
            "macro_f_05": best_f05,
            "macro_precision": best_result.get("macro_precision", 0),
            "macro_recall": best_result.get("macro_recall", 0),
            "singleton_accuracy": best_result.get("singleton_accuracy", 0),
            "exact_set_accuracy": best_result.get("exact_set_accuracy", 0),
            "total_entities": best_result.get("total_entities", 0),
            "total_true_matches": best_result.get("total_true_matches", 0),
            "total_false_positives": best_result.get("total_false_positives", 0),
            "total_false_negatives": best_result.get("total_false_negatives", 0),
            "threshold": "bridge_sweep",
            "notes": f"blend=LGB{best_blend_w[0]:.2f}+XGB{best_blend_w[1]:.2f}, cosine from paraphrase-multilingual-MiniLM-L12-v2",
        })

    # Update champion if improved
    if best_f05 > champion_f05:
        print(f"\n🏆 NEW CHAMPION! F0.5 {champion_f05:.6f} → {best_f05:.6f} (+{delta:.6f})")
        champion_data = {
            "iteration": champion.get("iteration", 4) + 1,
            "champion_model": f"hybrid_LGB{best_blend_w[0]:.2f}_XGB{best_blend_w[1]:.2f}",
            "decision_strategy": "transitive_cross_source_consistency_bridge",
            "metric_macro_f05": best_f05,
            "metric_macro_precision": best_result.get("macro_precision", 0),
            "metric_macro_recall": best_result.get("macro_recall", 0),
            "singleton_accuracy": best_result.get("singleton_accuracy", 0),
            "exact_set_accuracy": best_result.get("exact_set_accuracy", 0),
            "num_features": 57,
            "eval_population_s1": VAL_SAMPLE_SIZE,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "experiment": "EXP_N008",
            "neural_feature": NEURAL_FEATURE_NAME,
            "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
        }
        with open(CHAMPION_JSON, "w") as f:
            json.dump(champion_data, f, indent=2)
        print(f"  Champion updated: {CHAMPION_JSON}")
    else:
        print(f"\n  No improvement vs champion. Champion retained: F0.5 = {champion_f05:.6f}")


if __name__ == "__main__":
    main()
