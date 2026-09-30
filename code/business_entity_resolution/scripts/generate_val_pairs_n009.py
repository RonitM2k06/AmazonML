#!/usr/bin/env python3
"""
Generate EXP_N009 Validation Pairs using upgraded Blocker.
Incorporates:
- Upgraded candidate keys (domain stems, normalized house numbers, expanded address pairs)
- 56 hand-crafted pairwise features
- 57th feature: Multilingual MiniLM cosine similarity computed directly from cached embeddings
Saves as: experiments/val_pairs_n009.npz
"""
import os
import sys
import time
import random
from typing import Dict, List, Set, Tuple
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["PYTHONUNBUFFERED"] = "1"

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
code_dir = os.path.join(project_root, "code", "business_entity_resolution")
sys.path.insert(0, os.path.join(code_dir, "src"))

from pair_features import FEATURE_NAMES
from build_pair_dataset import UNION_E_KEYS, build_pair_dataset
from neural_encoder import load_embeddings

DATASET_DIR = os.path.join(project_root, "dataset", "train")
EXPERIMENTS_DIR = os.path.join(code_dir, "experiments")
VAL_IDS_PATH = os.path.join(EXPERIMENTS_DIR, "validation_s1_ids.txt")
OUTPUT_NPZ = os.path.join(EXPERIMENTS_DIR, "val_pairs_n009.npz")
S1_EMB_PATH = os.path.join(EXPERIMENTS_DIR, "s1_embeddings.npz")
TARGET_EMB_PATH = os.path.join(EXPERIMENTS_DIR, "target_embeddings.npz")

VAL_SAMPLE_SIZE = 25000
SEED = 42

def main():
    t_start = time.time()
    random.seed(SEED)
    print("=" * 70, flush=True)
    print("[EXP_N009] Building Validation Dataset with Upgraded Blocker...", flush=True)
    print("=" * 70, flush=True)

    # 1. Load validation S1 IDs
    with open(VAL_IDS_PATH, "r", encoding="utf-8") as f:
        val_pool = [line.strip() for line in f if line.strip()]
    selected_val_s1 = set(random.sample(val_pool, min(VAL_SAMPLE_SIZE, len(val_pool))))
    print(f"Loaded {len(selected_val_s1):,} validation S1 IDs.", flush=True)

    # 2. Load ground truth
    val_matches: Dict[str, List[str]] = {}
    val_needed_s2: Set[str] = set()
    val_needed_s3: Set[str] = set()
    with open(os.path.join(DATASET_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
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

    # 3. Load S1 records
    val_s1_records: Dict[str, Tuple] = {}
    with open(os.path.join(DATASET_DIR, "train_source1.tsv"), "r", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts[0] in selected_val_s1:
                val_s1_records[parts[0]] = (parts[0], parts[1], parts[2], parts[3])

    # 4. Load S2 and S3 records
    s2_records: Dict[str, Tuple] = {}
    bg_s2 = 0
    with open(os.path.join(DATASET_DIR, "train_source2.tsv"), "r", encoding="utf-8", errors="replace") as f:
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
    with open(os.path.join(DATASET_DIR, "train_source3.tsv"), "r", encoding="utf-8", errors="replace") as f:
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
    print(f"Loaded: {len(val_s1_records):,} S1, {len(target_records):,} targets.", flush=True)

    # 5. Build candidate pair dataset (56 features)
    X_val_56, y_val, val_meta, val_neg_cats = build_pair_dataset(
        val_s1_records, target_records, val_matches,
        max_bucket_size=1000,
        max_candidates_per_s1=200,
        max_negatives_per_s1=None,
        seed=SEED,
    )
    y_val = y_val.astype(np.int8)
    N_pairs = X_val_56.shape[0]
    print(f"Candidate pairs generated: {N_pairs:,} ({np.sum(y_val==1):,} positives, {np.sum(y_val==0):,} negatives)", flush=True)

    # 6. Add 57th Feature: Neural Cosine Similarity from cached embeddings
    print("Loading cached embeddings to append neural_cosine_sim (feature #57)...", flush=True)
    s1_ids_emb, s1_vecs = load_embeddings(S1_EMB_PATH)
    target_ids_emb, tgt_vecs = load_embeddings(TARGET_EMB_PATH)
    s1_emb_map = {id_: i for i, id_ in enumerate(s1_ids_emb)}
    tgt_emb_map = {id_: i for i, id_ in enumerate(target_ids_emb)}

    val_s1_ids_arr = np.array([m["s1_id"] for m in val_meta], dtype=object)
    val_cand_ids_arr = np.array([m["cand_id"] for m in val_meta], dtype=object)
    val_sources_arr = np.array([m["source"] for m in val_meta], dtype=object)
    val_neg_cats_arr = np.array([m["hard_neg_cat"] for m in val_meta], dtype=object)

    cosine_sims = np.zeros(N_pairs, dtype=np.float32)
    for i in range(N_pairs):
        s1_id = val_s1_ids_arr[i]
        c_id = val_cand_ids_arr[i]
        idx1 = s1_emb_map.get(s1_id)
        idx2 = tgt_emb_map.get(c_id)
        if idx1 is not None and idx2 is not None:
            cosine_sims[i] = float(np.dot(s1_vecs[idx1], tgt_vecs[idx2]))

    X_val_57 = np.hstack([X_val_56, cosine_sims.reshape(-1, 1)]).astype(np.float32)
    feature_names_57 = list(FEATURE_NAMES) + ["neural_cosine_sim"]

    # 7. Save val_pairs_n009.npz
    print(f"Saving {OUTPUT_NPZ}...", flush=True)
    np.savez_compressed(
        OUTPUT_NPZ,
        X=X_val_57,
        y=y_val,
        s1_ids=val_s1_ids_arr,
        cand_ids=val_cand_ids_arr,
        sources=val_sources_arr,
        neg_cats=val_neg_cats_arr,
        feature_names=np.array(feature_names_57, dtype=object),
    )
    print(f"Successfully created {OUTPUT_NPZ} ({os.path.getsize(OUTPUT_NPZ)/1024/1024:.1f} MB) in {time.time()-t_start:.1f}s", flush=True)

if __name__ == "__main__":
    main()
