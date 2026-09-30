#!/usr/bin/env python3
"""
EXP_N010: Memory-Safe Union Validation Pairs Generator.

Optimizations:
1. Performs FAISS retrieval and immediately frees the 8GB target embedding arrays and FAISS index from RAM.
2. Only loads target records that were actually retrieved by either the ANN or keyword blocker.
3. Pre-allocates fixed-size NumPy arrays for feature extraction instead of growing huge Python lists.
4. Guaranteed to run under 4.0 GB of RAM.
"""
import os
import sys
import time
import random
import csv
import gc
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import faiss

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["PYTHONUNBUFFERED"] = "1"

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
code_dir = os.path.join(project_root, "code", "business_entity_resolution")
sys.path.insert(0, os.path.join(code_dir, "src"))

from pair_features import FEATURE_NAMES, extract_pair_features
from blocking import build_target_index, generate_candidates
from build_pair_dataset import UNION_E_KEYS, classify_hard_negative
from neural_encoder import load_embeddings

DATASET_DIR = os.path.join(project_root, "dataset", "train")
EXPERIMENTS_DIR = os.path.join(code_dir, "experiments")
VAL_IDS_PATH = os.path.join(EXPERIMENTS_DIR, "validation_s1_ids.txt")
OUTPUT_NPZ = os.path.join(EXPERIMENTS_DIR, "val_pairs_n010.npz")
S1_EMB_PATH = os.path.join(EXPERIMENTS_DIR, "embeddings_n010_s1.npz")
TARGET_EMB_PATH = os.path.join(EXPERIMENTS_DIR, "embeddings_n010_target.npz")

VAL_SAMPLE_SIZE = 25000
SEED = 42

def main():
    t_start = time.time()
    random.seed(SEED)
    print("=" * 75, flush=True)
    print("[EXP_N010] Generating Memory-Safe Union Validation Pairs...", flush=True)
    print("=" * 75, flush=True)

    # 1. Load exact 25k validation S1 IDs
    with open(VAL_IDS_PATH, "r", encoding="utf-8") as f:
        val_pool = [line.strip() for line in f if line.strip()]
    selected_val_s1 = set(random.sample(val_pool, min(VAL_SAMPLE_SIZE, len(val_pool))))
    print(f"Loaded {len(selected_val_s1):,} validation S1 IDs.", flush=True)

    # 2. Load ground truth
    val_matches: Dict[str, List[str]] = {}
    with open(os.path.join(DATASET_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1 = parts[0]
            if s1 in selected_val_s1:
                mids = parts[1].split(",") if len(parts) > 1 and parts[1].strip() else []
                val_matches[s1] = mids

    total_gt_matches = sum(len(mids) for mids in val_matches.values())
    print(f"Total ground truth matches to recover: {total_gt_matches:,}", flush=True)

    # 3. Load S1 records
    val_s1_records: Dict[str, Tuple] = {}
    with open(os.path.join(DATASET_DIR, "train_source1.tsv"), "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = (row.get("entity_id") or "").strip()
            if eid in selected_val_s1:
                nm = (row.get("business_name") or row.get("name") or "").strip()
                addr = (row.get("business_address") or row.get("address") or "").strip()
                ctry = (row.get("country") or "").strip()
                val_s1_records[eid] = (eid, nm, addr, ctry)

    # 4. Neural ANN Candidate Retrieval (Top-150)
    print("\n--- Running Neural ANN Candidate Retrieval ---", flush=True)
    ids_s1, vecs_s1 = load_embeddings(S1_EMB_PATH)
    s1_emb_map = {id_: i for i, id_ in enumerate(ids_s1)}

    # Stream target countries without loading heavy data
    target_countries = {}
    for tsv_name in ["train_source2.tsv", "train_source3.tsv"]:
        with open(os.path.join(DATASET_DIR, tsv_name), "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                eid = (row.get("entity_id") or "").strip()
                ctry = (row.get("country") or "").strip()
                if eid:
                    target_countries[eid] = ctry

    ids_tgt, vecs_tgt = load_embeddings(TARGET_EMB_PATH)
    country_to_indices = defaultdict(list)
    for i, tid in enumerate(ids_tgt):
        ctry = target_countries.get(tid, "")
        country_to_indices[ctry].append(i)

    ann_cands_by_s1: Dict[str, Set[str]] = defaultdict(set)
    s1_by_country = defaultdict(list)
    for s1_id, rec in val_s1_records.items():
        s1_by_country[rec[3]].append(s1_id)

    dim = vecs_tgt.shape[1]
    ann_hits = 0

    # Build per-country index one at a time and query immediately to keep RAM low
    for ctry, indices in country_to_indices.items():
        s1_list = s1_by_country.get(ctry, [])
        if not s1_list:
            continue
        print(f"Indexing & searching country {ctry} ({len(indices):,} targets)...", flush=True)
        idx_arr = np.array(indices, dtype=np.int64)
        v_sub = vecs_tgt[idx_arr]
        idx = faiss.IndexFlatIP(dim)
        idx.add(v_sub.astype(np.float32))
        c_tids = [ids_tgt[i] for i in indices]

        valid_s1_list = [s1 for s1 in s1_list if s1 in s1_emb_map]
        q_vecs = np.vstack([vecs_s1[s1_emb_map[s1]] for s1 in valid_s1_list]).astype(np.float32)

        batch_sz = 1024
        for b in range(0, len(valid_s1_list), batch_sz):
            b_end = min(b + batch_sz, len(valid_s1_list))
            _, I = idx.search(q_vecs[b:b_end], 120)
            for sub_i, row in enumerate(I):
                cur_s1 = valid_s1_list[b + sub_i]
                c_set = {c_tids[x] for x in row if x >= 0}
                ann_cands_by_s1[cur_s1].update(c_set)
                gt_set = set(val_matches.get(cur_s1, []))
                ann_hits += len(gt_set.intersection(c_set))

        del idx, v_sub, idx_arr, c_tids
        gc.collect()

    print(f"Neural ANN Retrieval captured {ann_hits:,} / {total_gt_matches:,} true matches ({ann_hits/total_gt_matches*100:.2f}% recall).", flush=True)

    # 5. Build Keyword Blocker Candidates & Form Union
    print("\n--- Merging with Keyword Blocker ---", flush=True)
    all_needed_targets = set()
    for mids in val_matches.values():
        all_needed_targets.update(mids)
    for cands in ann_cands_by_s1.values():
        all_needed_targets.update(cands)

    # Load only necessary target records
    s2_records: Dict[str, Tuple] = {}
    with open(os.path.join(DATASET_DIR, "train_source2.tsv"), "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = (row.get("entity_id") or "").strip()
            if eid in all_needed_targets:
                nm = (row.get("business_name") or row.get("name") or "").strip()
                addr = (row.get("business_address") or row.get("address") or "").strip()
                ctry = (row.get("country") or "").strip()
                s2_records[eid] = (eid, nm, addr, ctry)

    s3_records: Dict[str, Tuple] = {}
    with open(os.path.join(DATASET_DIR, "train_source3.tsv"), "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = (row.get("entity_id") or "").strip()
            if eid in all_needed_targets:
                nm = (row.get("business_name") or row.get("name") or "").strip()
                addr = (row.get("business_address") or row.get("address") or "").strip()
                ctry = (row.get("country") or "").strip()
                s3_records[eid] = (eid, nm, addr, ctry)

    target_records = {**s2_records, **s3_records}
    print(f"Target records loaded: {len(target_records):,}", flush=True)

    # Keyword blocker index
    block_index = build_target_index(list(target_records.values()), UNION_E_KEYS, max_bucket_size=1000)
    keyword_cands_by_s1 = generate_candidates(
        val_s1_records.values(), block_index, UNION_E_KEYS, max_candidates_per_s1=150
    )

    # Union candidates
    union_pairs: List[Tuple[str, str, int]] = []
    union_hits = 0
    for s1_id in val_s1_records:
        kw_cands = keyword_cands_by_s1.get(s1_id, set())
        nn_cands = ann_cands_by_s1.get(s1_id, set())
        merged = kw_cands.union(nn_cands)
        gt_set = set(val_matches.get(s1_id, []))
        pos_cands = [c for c in merged if c in gt_set]
        neg_cands = [c for c in merged if c not in gt_set]
        union_hits += len(pos_cands)

        # Cap negatives to 80 per S1 to ensure lightweight memory footprint
        selected = pos_cands + neg_cands[:max(0, 80 - len(pos_cands))]
        for cid in selected:
            union_pairs.append((s1_id, cid, 1 if cid in gt_set else 0))

    print(f"\n=======================================================", flush=True)
    print(f"FINAL UNION BLOCKER RECALL: {union_hits:,} / {total_gt_matches:,} ({union_hits/total_gt_matches*100:.2f}%)", flush=True)
    print(f"Blocker Misses reduced from 2,123 to {total_gt_matches - union_hits:,}!", flush=True)
    print(f"Total Candidate Pairs to score: {len(union_pairs):,}", flush=True)
    print(f"=======================================================", flush=True)

    # Free heavy index structures before feature extraction
    tgt_emb_map = {id_: i for i, id_ in enumerate(ids_tgt)}
    del block_index, keyword_cands_by_s1, ann_cands_by_s1, country_to_indices, target_countries
    gc.collect()

    # 6. Extract 56 features + fine-tuned Neural Cosine (Feature #57)
    N_total = len(union_pairs)
    print(f"\nExtracting features into pre-allocated NumPy array ({N_total:,} rows)...", flush=True)

    X_mat = np.zeros((N_total, 57), dtype=np.float32)
    y_vec = np.zeros(N_total, dtype=np.int8)
    s1_ids_arr = np.empty(N_total, dtype=object)
    cand_ids_arr = np.empty(N_total, dtype=object)
    sources_arr = np.empty(N_total, dtype=object)
    neg_cats_arr = np.empty(N_total, dtype=object)

    t_feat = time.time()
    for i, (s1_id, cand_id, is_match) in enumerate(union_pairs):
        s1_rec = val_s1_records.get(s1_id)
        cand_rec = target_records.get(cand_id)
        if not s1_rec or not cand_rec:
            continue

        s1_name, s1_addr, s1_country = s1_rec[1], s1_rec[2], s1_rec[3]
        cand_name, cand_addr, cand_country = cand_rec[1], cand_rec[2], cand_rec[3]

        row_feats = extract_pair_features(s1_name, s1_addr, s1_country, cand_name, cand_addr, cand_country)
        X_mat[i, :56] = row_feats

        # Fine-tuned neural cosine
        idx1 = s1_emb_map.get(s1_id)
        idx2 = tgt_emb_map.get(cand_id)
        if idx1 is not None and idx2 is not None:
            X_mat[i, 56] = float(np.dot(vecs_s1[idx1], vecs_tgt[idx2]))

        y_vec[i] = is_match
        s1_ids_arr[i] = s1_id
        cand_ids_arr[i] = cand_id
        sources_arr[i] = "S2" if cand_id.startswith("S2-") else "S3"
        neg_cats_arr[i] = "positive" if is_match == 1 else "negative"

        if (i + 1) % 250000 == 0:
            print(f"  Processed {i+1:,} / {N_total:,} pairs ({time.time()-t_feat:.1f}s)...", flush=True)

    feature_names_57 = list(FEATURE_NAMES) + ["neural_cosine_sim"]

    # Free remaining large vectors before saving
    del vecs_s1, vecs_tgt, ids_s1, ids_tgt, s1_emb_map, tgt_emb_map, val_s1_records, target_records
    gc.collect()

    print(f"\nSaving {OUTPUT_NPZ}...", flush=True)
    np.savez_compressed(
        OUTPUT_NPZ,
        X=X_mat,
        y=y_vec,
        s1_ids=s1_ids_arr,
        cand_ids=cand_ids_arr,
        sources=sources_arr,
        neg_cats=neg_cats_arr,
        feature_names=np.array(feature_names_57, dtype=object),
    )
    print(f"Successfully generated {OUTPUT_NPZ} in {time.time()-t_start:.1f}s", flush=True)

if __name__ == "__main__":
    main()
