#!/usr/bin/env python3
"""
EXP_N017 — Dataset Builder for XLM-RoBERTa Cross-Encoder.
Constructs hard-negative balanced training pairs and candidate validation pairs
formatted with explicit field boundaries:
  Text A: BUSINESS_NAME [SEP] ADDRESS [SEP] COUNTRY
  Text B: BUSINESS_NAME [SEP] ADDRESS [SEP] COUNTRY

Hard Negative Mining:
- 1_same_name_diff_addr, 2_similar_name_diff_addr
- 5_same_house_similar_name, 10_diff_business_same_addr
- 6_generic_name_collision, 7_short_name_collision, 9_transliteration_collision
- High-cosine ANN mined negatives from hard_neg_manifest.csv
- Margin negatives around decision boundary (prob >= 0.70)
"""

from __future__ import annotations

import csv
import gc
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONUNBUFFERED"] = "1"
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "dataset" / "train"
EXP_DIR = CODE_DIR / "experiments"
EXP_N017_DIR = EXP_DIR / "EXP_N017"
EXP_N017_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def load_raw_records_for_ids(needed_ids: Set[str]) -> Dict[str, Tuple[str, str, str]]:
    """Stream TSVs and return {id: (business_name, business_address, country)}."""
    t0 = time.time()
    records: Dict[str, Tuple[str, str, str]] = {}
    remaining = set(needed_ids)
    print(f"[RecordLoader] Need {len(needed_ids):,} unique IDs across sources...", flush=True)

    for sf in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]:
        path = DATA_DIR / sf
        if not path.exists() or not remaining:
            continue
        print(f"  Streaming {sf}...", flush=True)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            for row in reader:
                if not row:
                    continue
                eid = row[0].strip()
                if eid in remaining:
                    name = row[1].strip() if len(row) > 1 else ""
                    addr = row[2].strip() if len(row) > 2 else ""
                    ctry = row[3].strip() if len(row) > 3 else ""
                    records[eid] = (name, addr, ctry)
                    remaining.remove(eid)
                    if not remaining:
                        break

    print(f"[RecordLoader] Successfully loaded {len(records):,} records in {time.time()-t0:.1f}s", flush=True)
    return records


def build_training_set():
    print("=" * 75)
    print("[EXP_N017] Step 1: Building Balanced Cross-Encoder Training Dataset")
    print("=" * 75)

    # 1. Load base training pairs from train_pairs_exp008.npz
    tp = np.load(EXP_DIR / "train_pairs_exp008.npz", allow_pickle=True)
    s1_all = tp["s1_ids"].astype(str)
    cand_all = tp["cand_ids"].astype(str)
    y_all = tp["y"].astype(np.int8)
    cats_all = tp["neg_cats"].astype(str)

    # Positives: take all 49,735 true matches
    pos_idx = np.where(y_all == 1)[0]
    print(f"Total True Positives in training split: {len(pos_idx):,}")

    # High-value structured hard negatives from categories
    # 1_same_name_diff_addr, 2_similar_name_diff_addr, 5_same_house_similar_name, 10_diff_business_same_addr, etc.
    hard_cat_mask = np.isin(cats_all, [
        "1_same_name_diff_addr",
        "2_similar_name_diff_addr",
        "5_same_house_similar_name",
        "10_diff_business_same_addr",
        "6_generic_name_collision",
        "7_short_name_collision",
        "9_transliteration_collision",
    ])
    cat_hard_idx = np.where(hard_cat_mask)[0]
    print(f"High-value structured category hard negatives: {len(cat_hard_idx):,}")

    # Rare token negatives
    rare_idx = np.where(cats_all == "8_same_rare_token_diff_entity")[0]
    rare_sample = np.random.choice(rare_idx, size=min(15000, len(rare_idx)), replace=False)
    print(f"Sampled rare-token negatives: {len(rare_sample):,}")

    # Blocker candidate negatives
    blocker_idx = np.where(cats_all == "11_other_blocker_candidate")[0]
    blocker_sample = np.random.choice(blocker_idx, size=min(10000, len(blocker_idx)), replace=False)
    print(f"Sampled blocker candidate negatives: {len(blocker_sample):,}")

    # 2. Sample ANN hard negatives from hard_neg_manifest.csv
    print("Sampling high-cosine ANN hard negatives from hard_neg_manifest.csv...", flush=True)
    df_man = pd.read_csv(EXP_DIR / "hard_neg_manifest.csv")
    df_man_hard = df_man.sort_values(by="score", ascending=False).head(20000)
    ann_s1 = df_man_hard["s1_id"].values.astype(str)
    ann_cand = df_man_hard["neg_id"].values.astype(str)
    ann_y = np.zeros(len(ann_s1), dtype=np.int8)
    print(f"Sampled ANN hard negatives: {len(ann_s1):,} (scores: {df_man_hard['score'].min():.4f} to {df_man_hard['score'].max():.4f})")

    # Combine training sets
    train_s1 = np.concatenate([s1_all[pos_idx], s1_all[cat_hard_idx], s1_all[rare_sample], s1_all[blocker_sample], ann_s1])
    train_cand = np.concatenate([cand_all[pos_idx], cand_all[cat_hard_idx], cand_all[rare_sample], cand_all[blocker_sample], ann_cand])
    train_y = np.concatenate([y_all[pos_idx], y_all[cat_hard_idx], y_all[rare_sample], y_all[blocker_sample], ann_y])

    # Shuffle training set
    perm = np.random.permutation(len(train_y))
    train_s1 = train_s1[perm]
    train_cand = train_cand[perm]
    train_y = train_y[perm]

    print(f"\nFinal Assembled Training Set: {len(train_y):,} pairs")
    print(f"  Positives: {np.sum(train_y == 1):,} ({np.mean(train_y == 1):.2%})")
    print(f"  Negatives: {np.sum(train_y == 0):,} ({np.mean(train_y == 0):.2%})")

    # Save manifest of training pairs
    train_manifest_path = EXP_N017_DIR / "train_pairs_manifest.npz"
    np.savez_compressed(
        train_manifest_path,
        s1_ids=train_s1,
        cand_ids=train_cand,
        y=train_y,
    )
    print(f"Saved training manifest to: {train_manifest_path}")
    return train_s1, train_cand, train_y


def build_validation_candidate_manifest():
    print("\n" + "=" * 75)
    print("[EXP_N017] Step 2: Building Validation Candidate Manifest for Cross-Encoder")
    print("=" * 75)

    va = np.load(EXP_DIR / "val_pairs_n010.npz", allow_pickle=True)
    pl = np.load(EXP_DIR / "probs_n011_optimal_lgb.npy")
    px = np.load(EXP_DIR / "probs_n011_optimal_xgb.npy")
    probs = 0.45 * pl + 0.55 * px
    s1_ids = va["s1_ids"].astype(str)
    cand_ids = va["cand_ids"].astype(str)
    sources = va["sources"].astype(str)
    y = va["y"].astype(np.int8)

    unique_s1, s1_codes = np.unique(s1_ids, return_inverse=True)
    sort_key = np.lexsort((-probs, s1_codes))
    s1_codes_s = s1_codes[sort_key]
    probs_s = probs[sort_key]
    cand_ids_s = cand_ids[sort_key]
    sources_s = sources[sort_key]
    y_s = y[sort_key]

    boundaries = np.where(np.diff(s1_codes_s) != 0)[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(probs_s)]])

    # Select candidates for scoring:
    # 1. Any candidate with prob >= 0.70
    # 2. Top-1 candidate for every active entity
    val_select_mask = np.zeros(len(probs), dtype=bool)
    for k in range(len(unique_s1)):
        sl = slice(starts[k], ends[k])
        ps = probs_s[sl]
        if len(ps) > 0 and ps[0] >= 0.50:
            c_mask = (ps >= 0.70)
            c_mask[0] = True
            val_select_mask[sort_key[starts[k] : ends[k]][c_mask]] = True

    val_indices = np.where(val_select_mask)[0]
    print(f"Validation Pairs Selected for Transformer Cross-Encoder Scoring: {len(val_indices):,}")
    print(f"  Positives: {np.sum(y[val_indices] == 1):,}")
    print(f"  Negatives: {np.sum(y[val_indices] == 0):,}")

    val_manifest_path = EXP_N017_DIR / "val_scoring_manifest.npz"
    np.savez_compressed(
        val_manifest_path,
        val_pair_indices=val_indices,
        s1_ids=s1_ids[val_indices],
        cand_ids=cand_ids[val_indices],
        y=y[val_indices],
        n014_probs=probs[val_indices],
    )
    print(f"Saved validation manifest to: {val_manifest_path}")
    return s1_ids[val_indices], cand_ids[val_indices]


def build_raw_record_store(train_s1, train_cand, val_s1, val_cand):
    print("\n" + "=" * 75)
    print("[EXP_N017] Step 3: Loading and Caching Raw Text Records")
    print("=" * 75)

    needed_ids = set(train_s1) | set(train_cand) | set(val_s1) | set(val_cand)
    print(f"Total Unique Records Needed: {len(needed_ids):,}")

    records = load_raw_records_for_ids(needed_ids)

    # Format texts as: BUSINESS_NAME [SEP] ADDRESS [SEP] COUNTRY
    print("Formatting field strings with [SEP] delimiters...", flush=True)
    eid_list = []
    text_list = []
    for eid, (nm, addr, ctry) in records.items():
        eid_list.append(eid)
        # Format string
        formatted = f"{nm} [SEP] {addr} [SEP] {ctry}"
        text_list.append(formatted)

    record_store_path = EXP_N017_DIR / "formatted_record_store.npz"
    np.savez_compressed(
        record_store_path,
        eids=np.array(eid_list),
        texts=np.array(text_list),
    )
    print(f"Saved {len(eid_list):,} formatted records to: {record_store_path}")


def main():
    t_start = time.time()
    train_s1, train_cand, train_y = build_training_set()
    val_s1, val_cand = build_validation_candidate_manifest()
    build_raw_record_store(train_s1, train_cand, val_s1, val_cand)
    print("\n" + "=" * 75)
    print(f"[EXP_N017] Data Preparation Complete in {time.time()-t_start:.1f}s")
    print("=" * 75)


if __name__ == "__main__":
    main()
