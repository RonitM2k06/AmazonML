"""
Candidate-Pair Dataset Generator and Hard Negative Mining Engine.
Extracts candidate pairs via Union E blocking for training and validation splits,
mines and categorizes hard negatives, extracts 54-dimensional feature vectors,
and outputs leakage-safe training and validation matrices.
"""

import os
import sys
import json
import time
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

# Ensure UTF-8 on Windows

from pair_features import (
    FEATURE_NAMES,
    extract_pair_features,
)
from blocking import (
    BlockIndex,
    build_target_index,
    generate_candidates,
    extract_blocking_keys,
    GENERIC_WORDS,
)
from normalization import (
    normalize_business_name,
    normalize_address,
    extract_postal_codes,
    extract_numbers,
    tokenize,
    character_similarity,
)

UNION_E_KEYS: Set[str] = {
    "exact_norm",
    "sorted_tokens",
    "postal_name",
    "address_name",
    "two_tokens",
    "address_signature",
    "char_ngram",
    "rare_token",
    "phonetic_skeleton",
    "address_pair",
}


def classify_hard_negative(
    s1_name: str,
    s1_addr: str,
    cand_name: str,
    cand_addr: str,
    features_dict: Dict[str, float],
) -> str:
    """
    Classify a negative candidate pair into one of 10 hard-negative categories.
    """
    name_sim = features_dict.get("name_levenshtein_sim", 0.0)
    addr_sim = features_dict.get("addr_levenshtein_sim", 0.0)
    name_exact = features_dict.get("name_norm_exact", 0.0)
    addr_exact = features_dict.get("addr_norm_exact", 0.0)
    postal_exact = features_dict.get("addr_postal_exact", 0.0)
    house_exact = features_dict.get("addr_house_num_exact", 0.0)
    rare_overlap = features_dict.get("name_rare_token_overlap", 0.0)
    translit_sim = features_dict.get("name_translit_sim", 0.0)

    # 1. Same normalized name + different address
    if name_exact == 1.0 and addr_sim < 0.35:
        return "1_same_name_diff_addr"

    # 2. Very similar name + different address
    if name_sim >= 0.85 and addr_sim < 0.35:
        return "2_similar_name_diff_addr"

    # 3. Same normalized address + different name
    if addr_exact == 1.0 and name_sim < 0.40:
        return "3_same_addr_diff_name"

    # 4. Same postal code + similar name
    if postal_exact == 1.0 and name_sim >= 0.55:
        return "4_same_postal_similar_name"

    # 5. Same house number + similar name
    if house_exact == 1.0 and name_sim >= 0.55:
        return "5_same_house_similar_name"

    # 6. Different business operating at same address
    if addr_sim >= 0.80 and name_sim < 0.40:
        return "10_diff_business_same_addr"

    # 7. Common/generic business name collision
    s1_toks = set(tokenize(normalize_business_name(s1_name)))
    if s1_toks.intersection(GENERIC_WORDS) and name_sim >= 0.60:
        return "6_generic_name_collision"

    # 8. Short-name collision
    if len(normalize_business_name(s1_name)) <= 5 and name_sim >= 0.60:
        return "7_short_name_collision"

    # 9. Same rare token + different entity
    if rare_overlap >= 1.0:
        return "8_same_rare_token_diff_entity"

    # 10. Transliteration collision
    if translit_sim >= 0.70 and name_sim < 0.50:
        return "9_transliteration_collision"

    return "11_other_blocker_candidate"


def build_pair_dataset(
    s1_records: Dict[str, Tuple[str, str, str, str]],
    target_records: Dict[str, Tuple[str, str, str, str]],
    s1_ground_truth: Dict[str, List[str]],
    max_bucket_size: int = 1000,
    max_candidates_per_s1: int = 200,
    max_negatives_per_s1: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[Dict], Dict[str, int]]:
    """
    Build candidate pair feature matrix and labels from S1 records and target records.
    Returns:
        X (np.ndarray float32): [num_pairs, 56] feature matrix
        y (np.ndarray int8): [num_pairs] binary labels (1 = match, 0 = non-match)
        pair_meta (List[Dict]): metadata for each pair (s1_id, cand_id, source, label, neg_category)
        hard_neg_counts (Dict[str, int]): counts of each negative category
    """
    t0 = time.time()
    random.seed(seed)

    print(f"Building Union F block index over {len(target_records):,} target records...")
    target_pool = list(target_records.values())
    block_index = build_target_index(target_pool, UNION_E_KEYS, max_bucket_size=max_bucket_size)

    print(f"Querying candidates for {len(s1_records):,} S1 entities (max_cands={max_candidates_per_s1})...")
    cands_by_s1 = generate_candidates(
        s1_records.values(), block_index, UNION_E_KEYS, max_candidates_per_s1=max_candidates_per_s1
    )

    pairs_X: List[List[float]] = []
    pairs_y: List[int] = []
    pairs_meta: List[Dict] = []
    hard_neg_counts: Dict[str, int] = defaultdict(int)

    total_true_matches = 0
    captured_true_matches = 0

    print("Extracting 56-dimensional feature vectors and mining hard negatives...")
    for s1_id, (s1_eid, s1_name, s1_addr, s1_country) in s1_records.items():
        true_mids = set(s1_ground_truth.get(s1_id, []))
        total_true_matches += len(true_mids)

        cands = list(cands_by_s1.get(s1_id, set()))
        if max_negatives_per_s1 is None and len(cands) > 100:
            # Cap validation candidates per S1 at 100 while strictly keeping all true positives
            pos_set = [c for c in cands if c in true_mids]
            neg_pool = [c for c in cands if c not in true_mids]
            needed_neg = max(0, 100 - len(pos_set))
            cands = set(pos_set + neg_pool[:needed_neg])
        else:
            cands = set(cands)
        pos_cands = [c for c in cands if c in true_mids]
        neg_cands = [c for c in cands if c not in true_mids]

        captured_true_matches += len(pos_cands)

        # Optional controlled negative sampling per S1 during training to manage balance
        if max_negatives_per_s1 is not None and len(neg_cands) > max_negatives_per_s1:
            neg_cands = random.sample(neg_cands, max_negatives_per_s1)

        # Process positive candidates
        for cand_id in pos_cands:
            cand_rec = target_records.get(cand_id)
            if not cand_rec:
                continue
            cand_eid, cand_name, cand_addr, cand_country = cand_rec
            feats = extract_pair_features(s1_name, s1_addr, s1_country, cand_name, cand_addr, cand_country)
            pairs_X.append(feats)
            pairs_y.append(1)
            pairs_meta.append({
                "s1_id": s1_id,
                "cand_id": cand_id,
                "source": "S2" if cand_id.startswith("S2-") else "S3",
                "label": 1,
                "hard_neg_cat": "0_positive_match",
            })

        # Process negative candidates
        for cand_id in neg_cands:
            cand_rec = target_records.get(cand_id)
            if not cand_rec:
                continue
            cand_eid, cand_name, cand_addr, cand_country = cand_rec
            feats = extract_pair_features(s1_name, s1_addr, s1_country, cand_name, cand_addr, cand_country)
            feat_dict = dict(zip(FEATURE_NAMES, feats))
            neg_cat = classify_hard_negative(s1_name, s1_addr, cand_name, cand_addr, feat_dict)
            hard_neg_counts[neg_cat] += 1

            pairs_X.append(feats)
            pairs_y.append(0)
            pairs_meta.append({
                "s1_id": s1_id,
                "cand_id": cand_id,
                "source": "S2" if cand_id.startswith("S2-") else "S3",
                "label": 0,
                "hard_neg_cat": neg_cat,
            })

    X = np.array(pairs_X, dtype=np.float32)
    y = np.array(pairs_y, dtype=np.int8)

    pos_count = int(np.sum(y == 1))
    neg_count = int(np.sum(y == 0))
    recall = round(captured_true_matches / total_true_matches, 6) if total_true_matches > 0 else 0.0

    print(f"Dataset generated in {time.time()-t0:.2f}s:")
    print(f"  Total Pairs:    {len(y):,}")
    print(f"  Positives:      {pos_count:,} ({pos_count/max(1, len(y))*100:.2f}%)")
    print(f"  Negatives:      {neg_count:,} ({neg_count/max(1, len(y))*100:.2f}%)")
    print(f"  Captured Match: {captured_true_matches:,} / {total_true_matches:,} ({recall*100:.2f}% recall)")
    print(f"  Feature Dim:    {X.shape[1]}")

    return X, y, pairs_meta, dict(hard_neg_counts)


def save_pair_dataset(
    output_path: str,
    X: np.ndarray,
    y: np.ndarray,
    pairs_meta: List[Dict],
    hard_neg_counts: Dict[str, int],
):
    """Save dataset arrays and metadata to compressed npz and json."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    s1_ids = np.array([m["s1_id"] for m in pairs_meta], dtype=object)
    cand_ids = np.array([m["cand_id"] for m in pairs_meta], dtype=object)
    sources = np.array([m["source"] for m in pairs_meta], dtype=object)
    neg_cats = np.array([m["hard_neg_cat"] for m in pairs_meta], dtype=object)

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        s1_ids=s1_ids,
        cand_ids=cand_ids,
        sources=sources,
        neg_cats=neg_cats,
        feature_names=np.array(FEATURE_NAMES, dtype=object),
    )

    meta_json_path = output_path.replace(".npz", "_meta.json")
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "num_pairs": len(y),
            "num_positives": int(np.sum(y == 1)),
            "num_negatives": int(np.sum(y == 0)),
            "feature_dim": int(X.shape[1]),
            "feature_names": FEATURE_NAMES,
            "hard_negative_breakdown": hard_neg_counts,
        }, f, indent=2)

    print(f"Saved dataset to {output_path} ({os.path.getsize(output_path)/1024/1024:.2f} MB)")
