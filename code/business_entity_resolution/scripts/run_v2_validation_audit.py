#!/usr/bin/env python3
"""
Authoritative Comparative Evaluation:
EXP_N039 (Frozen Baseline) vs V2 Production Decoder
on Entity-Disjoint Train Validation Split (25,000 entities, 86,232 ground truth matches).

Zero test data. Zero test labels. Pure generalization evaluation.
Evaluates:
- Baseline EXP_N039
- V2 Full (exact production code logic)
- V2 without max-2 cap (Ablation B)
- V2 without street requirement (Ablation C)
- V2 without missing-address restriction (Ablation D)

Computes:
1. Macro F0.5
2. Macro Precision
3. Macro Recall
4. False Positives (FP)
5. False Negatives (FN)
6. Singleton accuracy
7. Match-count distribution
8. Per-country metrics
9. Metrics specifically for entities whose true cardinality is 3+
10. Rejection counts by guard:
    - exact name failure
    - house-number conflict
    - street mismatch
    - missing-address guard
    - cardinality cap
"""

from __future__ import annotations

import csv
import gc
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = CODE_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "dataset" / "train"
EXP_DIR = CODE_DIR / "experiments"
REPORTS_DIR = CODE_DIR / "reports"

sys.path.insert(0, str(CODE_DIR / "src"))
sys.path.insert(0, str(CODE_DIR / "scripts"))

from authoritative_eval_fn import _load_val_s1_ids, _load_gt
from blocking import clean_business_name
from evaluation import Evaluator, compute_entity_metrics
from normalization import extract_numbers, normalize_address, tokenize


def get_house_number(addr: str) -> str:
    if not addr or str(addr) == "nan":
        return ""
    m = re.search(r"\b(\d+)\s*(?:bis|ter|[a-zA-Z])?\b", addr.lower())
    return m.group(1) if m else ""


def get_street_component(addr: str) -> str:
    if not addr or str(addr) == "nan":
        return ""
    parts = addr.split(",")
    st = parts[0].strip().lower()
    st = st.replace("boulevard", "bd").replace("blvd", "bd")
    st = st.replace("avenue", "av")
    st = st.replace("impasse", "imp")
    st = st.replace("street", "st").replace("road", "rd")
    return st


def parse_addr_full(addr: str) -> Tuple[str, Set[str], str]:
    if not addr or str(addr) == "nan":
        return "", set(), ""
    hn = get_house_number(addr)
    st = get_street_component(addr)
    norm = normalize_address(addr).lower()
    toks = set(tokenize(norm))
    return st, toks, hn


def run_evaluation():
    print("=" * 80)
    print("AUTHORITATIVE VALIDATION EVALUATION: EXP_N039 vs V2 PRODUCTION DECODER")
    print("=" * 80)

    # 1. Load Validation Entities and Ground Truth
    print("\n[Step 1] Loading 25,000 validation entities and ground truth...", flush=True)
    val_s1_set = _load_val_s1_ids()
    gt_full = _load_gt(val_s1_set)
    total_true_matches = sum(len(v) for v in gt_full.values())
    print(f"Loaded {len(val_s1_set):,} validation entities with {total_true_matches:,} true matches.")

    # 2. Load S1 Records for metadata (Name, Address, Country)
    print("\n[Step 2] Loading validation S1 entity metadata...", flush=True)
    s1_data: Dict[str, Tuple[str, str, Set[str], str, str]] = {} # id -> (norm_nm, st, toks, hn, country)
    with open(DATA_DIR / "train_source1.tsv", "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row.get("entity_id", "").strip()
            if eid in val_s1_set:
                nm = row.get("business_name", "") or ""
                addr = row.get("business_address", "") or ""
                country = (row.get("country") or "").strip()
                norm_nm, _ = clean_business_name(nm)
                st, toks, hn = parse_addr_full(addr)
                s1_data[eid] = (norm_nm, st, toks, hn, country)

    print(f"Loaded metadata for {len(s1_data):,} S1 entities.")

    # 3. Load Validation Candidate Pool
    print("\n[Step 3] Loading validation candidate pool (val_pairs_n010.npz)...", flush=True)
    va = np.load(EXP_DIR / "val_pairs_n010.npz", allow_pickle=True)
    s1_ids_str = va["s1_ids"].astype(str)
    cand_ids_str = va["cand_ids"].astype(str)
    sources_str = va["sources"].astype(str)

    N_ent = 25000
    K = 80
    cand_mat = cand_ids_str.reshape(N_ent, K)
    src_mat = sources_str.reshape(N_ent, K)
    unique_s1 = s1_ids_str[::80]

    all_cand_ids_set = set(cand_ids_str)
    print(f"Validation candidate pool: {N_ent:,} entities, {K} candidates/entity, {len(all_cand_ids_set):,} unique candidates.")

    # 4. Load Candidate Target Records from train_source2.tsv and train_source3.tsv
    print("\n[Step 4] Indexing target candidate metadata from train Source-2 and Source-3...", flush=True)
    target_info: Dict[str, Tuple[str, str, Set[str], str]] = {}
    for src_file, src_lbl in [("train_source2.tsv", "S2"), ("train_source3.tsv", "S3")]:
        t0 = time.time()
        c_count = 0
        with open(DATA_DIR / src_file, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                eid = row.get("entity_id", "").strip()
                if eid in all_cand_ids_set:
                    nm = row.get("business_name", "") or ""
                    addr = row.get("business_address", "") or ""
                    norm_nm, _ = clean_business_name(nm)
                    st, toks, hn = parse_addr_full(addr)
                    target_info[eid] = (norm_nm, st, toks, hn)
                    c_count += 1
        print(f"  Indexed {c_count:,} needed {src_lbl} records in {time.time()-t0:.2f}s", flush=True)

    print(f"Total target records indexed: {len(target_info):,}")

    # 5. Load Frozen EXP_N039 Baseline Predictions
    print("\n[Step 5] Loading Frozen EXP_N039 baseline predictions...", flush=True)
    rec_n014 = np.load(EXP_DIR / "n014_recovered_matches.npz")
    recovered_map = defaultdict(set)
    for s, c in zip(rec_n014["s1_ids"].astype(str), rec_n014["cand_ids"].astype(str)):
        recovered_map[s].add(c)

    df_rec = pd.read_csv(CODE_DIR / "reports" / "blocker_recovery_matrix.csv")
    for _, row in df_rec[df_rec["recovered_by"] != "none"].iterrows():
        recovered_map[str(row["s1_id"])].add(str(row["candidate_id"]))

    pl = np.load(EXP_DIR / "probs_n011_optimal_lgb.npy")
    px = np.load(EXP_DIR / "probs_n011_optimal_xgb.npy")
    p_n014 = (0.45 * pl + 0.55 * px).astype(np.float32)
    p_xlmr = np.load(EXP_DIR / "EXP_N017" / "probs_val_cross_encoder.npy").astype(np.float32)
    p_hyb = (0.73 * p_n014 + 0.27 * p_xlmr).astype(np.float32)

    p_mat = p_hyb.reshape(N_ent, K)
    sort_idx = np.argsort(-p_mat, axis=1)
    row_idx = np.arange(N_ent)[:, None]
    c_s = cand_mat[row_idx, sort_idx]

    pred_data = np.load(EXP_DIR / "pred_n039_matrix.npz")
    pred_L = pred_data["pred"]

    preds_n039: Dict[str, Set[str]] = {}
    for i in range(N_ent):
        s1 = unique_s1[i]
        ps = set(c_s[i][pred_L[i]])
        if s1 in recovered_map:
            ps.update(recovered_map[s1])
        preds_n039[s1] = ps

    # 6. Execute V2 Production Decoder Logic on Validation Pool
    print("\n[Step 6] Running V2 Production Decoder and Ablations...", flush=True)

    # Rejection counters for V2 full
    rejections = {
        "exact_name_failure": 0,
        "missing_address_guard": 0,
        "house_number_conflict": 0,
        "street_mismatch": 0,
        "cardinality_cap": 0,
    }

    preds_v2_full: Dict[str, Set[str]] = {}
    preds_v2_no_cap: Dict[str, Set[str]] = {}
    preds_v2_no_street: Dict[str, Set[str]] = {}
    preds_v2_no_missing_guard: Dict[str, Set[str]] = {}

    for i in range(N_ent):
        s1_id = unique_s1[i]
        cands = cand_mat[i].tolist()
        norm_nm, st1, toks1, hn1, country = s1_data[s1_id]

        admitted_s2_full = []
        admitted_s3_full = []

        admitted_s2_no_cap = []
        admitted_s3_no_cap = []

        admitted_s2_no_st = []
        admitted_s3_no_st = []

        admitted_s2_no_mg = []
        admitted_s3_no_mg = []

        for cid in cands:
            tup = target_info.get(cid)
            if not tup:
                continue
            cnorm, st2, toks2, hn2 = tup

            # Exact name failure
            if cnorm != norm_nm:
                rejections["exact_name_failure"] += 1
                continue

            # --- Ablation D (no missing address restriction) ---
            # Under no missing guard, any candidate with missing address is admitted
            if not toks2:
                # Full V2 logic
                if len(cands) <= 5:
                    sc = 0.50
                    if cid.startswith("S2-"):
                        admitted_s2_full.append((sc, cid))
                        admitted_s2_no_cap.append((sc, cid))
                        admitted_s2_no_st.append((sc, cid))
                    else:
                        admitted_s3_full.append((sc, cid))
                        admitted_s3_no_cap.append((sc, cid))
                        admitted_s3_no_st.append((sc, cid))
                else:
                    rejections["missing_address_guard"] += 1

                # In no-missing-guard ablation, unconditionally admit
                sc_mg = 0.50
                if cid.startswith("S2-"):
                    admitted_s2_no_mg.append((sc_mg, cid))
                else:
                    admitted_s3_no_mg.append((sc_mg, cid))
                continue
            else:
                # If candidate address is present, also add to no-missing-guard if it passes other checks
                pass

            # House number conflict check
            if hn1 and hn2 and hn1 != hn2:
                rejections["house_number_conflict"] += 1
                continue

            st_sim = fuzz.token_set_ratio(st1, st2) if (st1 and st2) else 0.0
            full_j = len(toks1 & toks2) / max(1, len(toks1 | toks2)) if (toks1 or toks2) else 0.0
            hn_match = (hn1 != "" and hn2 != "" and hn1 == hn2)

            # Match criterion for V2 Full & Ablations
            is_match = (
                (st_sim >= 80.0)
                or (hn_match and full_j >= 0.35)
                or (full_j >= 0.50)
            )

            score = 0.5 * (st_sim / 100.0) + (0.3 if hn_match else 0.0) + 0.2 * full_j

            if is_match:
                if cid.startswith("S2-"):
                    admitted_s2_full.append((score, cid))
                    admitted_s2_no_cap.append((score, cid))
                    admitted_s2_no_mg.append((score, cid))
                else:
                    admitted_s3_full.append((score, cid))
                    admitted_s3_no_cap.append((score, cid))
                    admitted_s3_no_mg.append((score, cid))
            else:
                rejections["street_mismatch"] += 1

            # Ablation C: V2 without street requirement (allows match based purely on address Jaccard >= 0.20 or house match)
            is_match_no_st = (
                (full_j >= 0.20)
                or hn_match
            )
            if is_match_no_st:
                if cid.startswith("S2-"):
                    admitted_s2_no_st.append((score, cid))
                else:
                    admitted_s3_no_st.append((score, cid))

        # Rejection by cardinality cap
        if len(admitted_s2_full) > 2:
            rejections["cardinality_cap"] += (len(admitted_s2_full) - 2)
        if len(admitted_s3_full) > 2:
            rejections["cardinality_cap"] += (len(admitted_s3_full) - 2)

        # Sort descending
        admitted_s2_full.sort(key=lambda x: -x[0])
        admitted_s3_full.sort(key=lambda x: -x[0])
        preds_v2_full[s1_id] = set([c for _, c in admitted_s2_full[:2]] + [c for _, c in admitted_s3_full[:2]])

        # Ablation B: without max-2 cap (takes all admitted candidates)
        preds_v2_no_cap[s1_id] = set([c for _, c in admitted_s2_no_cap] + [c for _, c in admitted_s3_no_cap])

        # Ablation C: without street requirement (with max-2 cap)
        admitted_s2_no_st.sort(key=lambda x: -x[0])
        admitted_s3_no_st.sort(key=lambda x: -x[0])
        preds_v2_no_street[s1_id] = set([c for _, c in admitted_s2_no_st[:2]] + [c for _, c in admitted_s3_no_st[:2]])

        # Ablation D: without missing-address restriction (with max-2 cap)
        admitted_s2_no_mg.sort(key=lambda x: -x[0])
        admitted_s3_no_mg.sort(key=lambda x: -x[0])
        preds_v2_no_missing_guard[s1_id] = set([c for _, c in admitted_s2_no_mg[:2]] + [c for _, c in admitted_s3_no_mg[:2]])

    print(f"Decoded all systems across {N_ent:,} validation entities.")

    # 7. Helper Evaluation Function
    def evaluate_system(preds_dict: Dict[str, Set[str]], subset_ids: Set[str] = None):
        evaluator = Evaluator()
        ids_to_eval = subset_ids if subset_ids is not None else val_s1_set
        for s1_id in ids_to_eval:
            true_set = gt_full.get(s1_id, set())
            pred_set = preds_dict.get(s1_id, set())
            evaluator.update(s1_id, true_set, pred_set)
        return evaluator.compute()

    # 8. Compute Full Metrics for All Systems
    print("\n[Step 7] Computing full validation metrics...", flush=True)

    res_n039 = evaluate_system(preds_n039)
    res_v2_full = evaluate_system(preds_v2_full)
    res_v2_no_cap = evaluate_system(preds_v2_no_cap)
    res_v2_no_street = evaluate_system(preds_v2_no_street)
    res_v2_no_mg = evaluate_system(preds_v2_no_missing_guard)

    # 9. Compute Subgroup Metrics:
    # A. Country Breakdown (US vs India)
    us_ids = {s for s in val_s1_set if s1_data[s][4] == "US"}
    in_ids = {s for s in val_s1_set if s1_data[s][4] == "India"}

    res_n039_us = evaluate_system(preds_n039, us_ids)
    res_v2_us = evaluate_system(preds_v2_full, us_ids)

    res_n039_in = evaluate_system(preds_n039, in_ids)
    res_v2_in = evaluate_system(preds_v2_full, in_ids)

    # B. Entities with True Cardinality >= 3
    card3_ids = {s for s in val_s1_set if len(gt_full.get(s, set())) >= 3}
    print(f"Entities with true cardinality >= 3: {len(card3_ids):,} ({len(card3_ids)/len(val_s1_set)*100:.2f}%)")

    res_n039_card3 = evaluate_system(preds_n039, card3_ids)
    res_v2_card3 = evaluate_system(preds_v2_full, card3_ids)
    res_v2_no_cap_card3 = evaluate_system(preds_v2_no_cap, card3_ids)

    # 10. Match Count Distribution
    def get_cardinality_dist(preds_dict):
        counts = [len(preds_dict.get(s, set())) for s in val_s1_set]
        return dict(sorted(Counter(counts).items()))

    card_dist_gt = dict(sorted(Counter([len(gt_full.get(s, set())) for s in val_s1_set]).items()))
    card_dist_n039 = get_cardinality_dist(preds_n039)
    card_dist_v2 = get_cardinality_dist(preds_v2_full)
    card_dist_v2_no_cap = get_cardinality_dist(preds_v2_no_cap)

    # Print Summary Tables
    print("\n" + "=" * 80)
    print("COMPARATIVE SUMMARY TABLE: EXP_N039 vs V2 PRODUCTION DECODER")
    print("=" * 80)
    print(f"{'Metric':<30} | {'EXP_N039':<15} | {'V2 Full (Prod)':<15} | {'Delta (V2 - N039)':<15}")
    print("-" * 80)
    print(f"{'Macro F0.5':<30} | {res_n039['macro_f_05']:<15.6f} | {res_v2_full['macro_f_05']:<15.6f} | {res_v2_full['macro_f_05'] - res_n039['macro_f_05']:<+15.6f}")
    print(f"{'Macro Precision':<30} | {res_n039['macro_precision']:<15.6f} | {res_v2_full['macro_precision']:<15.6f} | {res_v2_full['macro_precision'] - res_n039['macro_precision']:<+15.6f}")
    print(f"{'Macro Recall':<30} | {res_n039['macro_recall']:<15.6f} | {res_v2_full['macro_recall']:<15.6f} | {res_v2_full['macro_recall'] - res_n039['macro_recall']:<+15.6f}")
    print(f"{'False Positives (FP)':<30} | {res_n039['total_false_positives']:<15,d} | {res_v2_full['total_false_positives']:<15,d} | {res_v2_full['total_false_positives'] - res_n039['total_false_positives']:<+15,d}")
    print(f"{'False Negatives (FN)':<30} | {res_n039['total_false_negatives']:<15,d} | {res_v2_full['total_false_negatives']:<15,d} | {res_v2_full['total_false_negatives'] - res_n039['total_false_negatives']:<+15,d}")
    print(f"{'Singleton Accuracy':<30} | {res_n039['singleton_accuracy']:<15.4f} | {res_v2_full['singleton_accuracy']:<15.4f} | {res_v2_full['singleton_accuracy'] - res_n039['singleton_accuracy']:<+15.4f}")
    print(f"{'Exact Set Accuracy':<30} | {res_n039['exact_set_accuracy']:<15.4f} | {res_v2_full['exact_set_accuracy']:<15.4f} | {res_v2_full['exact_set_accuracy'] - res_n039['exact_set_accuracy']:<+15.4f}")
    print(f"{'Total Predicted Matches':<30} | {res_n039['total_predicted_matches']:<15,d} | {res_v2_full['total_predicted_matches']:<15,d} | {res_v2_full['total_predicted_matches'] - res_n039['total_predicted_matches']:<+15,d}")

    print("\n" + "=" * 80)
    print("PER-COUNTRY BREAKDOWN")
    print("=" * 80)
    print(f"{'Country / Metric':<25} | {'EXP_N039':<15} | {'V2 Full':<15}")
    print("-" * 65)
    print(f"{'US Macro F0.5':<25} | {res_n039_us['macro_f_05']:<15.6f} | {res_v2_us['macro_f_05']:<15.6f}")
    print(f"{'US Precision':<25} | {res_n039_us['macro_precision']:<15.6f} | {res_v2_us['macro_precision']:<15.6f}")
    print(f"{'US Recall':<25} | {res_n039_us['macro_recall']:<15.6f} | {res_v2_us['macro_recall']:<15.6f}")
    print(f"{'US FP / FN':<25} | {res_n039_us['total_false_positives']} / {res_n039_us['total_false_negatives']} | {res_v2_us['total_false_positives']} / {res_v2_us['total_false_negatives']}")
    print("-" * 65)
    print(f"{'India Macro F0.5':<25} | {res_n039_in['macro_f_05']:<15.6f} | {res_v2_in['macro_f_05']:<15.6f}")
    print(f"{'India Precision':<25} | {res_n039_in['macro_precision']:<15.6f} | {res_v2_in['macro_precision']:<15.6f}")
    print(f"{'India Recall':<25} | {res_n039_in['macro_recall']:<15.6f} | {res_v2_in['macro_recall']:<15.6f}")
    print(f"{'India FP / FN':<25} | {res_n039_in['total_false_positives']} / {res_n039_in['total_false_negatives']} | {res_v2_in['total_false_positives']} / {res_v2_in['total_false_negatives']}")

    print("\n" + "=" * 80)
    print("METRICS FOR ENTITIES WITH TRUE CARDINALITY >= 3 (17,495 entities)")
    print("=" * 80)
    print(f"{'Metric':<25} | {'EXP_N039':<15} | {'V2 Full':<15} | {'V2 No Cap':<15}")
    print("-" * 80)
    print(f"{'Macro F0.5':<25} | {res_n039_card3['macro_f_05']:<15.6f} | {res_v2_card3['macro_f_05']:<15.6f} | {res_v2_no_cap_card3['macro_f_05']:<15.6f}")
    print(f"{'Macro Precision':<25} | {res_n039_card3['macro_precision']:<15.6f} | {res_v2_card3['macro_precision']:<15.6f} | {res_v2_no_cap_card3['macro_precision']:<15.6f}")
    print(f"{'Macro Recall':<25} | {res_n039_card3['macro_recall']:<15.6f} | {res_v2_card3['macro_recall']:<15.6f} | {res_v2_no_cap_card3['macro_recall']:<15.6f}")
    print(f"{'FP / FN':<25} | {res_n039_card3['total_false_positives']} / {res_n039_card3['total_false_negatives']} | {res_v2_card3['total_false_positives']} / {res_v2_card3['total_false_negatives']} | {res_v2_no_cap_card3['total_false_positives']} / {res_v2_no_cap_card3['total_false_negatives']}")

    print("\n" + "=" * 80)
    print("V2 REJECTIONS BY GUARD")
    print("=" * 80)
    for guard, count in rejections.items():
        print(f"  - {guard:<28}: {count:,} candidate pairs rejected")

    print("\n" + "=" * 80)
    print("MATCH COUNT DISTRIBUTIONS")
    print("=" * 80)
    print(f"{'Cardinality':<12} | {'Ground Truth':<15} | {'EXP_N039':<15} | {'V2 Full':<15} | {'V2 No Cap':<15}")
    print("-" * 80)
    all_cards = sorted(set(card_dist_gt.keys()) | set(card_dist_n039.keys()) | set(card_dist_v2.keys()) | set(card_dist_v2_no_cap.keys()))
    for c in all_cards:
        print(f"{c:<12} | {card_dist_gt.get(c, 0):<15,d} | {card_dist_n039.get(c, 0):<15,d} | {card_dist_v2.get(c, 0):<15,d} | {card_dist_v2_no_cap.get(c, 0):<15,d}")

    print("\n" + "=" * 80)
    print("ABLATION STUDY: V2 ARCHITECTURAL COMPONENTS")
    print("=" * 80)
    print(f"{'Ablation Variant':<40} | {'Macro F0.5':<12} | {'Precision':<12} | {'Recall':<12} | {'FP':<8} | {'FN':<8}")
    print("-" * 95)
    print(f"{'A. V2 Full (Exact Production)':<40} | {res_v2_full['macro_f_05']:<12.6f} | {res_v2_full['macro_precision']:<12.6f} | {res_v2_full['macro_recall']:<12.6f} | {res_v2_full['total_false_positives']:<8,d} | {res_v2_full['total_false_negatives']:<8,d}")
    print(f"{'B. V2 without max-2 cap':<40} | {res_v2_no_cap['macro_f_05']:<12.6f} | {res_v2_no_cap['macro_precision']:<12.6f} | {res_v2_no_cap['macro_recall']:<12.6f} | {res_v2_no_cap['total_false_positives']:<8,d} | {res_v2_no_cap['total_false_negatives']:<8,d}")
    print(f"{'C. V2 without street requirement':<40} | {res_v2_no_street['macro_f_05']:<12.6f} | {res_v2_no_street['macro_precision']:<12.6f} | {res_v2_no_street['macro_recall']:<12.6f} | {res_v2_no_street['total_false_positives']:<8,d} | {res_v2_no_street['total_false_negatives']:<8,d}")
    print(f"{'D. V2 without missing-address restriction':<40} | {res_v2_no_mg['macro_f_05']:<12.6f} | {res_v2_no_mg['macro_precision']:<12.6f} | {res_v2_no_mg['macro_recall']:<12.6f} | {res_v2_no_mg['total_false_positives']:<8,d} | {res_v2_no_mg['total_false_negatives']:<8,d}")

    # Save detailed JSON report
    report_data = {
        "n039": res_n039,
        "v2_full": res_v2_full,
        "v2_no_cap": res_v2_no_cap,
        "v2_no_street": res_v2_no_street,
        "v2_no_missing_guard": res_v2_no_mg,
        "countries": {
            "n039_us": res_n039_us,
            "v2_us": res_v2_us,
            "n039_in": res_n039_in,
            "v2_in": res_v2_in,
        },
        "cardinality_3plus": {
            "n039": res_n039_card3,
            "v2_full": res_v2_card3,
            "v2_no_cap": res_v2_no_cap_card3,
        },
        "rejections": rejections,
        "cardinality_distributions": {
            "ground_truth": card_dist_gt,
            "n039": card_dist_n039,
            "v2_full": card_dist_v2,
            "v2_no_cap": card_dist_v2_no_cap,
        },
    }

    with open(REPORTS_DIR / "V2_VALIDATION_COMPARISON.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\nDetailed audit saved to {REPORTS_DIR / 'V2_VALIDATION_COMPARISON.json'}")


if __name__ == "__main__":
    run_evaluation()
