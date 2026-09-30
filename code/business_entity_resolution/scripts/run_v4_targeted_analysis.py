#!/usr/bin/env python3
"""
Comprehensive Targeted Error Analysis and Rule Exploration for V4
Testing Phase 1-6 rules on the 25,000 entity-disjoint validation split.
"""

import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

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
from evaluation import Evaluator
from normalization import extract_numbers, normalize_address, tokenize


def main():
    print("=" * 80)
    print("V4 TARGETED ERROR ANALYSIS & RULE SEARCH")
    print("=" * 80)

    # 1. Load Validation Data & Ground Truth
    print("\n[Step 1] Loading validation entities and ground truth...", flush=True)
    val_s1_set = _load_val_s1_ids()
    gt_full = _load_gt(val_s1_set)

    df_s1 = pd.read_csv(DATA_DIR / "train_source1.tsv", sep="\t")
    val_df_s1 = df_s1[df_s1["entity_id"].isin(val_s1_set)]
    s1_country = dict(zip(val_df_s1["entity_id"], val_df_s1["country"]))
    s1_addr = dict(zip(val_df_s1["entity_id"], val_df_s1["business_address"].fillna("")))
    s1_name = dict(zip(val_df_s1["entity_id"], val_df_s1["business_name"].fillna("")))

    us_ids = {s for s in val_s1_set if s1_country.get(s) == "US"}
    in_ids = {s for s in val_s1_set if s1_country.get(s) == "India"}
    card3_ids = {s for s in val_s1_set if len(gt_full.get(s, set())) >= 3}

    # Candidate metadata
    print("\n[Step 2] Indexing target records...", flush=True)
    cand_meta = {}
    for sf in ["train_source2.tsv", "train_source3.tsv"]:
        with open(DATA_DIR / sf, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                cand_meta[row["entity_id"]] = {
                    "name": row.get("business_name", "") or "",
                    "address": row.get("business_address", "") or "",
                    "country": row.get("country", "") or "",
                }

    # Load N039
    print("\n[Step 3] Loading N039 baseline...", flush=True)
    va = np.load(EXP_DIR / "val_pairs_n010.npz", allow_pickle=True)
    s1_ids = va["s1_ids"].astype(str)
    cand_ids = va["cand_ids"].astype(str)
    cand_mat = cand_ids.reshape(25000, 80)
    unique_s1 = s1_ids[::80]

    rec_n014 = np.load(EXP_DIR / "n014_recovered_matches.npz")
    recovered_map = defaultdict(set)
    for s, c in zip(rec_n014["s1_ids"].astype(str), rec_n014["cand_ids"].astype(str)):
        recovered_map[s].add(c)
    df_rec = pd.read_csv(REPORTS_DIR / "blocker_recovery_matrix.csv")
    for _, row in df_rec[df_rec["recovered_by"] != "none"].iterrows():
        recovered_map[str(row["s1_id"])].add(str(row["candidate_id"]))

    pred_data = np.load(EXP_DIR / "pred_n039_matrix.npz")
    pred_L = pred_data["pred"]
    pl = np.load(EXP_DIR / "probs_n011_optimal_lgb.npy")
    px = np.load(EXP_DIR / "probs_n011_optimal_xgb.npy")
    p_n014 = (0.45 * pl + 0.55 * px).astype(np.float32)
    p_xlmr = np.load(EXP_DIR / "EXP_N017" / "probs_val_cross_encoder.npy").astype(np.float32)
    p_hyb = (0.73 * p_n014 + 0.27 * p_xlmr).astype(np.float32)
    p_mat = p_hyb.reshape(25000, 80)
    sort_idx = np.argsort(-p_mat, axis=1)

    orig_pred_mask = np.zeros((25000, 80), dtype=bool)
    for i in range(25000):
        orig_pred_mask[i, sort_idx[i]] = pred_L[i]

    preds_n039 = {}
    for i in range(25000):
        s1 = unique_s1[i]
        cands_acc = set(cand_mat[i][orig_pred_mask[i]])
        if s1 in recovered_map:
            cands_acc |= recovered_map[s1]
        preds_n039[s1] = cands_acc

    s1_c_to_prob = {}
    for i in range(25000):
        s1 = unique_s1[i]
        for k in range(80):
            c = cand_mat[i, k]
            s1_c_to_prob[(s1, c)] = float(p_mat[i, k])

    def evaluate_predictions(pred_map: Dict[str, Set[str]], subset: Set[str] = None):
        evaluator = Evaluator()
        subset_to_use = subset if subset is not None else val_s1_set
        for s1 in subset_to_use:
            evaluator.update(s1, gt_full.get(s1, set()), pred_map.get(s1, set()))
        return evaluator.compute()

    base_eval = evaluate_predictions(preds_n039)
    base_us = evaluate_predictions(preds_n039, us_ids)
    base_in = evaluate_predictions(preds_n039, in_ids)
    base_c3 = evaluate_predictions(preds_n039, card3_ids)

    print("\n" + "=" * 80)
    print("PHASE 1: EXACT BASELINE REPRODUCTION")
    print(f"  Macro F0.5:      {base_eval['macro_f_05']:.6f}")
    print(f"  Macro Precision: {base_eval['macro_precision']:.6f}")
    print(f"  Macro Recall:    {base_eval['macro_recall']:.6f}")
    print(f"  FP: {base_eval['total_false_positives']} | FN: {base_eval['total_false_negatives']}")
    print("=" * 80)

    # Precompute cleaned names
    clean_name_cache = {}
    for s1 in val_s1_set:
        clean_name_cache[s1] = clean_business_name(s1_name[s1])[0]
    for c, meta in cand_meta.items():
        clean_name_cache[c] = clean_business_name(meta["name"])[0]

    # Precompute house numbers
    hn_cache = {}
    for s1 in val_s1_set:
        hn_cache[s1] = [str(int(n)) for n in re.findall(r"\b\d+\b", s1_addr[s1].lower()) if n.isdigit()]
    for c, meta in cand_meta.items():
        hn_cache[c] = [str(int(n)) for n in re.findall(r"\b\d+\b", meta["address"].lower()) if n.isdigit()]

    # Collect individual filter results
    experiment_results = []

    # Rule A: Missing candidate address filter
    print("\n[Phase 2.A] Testing Missing Candidate Address Filters...", flush=True)
    for p_thresh in [0.90, 0.93, 0.95, 0.97, 0.98, 0.99]:
        preds_test = {}
        tp_rem = 0
        fp_rem = 0
        for s1 in val_s1_set:
            cands = set(preds_n039.get(s1, set()))
            to_remove = set()
            for c in cands:
                c_addr = cand_meta.get(c, {}).get("address", "")
                prob = s1_c_to_prob.get((s1, c), 1.0)
                if not c_addr and prob < p_thresh:
                    if clean_name_cache.get(s1) != clean_name_cache.get(c):
                        to_remove.add(c)
            for c in to_remove:
                if c in gt_full.get(s1, set()):
                    tp_rem += 1
                else:
                    fp_rem += 1
            preds_test[s1] = cands - to_remove
        ev = evaluate_predictions(preds_test)
        res_dict = {
            "name": f"Filter Missing Addr (p<{p_thresh}, non-exact name)",
            "macro_f_05": ev["macro_f_05"],
            "macro_precision": ev["macro_precision"],
            "macro_recall": ev["macro_recall"],
            "fp": ev["total_false_positives"],
            "fn": ev["total_false_negatives"],
            "tp_removed": tp_rem,
            "fp_removed": fp_rem,
            "net_tp_fp": -tp_rem - fp_rem,
        }
        experiment_results.append(res_dict)
        print(f"  {res_dict['name']}: TP_rem={tp_rem}, FP_rem={fp_rem}, F0.5={ev['macro_f_05']:.6f}, P={ev['macro_precision']:.6f}, R={ev['macro_recall']:.6f}")

    # Rule B: House-Number Conflicts
    print("\n[Phase 2.B] Testing House Number Conflict Filters...", flush=True)
    for p_thresh in [0.95, 0.97, 0.98, 0.99]:
        preds_test = {}
        tp_rem = 0
        fp_rem = 0
        for s1 in val_s1_set:
            cands = set(preds_n039.get(s1, set()))
            to_remove = set()
            hn1 = hn_cache.get(s1, [])
            for c in cands:
                hn2 = hn_cache.get(c, [])
                prob = s1_c_to_prob.get((s1, c), 1.0)
                # Disjoint numbers AND prob < p_thresh
                if hn1 and hn2 and not (set(hn1) & set(hn2)) and prob < p_thresh:
                    to_remove.add(c)
            for c in to_remove:
                if c in gt_full.get(s1, set()):
                    tp_rem += 1
                else:
                    fp_rem += 1
            preds_test[s1] = cands - to_remove
        ev = evaluate_predictions(preds_test)
        res_dict = {
            "name": f"Filter House Number Conflict (p<{p_thresh})",
            "macro_f_05": ev["macro_f_05"],
            "macro_precision": ev["macro_precision"],
            "macro_recall": ev["macro_recall"],
            "fp": ev["total_false_positives"],
            "fn": ev["total_false_negatives"],
            "tp_removed": tp_rem,
            "fp_removed": fp_rem,
        }
        experiment_results.append(res_dict)
        print(f"  {res_dict['name']}: TP_rem={tp_rem}, FP_rem={fp_rem}, F0.5={ev['macro_f_05']:.6f}, P={ev['macro_precision']:.6f}, R={ev['macro_recall']:.6f}")

    # Rule C: Different-street branch collisions
    print("\n[Phase 2.C] Testing Low Street Similarity Filter...", flush=True)
    for fuzz_thresh in [30, 40, 50]:
        preds_test = {}
        tp_rem = 0
        fp_rem = 0
        for s1 in val_s1_set:
            cands = set(preds_n039.get(s1, set()))
            to_remove = set()
            a1 = s1_addr.get(s1, "")
            for c in cands:
                a2 = cand_meta.get(c, {}).get("address", "")
                prob = s1_c_to_prob.get((s1, c), 1.0)
                if a1 and a2 and prob < 0.97:
                    # check address fuzz ratio
                    sim = fuzz.token_set_ratio(a1, a2)
                    if sim < fuzz_thresh and clean_name_cache.get(s1) != clean_name_cache.get(c):
                        to_remove.add(c)
            for c in to_remove:
                if c in gt_full.get(s1, set()):
                    tp_rem += 1
                else:
                    fp_rem += 1
            preds_test[s1] = cands - to_remove
        ev = evaluate_predictions(preds_test)
        res_dict = {
            "name": f"Filter Low Street Sim (sim<{fuzz_thresh}, p<0.97)",
            "macro_f_05": ev["macro_f_05"],
            "macro_precision": ev["macro_precision"],
            "macro_recall": ev["macro_recall"],
            "fp": ev["total_false_positives"],
            "fn": ev["total_false_negatives"],
            "tp_removed": tp_rem,
            "fp_removed": fp_rem,
        }
        experiment_results.append(res_dict)
        print(f"  {res_dict['name']}: TP_rem={tp_rem}, FP_rem={fp_rem}, F0.5={ev['macro_f_05']:.6f}, P={ev['macro_precision']:.6f}, R={ev['macro_recall']:.6f}")

    # Rule D: Short / Generic brand collisions
    print("\n[Phase 2.D] Testing Short Generic Brand Collision Filter...", flush=True)
    for min_len in [4, 6]:
        preds_test = {}
        tp_rem = 0
        fp_rem = 0
        for s1 in val_s1_set:
            cands = set(preds_n039.get(s1, set()))
            to_remove = set()
            nm = clean_name_cache.get(s1, "")
            a1 = s1_addr.get(s1, "")
            if len(nm) <= min_len:
                for c in cands:
                    a2 = cand_meta.get(c, {}).get("address", "")
                    prob = s1_c_to_prob.get((s1, c), 1.0)
                    if (not a2 or not a1) and prob < 0.98:
                        to_remove.add(c)
            for c in to_remove:
                if c in gt_full.get(s1, set()):
                    tp_rem += 1
                else:
                    fp_rem += 1
            preds_test[s1] = cands - to_remove
        ev = evaluate_predictions(preds_test)
        res_dict = {
            "name": f"Filter Short Name (len<={min_len}, missing addr, p<0.98)",
            "macro_f_05": ev["macro_f_05"],
            "macro_precision": ev["macro_precision"],
            "macro_recall": ev["macro_recall"],
            "fp": ev["total_false_positives"],
            "fn": ev["total_false_negatives"],
            "tp_removed": tp_rem,
            "fp_removed": fp_rem,
        }
        experiment_results.append(res_dict)
        print(f"  {res_dict['name']}: TP_rem={tp_rem}, FP_rem={fp_rem}, F0.5={ev['macro_f_05']:.6f}, P={ev['macro_precision']:.6f}, R={ev['macro_recall']:.6f}")

    # Phase 3 & 4: Recovery Rules for False Negatives
    print("\n[Phase 3 & 4] Testing Targeted Recovery Rules for False Negatives...", flush=True)
    # 3.A Legal suffix exact match recovery
    # Check candidates in pool not accepted by N039
    rec_test = {}
    tp_added = 0
    fp_added = 0
    for i in range(25000):
        s1 = unique_s1[i]
        cands = set(preds_n039.get(s1, set()))
        true_s = gt_full.get(s1, set())
        n1 = clean_name_cache.get(s1, "")
        for k in range(80):
            c = cand_mat[i, k]
            if c not in cands:
                n2 = clean_name_cache.get(c, "")
                prob = float(p_mat[i, k])
                # Conservative rule: exact clean name + strong address agreement
                a1 = s1_addr.get(s1, "")
                a2 = cand_meta.get(c, {}).get("address", "")
                if n1 and n2 and n1 == n2 and a1 and a2:
                    sim = fuzz.token_set_ratio(a1, a2)
                    if sim >= 85 and prob >= 0.80:
                        cands.add(c)
                        if c in true_s:
                            tp_added += 1
                        else:
                            fp_added += 1
        rec_test[s1] = cands

    ev_rec = evaluate_predictions(rec_test)
    res_dict = {
        "name": "Recovery: Clean Name Exact + Addr Sim >=85 + p>=0.80",
        "macro_f_05": ev_rec["macro_f_05"],
        "macro_precision": ev_rec["macro_precision"],
        "macro_recall": ev_rec["macro_recall"],
        "fp": ev_rec["total_false_positives"],
        "fn": ev_rec["total_false_negatives"],
        "tp_added": tp_added,
        "fp_added": fp_added,
    }
    experiment_results.append(res_dict)
    print(f"  {res_dict['name']}: TP_add={tp_added}, FP_add={fp_added}, F0.5={ev_rec['macro_f_05']:.6f}, P={ev_rec['macro_precision']:.6f}, R={ev_rec['macro_recall']:.6f}")

    # Output Summary JSON
    output_data = {
        "baseline_n039": {
            "macro_f_05": base_eval["macro_f_05"],
            "macro_precision": base_eval["macro_precision"],
            "macro_recall": base_eval["macro_recall"],
            "fp": base_eval["total_false_positives"],
            "fn": base_eval["total_false_negatives"],
            "exact_set_accuracy": base_eval["exact_set_accuracy"],
            "singleton_accuracy": base_eval["singleton_accuracy"],
            "total_predicted_matches": base_eval["total_predicted_matches"],
            "us": {
                "macro_f_05": base_us["macro_f_05"],
                "macro_precision": base_us["macro_precision"],
                "macro_recall": base_us["macro_recall"],
            },
            "india": {
                "macro_f_05": base_in["macro_f_05"],
                "macro_precision": base_in["macro_precision"],
                "macro_recall": base_in["macro_recall"],
            },
            "cardinality_ge_3": {
                "macro_f_05": base_c3["macro_f_05"],
                "macro_precision": base_c3["macro_precision"],
                "macro_recall": base_c3["macro_recall"],
            },
        },
        "experiments": experiment_results,
    }

    with open(REPORTS_DIR / "V4_TARGETED_ERROR_ANALYSIS.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved JSON results to: {REPORTS_DIR / 'V4_TARGETED_ERROR_ANALYSIS.json'}")


if __name__ == "__main__":
    main()
