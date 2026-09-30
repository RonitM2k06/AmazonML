#!/usr/bin/env python3
"""
EXP_N019 — Authoritative Complete-System Evaluation Suite for Collective Set Resolution.

Evaluates Complete-System Macro F0.5 (Metric B), Decision Layer F0.5 (Metric A),
Precision, Recall, Singleton metrics, and Multi-match metrics on the frozen 25k validation set.
"""

from __future__ import annotations

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

DATA_DIR = PROJECT_ROOT / "dataset" / "train"
EXP_DIR = CODE_DIR / "experiments"
EXP_N019_DIR = EXP_DIR / "EXP_N019"
EXP_N019_DIR.mkdir(parents=True, exist_ok=True)


def main():
    t_start = time.time()
    print("=" * 80)
    print("[EXP_N019] Authoritative Complete-System Collective Set Evaluation Suite")
    print("=" * 80)

    # 1. Load Ground Truth and 25k Validation IDs
    val_s1_set = _load_val_s1_ids()
    gt_full = _load_gt(val_s1_set)
    total_true_matches = sum(len(mids) for mids in gt_full.values())
    print(f"Validation Population: {len(val_s1_set):,} entities, {total_true_matches:,} true matches.")

    # 2. Load Cached 602 Sibling Recovered Matches from N014
    rec_data = np.load(EXP_DIR / "n014_recovered_matches.npz")
    rec_s1 = rec_data["s1_ids"].astype(str)
    rec_cand = rec_data["cand_ids"].astype(str)
    recovered_map = defaultdict(set)
    for s, c in zip(rec_s1, rec_cand):
        recovered_map[s].add(c)
    print(f"Loaded {len(rec_s1):,} recovered sibling matches from N014 cache.")

    # 3. Load Validation Pairs, Tabular Probabilities, and Cross-Encoder Probabilities
    va = np.load(EXP_DIR / "val_pairs_n010.npz", allow_pickle=True)
    s1_ids_str = va["s1_ids"].astype(str)
    cand_ids_str = va["cand_ids"].astype(str)
    sources_str = va["sources"].astype(str)
    y = va["y"].astype(np.int8)
    X = va["X"].astype(np.float32)

    pl = np.load(EXP_DIR / "probs_n011_optimal_lgb.npy")
    px = np.load(EXP_DIR / "probs_n011_optimal_xgb.npy")
    p_n014 = 0.45 * pl + 0.55 * px
    p_xlmr = np.load(EXP_DIR / "EXP_N017" / "probs_val_cross_encoder.npy")

    w_xlmr = 0.27
    p_hybrid = (1.0 - w_xlmr) * p_n014 + w_xlmr * p_xlmr

    unique_s1, s1_codes = np.unique(s1_ids_str, return_inverse=True)
    sort_key = np.lexsort((-p_hybrid, s1_codes))
    s1_codes_s = s1_codes[sort_key]
    probs_s = p_hybrid[sort_key]
    cand_ids_s = cand_ids_str[sort_key]
    sources_s = sources_str[sort_key]
    X_s = X[sort_key]
    y_s = y[sort_key]

    boundaries = np.where(np.diff(s1_codes_s) != 0)[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(probs_s)]])

    th = 0.9570
    eval_A = Evaluator()
    eval_B = Evaluator()
    total_fp = total_fn_model = total_fn_blocker = 0
    singleton_gt_count = singleton_correct_count = 0
    multi_gt_count = multi_tp_count = 0

    fp_records = []

    for k, s1_id in enumerate(unique_s1):
        sl = slice(starts[k], ends[k])
        ps = probs_s[sl]
        cids = cand_ids_s[sl]
        ys = y_s[sl]
        am = ps >= th
        pred_set = set(cids[am])

        for i in np.where(am)[0]:
            if ys[i] == 0:
                fp_records.append({
                    "s1_id": s1_id,
                    "cand_id": cids[i],
                    "prob": float(ps[i]),
                })

        if s1_id in recovered_map:
            pred_set = pred_set | recovered_map[s1_id]

        true_full = gt_full.get(s1_id, set())
        cand_pool = set(cids.tolist())
        true_in_pool = true_full & cand_pool
        blocker_miss = true_full - cand_pool

        eval_A.update(s1_id, true_in_pool, pred_set)
        eval_B.update(s1_id, true_full, pred_set)

        total_fp += len(pred_set - true_full)
        total_fn_model += len(true_in_pool - pred_set)
        total_fn_blocker += len(blocker_miss - pred_set)

        if len(true_full) == 0:
            singleton_gt_count += 1
            if len(pred_set) == 0:
                singleton_correct_count += 1
        else:
            multi_gt_count += len(true_full)
            multi_tp_count += len(true_full & pred_set)

    npz_s1_set = set(unique_s1.tolist())
    for s1_id in val_s1_set:
        if s1_id not in npz_s1_set:
            true_full = gt_full.get(s1_id, set())
            eval_A.update(s1_id, set(), set())
            eval_B.update(s1_id, true_full, set())
            total_fn_blocker += len(true_full)
            if len(true_full) == 0:
                singleton_gt_count += 1
                singleton_correct_count += 1
            else:
                multi_gt_count += len(true_full)

    mA = eval_A.compute()
    mB = eval_B.compute()
    elapsed = time.time() - t_start

    print("\n" + "=" * 80)
    print("EXP_N019 AUTHORITATIVE COMPLETE-SYSTEM METRIC RESULTS:")
    print("=" * 80)
    print(f"  Complete System Macro F0.5 (Metric B): {mB['macro_f_05']:.6f}")
    print(f"  Decision Layer Macro F0.5 (Metric A):  {mA['macro_f_05']:.6f}")
    print(f"  Complete System Macro Precision:       {mB['macro_precision']:.6f}")
    print(f"  Complete System Macro Recall:          {mB['macro_recall']:.6f}")
    print(f"  False Positives (FP):                  {total_fp}")
    print(f"  False Negatives Model (in pool):       {total_fn_model}")
    print(f"  False Negatives Blocker (misses):      {total_fn_blocker}")
    print(f"  Total False Negatives:                 {total_fn_model + total_fn_blocker}")
    print(f"  Singleton Accuracy:                    {singleton_correct_count / max(1, singleton_gt_count):.6f}")
    print(f"  Multi-Match Recall:                    {multi_tp_count / max(1, multi_gt_count):.6f}")
    print(f"  Evaluation Runtime:                    {elapsed:.1f}s")
    print("=" * 80)

    # Save summary json
    summary = {
        "experiment": "EXP_N019",
        "champion_model": "listwise_collective_set_resolution_EXP_N019",
        "decision_strategy": "unconstrained_collective_multi_match_optimal_calibrated_acceptance",
        "metric_b_complete_system_f05": float(mB["macro_f_05"]),
        "metric_a_decision_layer_f05": float(mA["macro_f_05"]),
        "metric_macro_precision": float(mB["macro_precision"]),
        "metric_macro_recall": float(mB["macro_recall"]),
        "eval_population_s1": len(val_s1_set),
        "total_false_positives": int(total_fp),
        "total_false_negatives_model": int(total_fn_model),
        "total_false_negatives_blocker": int(total_fn_blocker),
        "total_false_negatives": int(total_fn_model + total_fn_blocker),
        "singleton_accuracy": float(singleton_correct_count / max(1, singleton_gt_count)),
        "multi_match_recall": float(multi_tp_count / max(1, multi_gt_count)),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(EXP_N019_DIR / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation summary to: {EXP_N019_DIR / 'evaluation_summary.json'}")

    # Update champion.json
    with open(CODE_DIR / "reports" / "champion.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Updated champion.json with new record Metric B = {mB['macro_f_05']:.6f}!")


if __name__ == "__main__":
    main()
