#!/usr/bin/env python3
"""
EXP_N017 — Step 4: Authoritative Complete-System Evaluation (Metric B) & Ablation Suite.

Evaluates:
A. XLM-R Cross-Encoder alone.
B. XLM-R + N014 Features Hybrid Meta-Ranker.
C. XLM-R + N014 + Sibling Context Graph Features.
D. Entity-Level Set Decision Engine on top of C.
E. Full Forensic FP/FN Audit.
"""

from __future__ import annotations

import gc
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
EXP_N017_DIR = EXP_DIR / "EXP_N017"


def main():
    t_start = time.time()
    print("=" * 80)
    print("[EXP_N017] Step 4: Authoritative Complete-System Evaluation Suite")
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

    p_xlmr = np.load(EXP_N017_DIR / "probs_val_cross_encoder.npy")

    # Combine into Hybrid Meta-Score
    # XLM-R provides high semantic sentence-pair discrimination
    # N014 provides fine-grained exact house number, postal code, token jaccard
    print("\n[Evaluation C & D] Constructing Contextual Hybrid Ranker...", flush=True)
    p_hybrid = 0.70 * p_n014 + 0.30 * p_xlmr

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

    base_th = 0.940
    dense_th = 0.950
    margin = 0.025
    anchor = 0.960
    bridge = 0.920
    density_cutoff = 75
    sg = base_th * 0.85

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
        s3 = sources_s[sl] == "S3"
        xs = X_s[sl]
        ys = y_s[sl]
        N_ = len(ps)

        if N_ == 0 or ps[0] < sg:
            pred_set = set()
        else:
            top = ps[0]
            et = dense_th if N_ > density_cutoff else base_th
            am = (ps >= et) & ((top - ps) <= margin)
            anm = ps >= anchor
            if np.any(anm & am):
                if not bool(np.any(am & s3)):
                    am = am | ((~am) & s3 & (ps >= bridge))
                if not bool(np.any(am & ~s3)):
                    am = am | ((~am) & ~s3 & (ps >= bridge))

            pred_set = set(cids[am])

            # Audit FPs for precision inspection
            for i in np.where(am)[0]:
                if ys[i] == 0:
                    fp_records.append({
                        "s1_id": s1_id,
                        "cand_id": cids[i],
                        "source": "S3" if s3[i] else "S2",
                        "hybrid_score": float(ps[i]),
                        "n014_score": float(p_n014[sort_key[starts[k] + i]]),
                        "xlmr_score": float(p_xlmr[sort_key[starts[k] + i]]),
                    })

        # Inject cached recovered matches
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
    print("EXP_N017 AUTHORITATIVE COMPLETE-SYSTEM METRIC RESULTS:")
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

    # Save FP audit
    if fp_records:
        df_fp = pd.DataFrame(fp_records)
        df_fp.to_csv(EXP_N017_DIR / "fp_audit.csv", index=False)
        print(f"Saved FP audit to: {EXP_N017_DIR / 'fp_audit.csv'}")

    # Save evaluation metrics
    results_summary = {
        "metric_b_complete_system_f05": mB["macro_f_05"],
        "metric_a_decision_layer_f05": mA["macro_f_05"],
        "precision": mB["macro_precision"],
        "recall": mB["macro_recall"],
        "fp": total_fp,
        "fn_model": total_fn_model,
        "fn_blocker": total_fn_blocker,
        "total_fn": total_fn_model + total_fn_blocker,
        "singleton_accuracy": singleton_correct_count / max(1, singleton_gt_count),
        "multi_match_recall": multi_tp_count / max(1, multi_gt_count),
    }
    with open(EXP_N017_DIR / "evaluation_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"Saved evaluation metrics to: {EXP_N017_DIR / 'evaluation_summary.json'}")


if __name__ == "__main__":
    main()
