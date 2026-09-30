#!/usr/bin/env python3
"""
AUTHORITATIVE EVALUATOR — EXP_N008

Produces two canonical numbers:

  A. Decision-layer F0.5
     Ground truth = true matches captured by the blocker (in val_pairs_n008.npz)
     This is what previous NPZ-based evaluation measured.

  B. Complete-system F0.5
     Ground truth = ALL true matches from train_ground_truth.tsv for the 25k val S1s.
     Blocker misses count as FNs.
     This is the number we must optimize toward ≥ 0.9950.

Also produces per-entity error records for Phase 1 error analysis.

Output files:
  reports/authoritative_eval.json          — A and B metrics side by side
  reports/exp_n008_errors_fp.csv           — false positives with features
  reports/exp_n008_errors_fn_model.csv     — false negatives (in pool, missed by model)
  reports/exp_n008_errors_fn_blocker.csv   — false negatives (not in pool / blocker miss)
"""

import csv
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONUNBUFFERED"] = "1"

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
code_dir = os.path.join(project_root, "code", "business_entity_resolution")
sys.path.insert(0, os.path.join(code_dir, "src"))

from evaluation import Evaluator
from pair_features import FEATURE_NAMES

EXPERIMENTS_DIR = os.path.join(code_dir, "experiments")
REPORTS_DIR = os.path.join(code_dir, "reports")
TRAIN_DIR = os.path.join(project_root, "dataset", "train")
VAL_IDS_PATH = os.path.join(EXPERIMENTS_DIR, "validation_s1_ids.txt")

SEED = 42
VAL_SAMPLE_SIZE = 25000
W_LGB = 0.45

# Best EXP_N008 params
BT, DT, MG, AG, BR, DC = 0.956, 0.963, 0.025, 0.978, 0.938, 75


def load_val_s1_ids():
    random.seed(SEED)
    with open(VAL_IDS_PATH) as f:
        pool = [l.strip() for l in f if l.strip()]
    return set(random.sample(pool, min(VAL_SAMPLE_SIZE, len(pool))))


def stream_ground_truth(val_s1_set):
    """Single-pass stream of ground truth TSV — returns only val S1 entries."""
    print("Streaming ground truth TSV...", flush=True)
    t0 = time.time()
    gt = {}
    path = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1 = parts[0]
            if s1 not in val_s1_set:
                continue
            mids = set(parts[1].split(",")) if len(parts) > 1 and parts[1].strip() else set()
            gt[s1] = mids
            if len(gt) >= len(val_s1_set):
                break
    print(f"  Loaded GT for {len(gt):,} S1 entities in {time.time()-t0:.1f}s", flush=True)
    total_matches = sum(len(v) for v in gt.values())
    n_singletons = sum(1 for v in gt.values() if not v)
    print(f"  Total true match pairs: {total_matches:,}  |  Singletons: {n_singletons:,}", flush=True)
    return gt


def load_npz_and_probs():
    """Load val_pairs_n008.npz and blended probabilities."""
    print("Loading NPZ + probs...", flush=True)
    t0 = time.time()
    va = np.load(os.path.join(EXPERIMENTS_DIR, "val_pairs_n008.npz"), allow_pickle=True)
    probs_lgb = np.load(os.path.join(EXPERIMENTS_DIR, "probs_n008_lgb.npy"))
    probs_xgb = np.load(os.path.join(EXPERIMENTS_DIR, "probs_n008_xgb.npy"))
    probs = (W_LGB * probs_lgb + (1 - W_LGB) * probs_xgb).astype(np.float32)
    print(f"  {len(probs):,} pairs in {time.time()-t0:.1f}s  |  mean_prob={probs.mean():.4f}", flush=True)
    return va, probs


def vectorized_group(va, probs):
    """
    Group all 1.66M pairs by S1 entity using vectorized numpy ops.
    Returns per-entity sorted structs.
    """
    print("Vectorized grouping by S1 entity...", flush=True)
    t0 = time.time()

    s1_ids_str   = va["s1_ids"].astype(str)
    cand_ids_str = va["cand_ids"].astype(str)
    sources_str  = va["sources"].astype(str)
    y            = va["y"].astype(np.int8)
    X            = va["X"].astype(np.float32)

    unique_s1, s1_codes = np.unique(s1_ids_str, return_inverse=True)

    # Sort by (s1_code, -prob) for descending prob within each entity
    sort_key = np.lexsort((-probs, s1_codes))
    s1_codes_s  = s1_codes[sort_key]
    probs_s     = probs[sort_key]
    cand_ids_s  = cand_ids_str[sort_key]
    sources_s   = sources_str[sort_key]
    y_s         = y[sort_key]
    X_s         = X[sort_key]

    boundaries = np.where(np.diff(s1_codes_s) != 0)[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends   = np.concatenate([boundaries, [len(probs_s)]])

    print(f"  Grouped {len(unique_s1):,} entities in {time.time()-t0:.1f}s", flush=True)
    return unique_s1, starts, ends, probs_s, cand_ids_s, sources_s, y_s, X_s


def run_decision(ps, cids, is_s3):
    """Apply bridge decision logic at best EXP_N008 params."""
    N = len(ps)
    sg = BT * 0.85
    if N == 0 or ps[0] < sg:
        return set()
    top = ps[0]
    et = DT if N > DC else BT
    am = (ps >= et) & ((top - ps) <= MG)
    anm = ps >= AG
    if np.any(anm & am):
        if not bool(np.any(am & is_s3)):
            am = am | ((~am) & is_s3 & (ps >= BR))
        if not bool(np.any(am & ~is_s3)):
            am = am | ((~am) & ~is_s3 & (ps >= BR))
    return set(cids[am])


def main():
    t_start = time.time()
    print("=" * 70)
    print("AUTHORITATIVE EVALUATOR — EXP_N008")
    print("=" * 70)

    # ── Load data ──
    val_s1_set = load_val_s1_ids()
    gt_full = stream_ground_truth(val_s1_set)

    # Fill in val S1s with no matches (true singletons not in GT file)
    for s1 in val_s1_set:
        if s1 not in gt_full:
            gt_full[s1] = set()

    va, probs = load_npz_and_probs()
    unique_s1, starts, ends, probs_s, cand_ids_s, sources_s, y_s, X_s = vectorized_group(va, probs)

    # Build set of S1 IDs present in NPZ
    npz_s1_set = set(unique_s1.tolist())

    # ── Evaluate ──
    print("\nEvaluating A (decision-only) and B (complete-system)...", flush=True)
    t0 = time.time()

    eval_A = Evaluator()  # true_set = only what's in NPZ pool
    eval_B = Evaluator()  # true_set = full GT (including blocker misses)

    # Error records
    fp_rows   = []
    fn_model_rows    = []
    fn_blocker_rows  = []

    n_s1_total      = len(val_s1_set)
    n_s1_in_npz     = 0
    n_s1_not_in_npz = 0
    total_blocker_misses = 0
    total_model_fn  = 0
    total_fp        = 0

    # Process entities in NPZ
    for k, s1_id in enumerate(unique_s1):
        sl = slice(starts[k], ends[k])
        ps    = probs_s[sl]
        cids  = cand_ids_s[sl]
        s3    = (sources_s[sl] == "S3")
        ys    = y_s[sl]
        Xs    = X_s[sl]

        # Ground truth
        true_full  = gt_full.get(s1_id, set())
        cand_pool  = set(cids.tolist())
        true_in_pool = {c for c in true_full if c in cand_pool}
        blocker_misses = true_full - cand_pool

        # Decision
        predicted = run_decision(ps, cids, s3)

        # Metric A: decision-only
        eval_A.update(s1_id, true_in_pool, predicted)

        # Metric B: complete-system
        eval_B.update(s1_id, true_full, predicted)

        # FP: predicted but not in true_full
        fps = predicted - true_full
        total_fp += len(fps)
        for cid in fps:
            idx_arr = np.where(cids == cid)[0]
            if len(idx_arr) == 0:
                continue
            idx = idx_arr[0]
            fp_rows.append({
                "s1_id": s1_id,
                "cand_id": cid,
                "source": "S3" if s3[idx] else "S2",
                "prob": float(ps[idx]),
                "rank": int(idx) + 1,
                "cosine_sim": float(Xs[idx, 56]) if Xs.shape[1] > 56 else None,
                **{FEATURE_NAMES[j]: float(Xs[idx, j]) for j in range(min(56, Xs.shape[1]))},
            })

        # FN model: in pool, in true, not predicted
        fn_model = true_in_pool - predicted
        total_model_fn += len(fn_model)
        for cid in fn_model:
            idx_arr = np.where(cids == cid)[0]
            if len(idx_arr) == 0:
                continue
            idx = idx_arr[0]
            fn_model_rows.append({
                "s1_id": s1_id,
                "cand_id": cid,
                "source": "S3" if s3[idx] else "S2",
                "prob": float(ps[idx]),
                "rank": int(idx) + 1,
                "cosine_sim": float(Xs[idx, 56]) if Xs.shape[1] > 56 else None,
                "top_prob": float(ps[0]),
                "top_minus_prob": float(ps[0] - ps[idx]),
                "n_cands": int(len(ps)),
                **{FEATURE_NAMES[j]: float(Xs[idx, j]) for j in range(min(56, Xs.shape[1]))},
            })

        # FN blocker: in true_full but not in any candidate
        for cid in blocker_misses:
            total_blocker_misses += 1
            fn_blocker_rows.append({
                "s1_id": s1_id,
                "cand_id": cid,
                "source": "S3" if cid.startswith("S3") else "S2",
            })

        n_s1_in_npz += 1

    # Process val S1s NOT in NPZ at all (never entered blocker)
    for s1_id in val_s1_set:
        if s1_id not in npz_s1_set:
            n_s1_not_in_npz += 1
            true_full = gt_full.get(s1_id, set())
            eval_A.update(s1_id, set(), set())     # no candidates, no predictions
            eval_B.update(s1_id, true_full, set()) # all true matches are FNs
            for cid in true_full:
                total_blocker_misses += 1
                fn_blocker_rows.append({
                    "s1_id": s1_id,
                    "cand_id": cid,
                    "source": "S3" if cid.startswith("S3") else "S2",
                })

    mA = eval_A.compute()
    mB = eval_B.compute()
    elapsed = time.time() - t0

    print(f"\nEvaluation done in {elapsed:.1f}s", flush=True)

    print("\n" + "=" * 70)
    print("AUTHORITATIVE EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Val S1 entities total         : {n_s1_total:,}")
    print(f"  S1 entities with candidates   : {n_s1_in_npz:,}")
    print(f"  S1 entities with NO candidates: {n_s1_not_in_npz:,}")
    print()
    print(f"  METRIC A — Decision-layer (candidates only):")
    print(f"    Macro F0.5    : {mA['macro_f_05']:.6f}")
    print(f"    Precision     : {mA['macro_precision']:.6f}")
    print(f"    Recall        : {mA['macro_recall']:.6f}")
    print(f"    Singleton Acc : {mA['singleton_accuracy']:.6f}")
    print(f"    Exact Set Acc : {mA['exact_set_accuracy']:.6f}")
    print(f"    False Positives: {mA['total_false_positives']:,}")
    print(f"    False Negatives: {mA['total_false_negatives']:,}")
    print()
    print(f"  METRIC B — Complete system (blocker misses as FN):")
    print(f"    Macro F0.5    : {mB['macro_f_05']:.6f}   ← THE REAL NUMBER")
    print(f"    Precision     : {mB['macro_precision']:.6f}")
    print(f"    Recall        : {mB['macro_recall']:.6f}")
    print(f"    Singleton Acc : {mB['singleton_accuracy']:.6f}")
    print(f"    Exact Set Acc : {mB['exact_set_accuracy']:.6f}")
    print(f"    False Positives: {mB['total_false_positives']:,}")
    print(f"    False Negatives: {mB['total_false_negatives']:,}")
    print()
    print(f"  ERROR BREAKDOWN:")
    print(f"    FP (decision layer)  : {total_fp:,}")
    print(f"    FN (model miss)      : {total_model_fn:,}")
    print(f"    FN (blocker miss)    : {total_blocker_misses:,}")
    print(f"    FN total             : {total_model_fn + total_blocker_misses:,}")
    print("=" * 70)

    # Save results
    result = {
        "experiment": "EXP_N008",
        "eval_population": n_s1_total,
        "metric_A_decision_layer": {
            "macro_f_05": mA["macro_f_05"],
            "macro_precision": mA["macro_precision"],
            "macro_recall": mA["macro_recall"],
            "singleton_accuracy": mA["singleton_accuracy"],
            "exact_set_accuracy": mA["exact_set_accuracy"],
            "false_positives": mA["total_false_positives"],
            "false_negatives": mA["total_false_negatives"],
        },
        "metric_B_complete_system": {
            "macro_f_05": mB["macro_f_05"],
            "macro_precision": mB["macro_precision"],
            "macro_recall": mB["macro_recall"],
            "singleton_accuracy": mB["singleton_accuracy"],
            "exact_set_accuracy": mB["exact_set_accuracy"],
            "false_positives": mB["total_false_positives"],
            "false_negatives": mB["total_false_negatives"],
        },
        "error_breakdown": {
            "fp_decision": total_fp,
            "fn_model": total_model_fn,
            "fn_blocker": total_blocker_misses,
        },
        "params": {"base_th": BT, "dense_th": DT, "margin": MG,
                   "anchor": AG, "bridge": BR, "density_cutoff": DC,
                   "blend_w_lgb": W_LGB, "blend_w_xgb": round(1 - W_LGB, 2)},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path = os.path.join(REPORTS_DIR, "authoritative_eval.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Save FP CSV
    fp_path = os.path.join(REPORTS_DIR, "exp_n008_errors_fp.csv")
    if fp_rows:
        with open(fp_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(fp_rows[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(fp_rows)
        print(f"Saved: {fp_path} ({len(fp_rows):,} FP rows)")

    # Save FN model CSV
    fn_m_path = os.path.join(REPORTS_DIR, "exp_n008_errors_fn_model.csv")
    if fn_model_rows:
        with open(fn_m_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(fn_model_rows[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(fn_model_rows)
        print(f"Saved: {fn_m_path} ({len(fn_model_rows):,} FN-model rows)")

    # Save FN blocker CSV
    fn_b_path = os.path.join(REPORTS_DIR, "exp_n008_errors_fn_blocker.csv")
    if fn_blocker_rows:
        with open(fn_b_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["s1_id", "cand_id", "source"])
            writer.writeheader()
            writer.writerows(fn_blocker_rows)
        print(f"Saved: {fn_b_path} ({len(fn_blocker_rows):,} FN-blocker rows)")

    print(f"\nTotal runtime: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
