#!/usr/bin/env python3
"""
Reusable Authoritative Evaluator — callable from any experiment script.

Usage:
    from scripts.authoritative_eval_fn import run_authoritative_eval

    metrics = run_authoritative_eval(
        decision_fn=my_decision_fn,  # (ps, cids, is_s3) -> Set[str]
        val_npz_path="experiments/val_pairs_n009.npz",
        probs_lgb_path="experiments/probs_n009_lgb.npy",
        probs_xgb_path="experiments/probs_n009_xgb.npy",
        w_lgb=0.45,
        label="EXP_N009",
    )
    # metrics["B"]["macro_f_05"] is the real complete-system metric

Returns dict with keys "A" and "B", each a metrics dict from Evaluator.compute().
"""

import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Callable, Dict, Optional, Set, Tuple

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONUNBUFFERED"] = "1"

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CODE_DIR, "..", ".."))
sys.path.insert(0, os.path.join(_CODE_DIR, "src"))

from evaluation import Evaluator

_EXPERIMENTS_DIR = os.path.join(_CODE_DIR, "experiments")
_TRAIN_DIR = os.path.join(_PROJECT_ROOT, "dataset", "train")
_VAL_IDS_PATH = os.path.join(_EXPERIMENTS_DIR, "validation_s1_ids.txt")
_SEED = 42
_VAL_SAMPLE = 25000


def _load_val_s1_ids(seed: int = _SEED, sample: int = _VAL_SAMPLE) -> Set[str]:
    random.seed(seed)
    with open(_VAL_IDS_PATH) as f:
        pool = [l.strip() for l in f if l.strip()]
    return set(random.sample(pool, min(sample, len(pool))))


def _load_gt(val_s1_set: Set[str]) -> Dict[str, Set[str]]:
    """Stream ground truth TSV — O(n) single pass."""
    gt: Dict[str, Set[str]] = {}
    path = os.path.join(_TRAIN_DIR, "train_ground_truth.tsv")
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1 = parts[0]
            if s1 in val_s1_set:
                mids = set(parts[1].split(",")) if len(parts) > 1 and parts[1].strip() else set()
                gt[s1] = mids
    # Fill singletons not found in GT
    for s1 in val_s1_set:
        if s1 not in gt:
            gt[s1] = set()
    return gt


def run_authoritative_eval(
    val_npz_path: str,
    probs_lgb_path: str,
    probs_xgb_path: str,
    w_lgb: float = 0.45,
    label: str = "EXP",
    base_th: float = 0.956,
    dense_th: float = 0.963,
    margin: float = 0.025,
    anchor: float = 0.978,
    bridge: float = 0.938,
    density_cutoff: int = 75,
    decision_fn: Optional[Callable] = None,
    verbose: bool = True,
) -> Dict:
    """
    Run authoritative A+B evaluation.

    Parameters
    ----------
    val_npz_path       : path to val_pairs_XXXX.npz
    probs_lgb_path     : path to saved LGB probabilities (float32 array)
    probs_xgb_path     : path to saved XGB probabilities (float32 array)
    w_lgb              : LGB blend weight (XGB weight = 1 - w_lgb)
    label              : experiment name for logging
    base_th .. density_cutoff : bridge decision parameters
    decision_fn        : optional override — fn(ps, cids, is_s3, params) -> Set[str]
    verbose            : print progress

    Returns
    -------
    {
        "A": { macro_f_05, macro_precision, macro_recall, ... },
        "B": { macro_f_05, macro_precision, macro_recall, ... },
        "errors": {
            "fp": int,
            "fn_model": int,
            "fn_blocker": int,
        },
        "n_entities": int,
        "n_in_npz": int,
        "n_not_in_npz": int,
    }
    """
    t0 = time.time()
    if verbose:
        print(f"[AuthEval:{label}] Loading...", flush=True)

    val_s1_set = _load_val_s1_ids()
    gt_full = _load_gt(val_s1_set)

    va = np.load(val_npz_path, allow_pickle=True)
    probs_lgb = np.load(probs_lgb_path).astype(np.float32)
    probs_xgb = np.load(probs_xgb_path).astype(np.float32)
    probs = w_lgb * probs_lgb + (1 - w_lgb) * probs_xgb

    # Vectorized grouping
    s1_ids_str   = va["s1_ids"].astype(str)
    cand_ids_str = va["cand_ids"].astype(str)
    sources_str  = va["sources"].astype(str)
    y            = va["y"].astype(np.int8)

    unique_s1, s1_codes = np.unique(s1_ids_str, return_inverse=True)
    sort_key = np.lexsort((-probs, s1_codes))
    s1_codes_s  = s1_codes[sort_key]
    probs_s     = probs[sort_key]
    cand_ids_s  = cand_ids_str[sort_key]
    sources_s   = sources_str[sort_key]
    y_s         = y[sort_key]

    boundaries = np.where(np.diff(s1_codes_s) != 0)[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends   = np.concatenate([boundaries, [len(probs_s)]])

    npz_s1_set = set(unique_s1.tolist())
    sg = base_th * 0.85

    def _default_decision(ps_, cids_, is_s3_):
        N_ = len(ps_)
        if N_ == 0 or ps_[0] < sg:
            return set()
        top = ps_[0]
        et = dense_th if N_ > density_cutoff else base_th
        am = (ps_ >= et) & ((top - ps_) <= margin)
        anm = ps_ >= anchor
        if np.any(anm & am):
            if not bool(np.any(am & is_s3_)):
                am = am | ((~am) & is_s3_ & (ps_ >= bridge))
            if not bool(np.any(am & ~is_s3_)):
                am = am | ((~am) & ~is_s3_ & (ps_ >= bridge))
        return set(cids_[am])

    decide = decision_fn if decision_fn is not None else _default_decision

    eval_A = Evaluator()
    eval_B = Evaluator()
    total_fp = total_fn_model = total_fn_blocker = 0
    n_in_npz = 0

    for k, s1_id in enumerate(unique_s1):
        sl = slice(starts[k], ends[k])
        ps    = probs_s[sl]
        cids  = cand_ids_s[sl]
        s3    = sources_s[sl] == "S3"
        ys    = y_s[sl]

        true_full    = gt_full.get(s1_id, set())
        cand_pool    = set(cids.tolist())
        true_in_pool = true_full & cand_pool
        blocker_miss = true_full - cand_pool

        predicted = decide(ps, cids, s3)

        eval_A.update(s1_id, true_in_pool, predicted)
        eval_B.update(s1_id, true_full, predicted)

        total_fp        += len(predicted - true_full)
        total_fn_model  += len(true_in_pool - predicted)
        total_fn_blocker += len(blocker_miss)
        n_in_npz += 1

    # S1s not in NPZ at all
    n_not_in_npz = 0
    for s1_id in val_s1_set:
        if s1_id not in npz_s1_set:
            true_full = gt_full.get(s1_id, set())
            eval_A.update(s1_id, set(), set())
            eval_B.update(s1_id, true_full, set())
            total_fn_blocker += len(true_full)
            n_not_in_npz += 1

    mA = eval_A.compute()
    mB = eval_B.compute()
    elapsed = time.time() - t0

    if verbose:
        print(f"[AuthEval:{label}] A={mA['macro_f_05']:.6f}  B={mB['macro_f_05']:.6f}  "
              f"FP={total_fp}  FN-model={total_fn_model}  FN-blocker={total_fn_blocker}  "
              f"({elapsed:.1f}s)", flush=True)

    return {
        "A": mA,
        "B": mB,
        "errors": {
            "fp": total_fp,
            "fn_model": total_fn_model,
            "fn_blocker": total_fn_blocker,
        },
        "n_entities": len(val_s1_set),
        "n_in_npz": n_in_npz,
        "n_not_in_npz": n_not_in_npz,
    }


if __name__ == "__main__":
    # Self-test: reproduce EXP_N008 authoritative eval
    result = run_authoritative_eval(
        val_npz_path=os.path.join(_EXPERIMENTS_DIR, "val_pairs_n008.npz"),
        probs_lgb_path=os.path.join(_EXPERIMENTS_DIR, "probs_n008_lgb.npy"),
        probs_xgb_path=os.path.join(_EXPERIMENTS_DIR, "probs_n008_xgb.npy"),
        w_lgb=0.45,
        label="EXP_N008_selftest",
    )
    print(f"\nA  macro_f_05 = {result['A']['macro_f_05']:.6f}  (expected ~0.986154)")
    print(f"B  macro_f_05 = {result['B']['macro_f_05']:.6f}  (expected ~0.972440)")
    assert abs(result["B"]["macro_f_05"] - 0.972440) < 0.001, "Self-test FAILED"
    print("Self-test PASSED")
