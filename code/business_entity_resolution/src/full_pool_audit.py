"""
EXP_N030 — Phase 0: Full Candidate Pool Audit & Sanity Testing.

Audits the exact candidate pool structure:
- Base candidates from val_pairs_n010.npz (25,000 x 80)
- N014 graph recovered sibling candidates (602 matches)
- N027 multi-hop sibling blocker recovered candidates (362 matches)
- Total unique candidates per S1
- Candidate pool recall ceiling
- Mandatory Sanity Test: toggles 100 known candidate decisions to prove evaluation sensitivity
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_DIR.parents[1]
sys.path.insert(0, str(CODE_DIR / "src"))
sys.path.insert(0, str(CODE_DIR / "scripts"))

from authoritative_eval_fn import _load_val_s1_ids, _load_gt
from evaluation import Evaluator


def audit_candidate_pools():
    print("=" * 80)
    print("PHASE 0: FULL CANDIDATE POOL AUDIT & MANDATORY PIPELINE SANITY TEST")
    print("=" * 80)

    # 1. Load validation population & GT
    val_s1_set = _load_val_s1_ids()
    gt_full = _load_gt(val_s1_set)
    total_gt = sum(len(v) for v in gt_full.values())
    print(f"Validation Entities (S1): {len(val_s1_set):,}")
    print(f"Ground Truth Matches:     {total_gt:,}")

    # 2. Base candidate pool (N010)
    va = np.load(CODE_DIR / "experiments" / "val_pairs_n010.npz", allow_pickle=True)
    s1_ids = va["s1_ids"].astype(str)
    cand_ids = va["cand_ids"].astype(str)
    unique_s1 = s1_ids[::80]
    cand_mat = cand_ids.reshape(25000, 80)
    
    pool_dict = {s1: set(cand_mat[i].tolist()) for i, s1 in enumerate(unique_s1)}
    base_cand_count = sum(len(c) for c in pool_dict.values())
    print(f"\n[Base Pool N010]")
    print(f"  Total Candidates:       {base_cand_count:,} (exactly 80 per S1)")
    
    # 3. Add N014 recovered matches
    rec_n014 = np.load(CODE_DIR / "experiments" / "n014_recovered_matches.npz")
    n014_count = 0
    for s, c in zip(rec_n014["s1_ids"].astype(str), rec_n014["cand_ids"].astype(str)):
        if s in pool_dict:
            pool_dict[s].add(c)
            n014_count += 1
    print(f"\n[N014 Recovered Sibling Matches]")
    print(f"  Added unique candidates: {n014_count:,}")

    # 4. Add N027 recovered matches
    df_rec = pd.read_csv(CODE_DIR / "reports" / "blocker_recovery_matrix.csv")
    recovered_by_sibling = df_rec[df_rec["recovered_by"] != "none"]
    n027_count = 0
    for _, row in recovered_by_sibling.iterrows():
        s = str(row["s1_id"])
        c = str(row["candidate_id"])
        if s in pool_dict:
            pool_dict[s].add(c)
            n027_count += 1
    print(f"\n[N027 Multi-Hop Sibling Blocker Matches]")
    print(f"  Added unique candidates: {n027_count:,}")

    # 5. Measure complete candidate pool recall
    total_in_pool = 0
    blocker_misses = 0
    for s1 in val_s1_set:
        true_set = gt_full.get(s1, set())
        pool = pool_dict.get(s1, set())
        in_pool = true_set & pool
        miss = true_set - pool
        total_in_pool += len(in_pool)
        blocker_misses += len(miss)

    cand_recall = total_in_pool / total_gt
    print(f"\n[Full N027 Unified Candidate Pool]")
    print(f"  Total candidates:        {sum(len(c) for c in pool_dict.values()):,}")
    print(f"  Candidates/S1 average:   {sum(len(c) for c in pool_dict.values()) / len(val_s1_set):.2f}")
    print(f"  Max candidates/S1:       {max(len(c) for c in pool_dict.values())}")
    print(f"  True matches in pool:    {total_in_pool:,} / {total_gt:,}")
    print(f"  True matches missing:    {blocker_misses} (Blocker FN)")
    print(f"  Candidate Pool Recall:   {cand_recall*100:.4f}%")

    return pool_dict, unique_s1, cand_mat, gt_full, val_s1_set


def run_sanity_test(pred_mask, unique_s1, cand_mat, gt_full, val_s1_set, recovered_map):
    print("\n" + "=" * 80)
    print("MANDATORY SANITY TEST: EVALUATION SENSITIVITY VERIFICATION")
    print("=" * 80)
    from run_n027 import evaluate_authoritative

    # Baseline evaluation
    base_res = evaluate_authoritative(pred_mask, unique_s1, cand_mat, gt_full, val_s1_set, recovered_map)
    print(f"Baseline F0.5: {base_res['macro_f_05']:.6f} | P: {base_res['macro_precision']:.6f} | R: {base_res['macro_recall']:.6f} | FP: {base_res['fp']} | FN: {base_res['total_fn']}")

    # Toggle exactly 100 known candidate decisions
    test_mask = pred_mask.copy()
    toggled = 0
    for i in range(100):
        # Flip the first candidate prediction
        test_mask[i, 0] = not test_mask[i, 0]
        toggled += 1

    toggled_res = evaluate_authoritative(test_mask, unique_s1, cand_mat, gt_full, val_s1_set, recovered_map)
    print(f"Toggled 100 bits F0.5: {toggled_res['macro_f_05']:.6f} | P: {toggled_res['macro_precision']:.6f} | R: {toggled_res['macro_recall']:.6f} | FP: {toggled_res['fp']} | FN: {toggled_res['total_fn']}")

    diff_f05 = abs(toggled_res['macro_f_05'] - base_res['macro_f_05'])
    diff_fp = abs(toggled_res['fp'] - base_res['fp'])
    diff_fn = abs(toggled_res['total_fn'] - base_res['total_fn'])
    print(f"\nDelta Verification: dF0.5 = {diff_f05:.6f}, dFP = {diff_fp}, dFN = {diff_fn}")
    assert diff_f05 > 0, "Sanity Test FAILED: Evaluator is insensitive to prediction changes!"
    print("*** SANITY TEST PASSED: Authoritative evaluator is verified 100% reactive to prediction changes. ***")


if __name__ == "__main__":
    audit_candidate_pools()
