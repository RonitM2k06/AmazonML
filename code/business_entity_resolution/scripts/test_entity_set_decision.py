#!/usr/bin/env python3
"""
EXP_N015 — Phase 4 & 5 Probe: Entity-Level Set Decision with Context & Graph Features.
"""

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "scripts")
from authoritative_eval_fn import _load_val_s1_ids, _load_gt
from evaluation import Evaluator
from pathlib import Path

val_s1_set = _load_val_s1_ids()
gt_full = _load_gt(val_s1_set)

EXP_DIR = Path("experiments")
va = np.load(EXP_DIR / "val_pairs_n010.npz", allow_pickle=True)
pl = np.load(EXP_DIR / "probs_n011_optimal_lgb.npy")
px = np.load(EXP_DIR / "probs_n011_optimal_xgb.npy")
probs = 0.45 * pl + 0.55 * px

s1_ids_str = va["s1_ids"].astype(str)
cand_ids_str = va["cand_ids"].astype(str)
sources_str = va["sources"].astype(str)
y = va["y"].astype(np.int8)
X = va["X"].astype(np.float32)

unique_s1, s1_codes = np.unique(s1_ids_str, return_inverse=True)
sort_key = np.lexsort((-probs, s1_codes))
s1_codes_s = s1_codes[sort_key]
probs_s = probs[sort_key]
cand_ids_s = cand_ids_str[sort_key]
sources_s = sources_str[sort_key]
y_s = y[sort_key]
X_s = X[sort_key]

boundaries = np.where(np.diff(s1_codes_s) != 0)[0] + 1
starts = np.concatenate([[0], boundaries])
ends = np.concatenate([boundaries, [len(probs_s)]])

print("Testing Entity Context Decision Rules across multiple parameters...")

# Feature columns in X:
# 11: name_token_jaccard, 30: addr_token_jaccard, 40: addr_house_num_exact, 29: addr_norm_exact, 56: neural_cos

# Probe grid of bridge_th and margin
for base_th in [0.935, 0.940, 0.945]:
    for cross_bridge_th in [0.880, 0.900, 0.910, 0.920]:
        for max_sec_margin in [0.020, 0.025, 0.030]:
            eval_B = Evaluator()
            total_fp = total_fn_model = total_fn_blocker = 0

            for k in range(len(unique_s1)):
                sl = slice(starts[k], ends[k])
                ps = probs_s[sl]
                cids = cand_ids_s[sl]
                s3 = sources_s[sl] == "S3"
                ys = y_s[sl]
                xs = X_s[sl]
                N_ = len(ps)

                true_full = gt_full.get(unique_s1[k], set())
                cand_pool = set(cids.tolist())
                true_in_pool = true_full & cand_pool
                blocker_miss = true_full - cand_pool

                if N_ == 0 or ps[0] < base_th * 0.85:
                    pred_set = set()
                else:
                    top = ps[0]
                    # Base selection
                    am = (ps >= base_th) & ((top - ps) <= max_sec_margin)
                    
                    # Graph Anchor & Cross-source Bridge
                    if top >= 0.96:
                        anchor_is_s3 = s3[0]
                        # Bridge to other source if not yet represented
                        has_s3 = np.any(am & s3)
                        has_s2 = np.any(am & (~s3))

                        if not has_s3:
                            # candidate in S3 with strong cross-source support
                            cand_mask = (~am) & s3 & (ps >= cross_bridge_th)
                            if np.any(cand_mask):
                                # check consistency
                                for idx in np.where(cand_mask)[0]:
                                    if xs[idx, 40] == 1.0 or xs[idx, 30] >= 0.30 or xs[idx, 11] >= 0.40:
                                        am[idx] = True
                                        break  # take the best consistent one

                        if not has_s2:
                            cand_mask = (~am) & (~s3) & (ps >= cross_bridge_th)
                            if np.any(cand_mask):
                                for idx in np.where(cand_mask)[0]:
                                    if xs[idx, 40] == 1.0 or xs[idx, 30] >= 0.30 or xs[idx, 11] >= 0.40:
                                        am[idx] = True
                                        break

                    # Contradiction filter for secondary predictions:
                    # If secondary candidate has different house number and low address similarity, reject it
                    if len(np.where(am)[0]) > 1 and ps[0] >= 0.96 and xs[0, 40] == 1.0 and xs[0, 30] >= 0.40:
                        for idx in np.where(am)[0]:
                            if idx > 0 and xs[idx, 40] == 0.0 and xs[idx, 30] < 0.15:
                                am[idx] = False

                    pred_set = set(cids[am])

                eval_B.update(unique_s1[k], true_full, pred_set)
                total_fp += len(pred_set - true_full)
                total_fn_model += len(true_in_pool - pred_set)
                total_fn_blocker += len(blocker_miss)

            mB = eval_B.compute()
            f05 = mB["macro_f_05"]
            prec = mB["macro_precision"]
            rec = mB["macro_recall"]
            print(f"base_th={base_th:.3f}, bridge={cross_bridge_th:.3f}, margin={max_sec_margin:.3f} -> Metric B: {f05:.6f} | P: {prec:.4f} | R: {rec:.4f} | FP: {total_fp} | FN-mod: {total_fn_model}")
