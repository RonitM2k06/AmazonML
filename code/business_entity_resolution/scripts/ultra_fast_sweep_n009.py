#!/usr/bin/env python3
"""
Hyper-Fast Pre-Indexed Authoritative Sweep for EXP_N009.
Pre-loads NPZ and GT once into memory.
Per-config time: 0.05s (Total 20 seconds).
"""
import os
import sys
import time
import json
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONUNBUFFERED"] = "1"

script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(code_dir, "src"))

from evaluation import Evaluator
from authoritative_eval_fn import _load_val_s1_ids, _load_gt

EXPERIMENTS_DIR = os.path.join(code_dir, "experiments")
VAL_NPZ = os.path.join(EXPERIMENTS_DIR, "val_pairs_n009.npz")
PROBS_LGB = os.path.join(EXPERIMENTS_DIR, "probs_n009_lgb.npy")
PROBS_XGB = os.path.join(EXPERIMENTS_DIR, "probs_n009_xgb.npy")

def main():
    t0 = time.time()
    print("=" * 70, flush=True)
    print("[EXP_N009] Hyper-Fast Authoritative Parameter Sweep...", flush=True)
    print("=" * 70, flush=True)

    print("Loading datasets and pre-indexing...", flush=True)
    val_s1_set = _load_val_s1_ids()
    gt_full = _load_gt(val_s1_set)

    va = np.load(VAL_NPZ, allow_pickle=True)
    probs_lgb = np.load(PROBS_LGB).astype(np.float32)
    probs_xgb = np.load(PROBS_XGB).astype(np.float32)

    s1_ids_str = va["s1_ids"].astype(str)
    cand_ids_str = va["cand_ids"].astype(str)
    sources_str = va["sources"].astype(str)
    unique_s1, s1_codes = np.unique(s1_ids_str, return_inverse=True)

    print(f"Data cached in {time.time()-t0:.1f}s. Running grid...", flush=True)

    # Blend weights to test
    blend_weights = [0.45, 0.50, 0.55]
    base_ths = [0.945, 0.950, 0.956, 0.962]
    dense_ths = [0.958, 0.965, 0.970]
    margins = [0.020, 0.025, 0.035]
    anchors = [0.970, 0.978]
    bridges = [0.930, 0.938]
    cutoffs = [75]

    best_B = 0.0
    best_config = None
    best_res = None

    for w_lgb in blend_weights:
        probs = w_lgb * probs_lgb + (1.0 - w_lgb) * probs_xgb
        sort_key = np.lexsort((-probs, s1_codes))
        s1_codes_s = s1_codes[sort_key]
        probs_s = probs[sort_key]
        cand_ids_s = cand_ids_str[sort_key]
        sources_s = sources_str[sort_key]

        boundaries = np.where(np.diff(s1_codes_s) != 0)[0] + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(probs_s)]])

        for base_th in base_ths:
            for dense_th in dense_ths:
                if dense_th <= base_th:
                    continue
                for margin in margins:
                    for anchor in anchors:
                        for bridge in bridges:
                            for dc in cutoffs:
                                sg = base_th * 0.85
                                eval_A = Evaluator()
                                eval_B = Evaluator()
                                fp = fn_m = fn_b = 0

                                for k, s1_id in enumerate(unique_s1):
                                    sl = slice(starts[k], ends[k])
                                    ps = probs_s[sl]
                                    cids = cand_ids_s[sl]
                                    is_s3 = (sources_s[sl] == "S3")

                                    true_full = gt_full.get(s1_id, set())
                                    cand_pool = set(cids.tolist())
                                    true_in_pool = true_full & cand_pool
                                    blocker_miss = true_full - cand_pool

                                    # Decision
                                    N_ = len(ps)
                                    if N_ == 0 or ps[0] < sg:
                                        pred = set()
                                    else:
                                        top = ps[0]
                                        et = dense_th if N_ > dc else base_th
                                        am = (ps >= et) & ((top - ps) <= margin)
                                        anm = ps >= anchor
                                        if np.any(anm & am):
                                            if not bool(np.any(am & is_s3)):
                                                am = am | ((~am) & is_s3 & (ps >= bridge))
                                            if not bool(np.any(am & ~is_s3)):
                                                am = am | ((~am) & ~is_s3 & (ps >= bridge))
                                        pred = set(cids[am])

                                    eval_A.update(s1_id, true_in_pool, pred)
                                    eval_B.update(s1_id, true_full, pred)
                                    fp += len(pred - true_full)
                                    fn_m += len(true_in_pool - pred)
                                    fn_b += len(blocker_miss)

                                # S1s not in NPZ
                                npz_set = set(unique_s1.tolist())
                                for s1_id in val_s1_set:
                                    if s1_id not in npz_set:
                                        true_full = gt_full.get(s1_id, set())
                                        eval_A.update(s1_id, set(), set())
                                        eval_B.update(s1_id, true_full, set())
                                        fn_b += len(true_full)

                                mB = eval_B.compute()
                                b_f05 = mB["macro_f_05"]
                                if b_f05 > best_B:
                                    best_B = b_f05
                                    mA = eval_A.compute()
                                    best_config = {
                                        "w_lgb": w_lgb,
                                        "base_th": base_th,
                                        "dense_th": dense_th,
                                        "margin": margin,
                                        "anchor": anchor,
                                        "bridge": bridge,
                                        "density_cutoff": dc,
                                    }
                                    best_res = {
                                        "A": mA,
                                        "B": mB,
                                        "fp": fp,
                                        "fn_model": fn_m,
                                        "fn_blocker": fn_b,
                                    }
                                    print(f"[BEST] Metric B = {b_f05:.6f} (A={mA['macro_f_05']:.6f}) | Config: {best_config}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"Sweep completed in {time.time()-t0:.1f}s", flush=True)
    print(f"CHAMPION METRIC B = {best_B:.6f}", flush=True)
    print(f"Metrics Summary:")
    print(f"  Metric A (Decision-only)   : {best_res['A']['macro_f_05']:.6f}")
    print(f"  Metric B (Complete System) : {best_res['B']['macro_f_05']:.6f}")
    print(f"  Precision B: {best_res['B']['macro_precision']:.6f}, Recall B: {best_res['B']['macro_recall']:.6f}")
    print(f"  FP: {best_res['fp']}, FN-model: {best_res['fn_model']}, FN-blocker: {best_res['fn_blocker']}")
    print(f"Optimal parameters: {json.dumps(best_config, indent=2)}", flush=True)

if __name__ == "__main__":
    main()
