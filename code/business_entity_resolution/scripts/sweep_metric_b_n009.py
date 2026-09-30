#!/usr/bin/env python3
"""
Vectorized Decision Parameter Sweep for EXP_N009 on Complete System (Metric B).
Sweeps blend weight w_lgb, base_th, dense_th, margin, anchor, bridge, and density_cutoff
directly against authoritative ground truth B to find the optimal decision boundary.
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

EXPERIMENTS_DIR = os.path.join(code_dir, "experiments")
TRAIN_DIR = os.path.abspath(os.path.join(code_dir, "..", "..", "dataset", "train"))
VAL_NPZ = os.path.join(EXPERIMENTS_DIR, "val_pairs_n009.npz")
PROBS_LGB = os.path.join(EXPERIMENTS_DIR, "probs_n009_lgb.npy")
PROBS_XGB = os.path.join(EXPERIMENTS_DIR, "probs_n009_xgb.npy")
VAL_IDS_PATH = os.path.join(EXPERIMENTS_DIR, "validation_s1_ids.txt")

def main():
    t0 = time.time()
    print("=" * 70, flush=True)
    print("[EXP_N009] Fast Vectorized Decision Parameter Sweep on Metric B...", flush=True)
    print("=" * 70, flush=True)

    # Load 25k validation IDs
    with open(VAL_IDS_PATH) as f:
        val_s1_list = [l.strip() for l in f if l.strip()][:25000]
    val_s1_set = set(val_s1_list)

    # Load full ground truth for Metric B
    gt_full = {}
    with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts[0] in val_s1_set:
                gt_full[parts[0]] = set(parts[1].split(",")) if len(parts) > 1 and parts[1].strip() else set()

    # Load val pairs and raw probs
    va = np.load(VAL_NPZ, allow_pickle=True)
    s1_ids = va["s1_ids"].astype(str)
    cand_ids = va["cand_ids"].astype(str)
    sources = va["sources"].astype(str)
    is_s3_arr = (sources == "S3")

    p_lgb = np.load(PROBS_LGB).astype(np.float32)
    p_xgb = np.load(PROBS_XGB).astype(np.float32)

    unique_s1, s1_codes = np.unique(s1_ids, return_inverse=True)
    n_unique = len(unique_s1)
    print(f"Loaded {len(s1_ids):,} pairs across {n_unique:,} S1 entities.", flush=True)

    # Test blend weights and thresholds
    best_B = 0.0
    best_config = None

    # Sweep grid
    blend_weights = [0.40, 0.45, 0.50, 0.55]
    base_ths = [0.930, 0.940, 0.948, 0.952, 0.956, 0.960]
    dense_ths = [0.945, 0.955, 0.963, 0.970]
    margins = [0.020, 0.025, 0.035, 0.050]
    anchors = [0.965, 0.972, 0.978]
    bridges = [0.910, 0.925, 0.938]
    cutoffs = [60, 75, 100]

    # Pre-sort entities for fast per-entity slices
    for w_lgb in blend_weights:
        probs = (w_lgb * p_lgb + (1.0 - w_lgb) * p_xgb).astype(np.float32)
        sort_key = np.lexsort((-probs, s1_codes))
        s1_codes_s = s1_codes[sort_key]
        probs_s = probs[sort_key]
        cand_ids_s = cand_ids[sort_key]
        is_s3_s = is_s3_arr[sort_key]

        boundaries = np.where(np.diff(s1_codes_s) != 0)[0] + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(probs_s)]])

        # Prepack entities
        entity_data = []
        for k in range(n_unique):
            sl = slice(starts[k], ends[k])
            entity_data.append((
                unique_s1[k],
                probs_s[sl],
                cand_ids_s[sl],
                is_s3_s[sl],
                gt_full.get(unique_s1[k], set())
            ))

        # Sweep inside this blend
        for base_th in base_ths:
            for dense_th in dense_ths:
                if dense_th < base_th:
                    continue
                for margin in margins:
                    for anchor in anchors:
                        for bridge in bridges:
                            for dc in cutoffs:
                                sg = base_th * 0.85
                                sum_f05 = 0.0
                                for s1_id, ps, cids, is_s3, true_full in entity_data:
                                    n_c = len(ps)
                                    if n_c == 0 or ps[0] < sg:
                                        if not true_full:
                                            sum_f05 += 1.0
                                        continue
                                    top = ps[0]
                                    eff_th = dense_th if n_c > dc else base_th
                                    am = (ps >= eff_th) & ((top - ps) <= margin)
                                    anm = ps >= anchor
                                    if np.any(anm & am):
                                        if not bool(np.any(am & is_s3)):
                                            am = am | ((~am) & is_s3 & (ps >= bridge))
                                        if not bool(np.any(am & ~is_s3)):
                                            am = am | ((~am) & ~is_s3 & (ps >= bridge))
                                    pred = set(cids[am])
                                    n_pred = len(pred)
                                    if not true_full:
                                        if n_pred == 0:
                                            sum_f05 += 1.0
                                    elif n_pred > 0:
                                        tp = len(true_full & pred)
                                        if tp > 0:
                                            p = tp / n_pred
                                            r = tp / len(true_full)
                                            d = 0.25 * p + r
                                            if d > 0:
                                                sum_f05 += 1.25 * p * r / d

                                # Add S1 singletons not in NPZ
                                not_in_npz = len(val_s1_set) - n_unique
                                for s1_id in val_s1_set:
                                    if s1_id not in unique_s1 and not gt_full.get(s1_id):
                                        sum_f05 += 1.0

                                f05_b = sum_f05 / len(val_s1_set)
                                if f05_b > best_B:
                                    best_B = f05_b
                                    best_config = {
                                        "w_lgb": w_lgb,
                                        "base_th": base_th,
                                        "dense_th": dense_th,
                                        "margin": margin,
                                        "anchor": anchor,
                                        "bridge": bridge,
                                        "density_cutoff": dc,
                                        "metric_B_f05": f05_b,
                                    }
                                    print(f"[BEST] Metric B: {f05_b:.6f} with config {best_config}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"Sweep completed in {time.time()-t0:.1f}s", flush=True)
    print(f"Optimal Metric B Macro F0.5 = {best_B:.6f}", flush=True)
    print(f"Optimal Configuration: {json.dumps(best_config, indent=2)}", flush=True)

if __name__ == "__main__":
    main()
