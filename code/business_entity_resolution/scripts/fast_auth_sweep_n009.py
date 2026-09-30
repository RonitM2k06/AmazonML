#!/usr/bin/env python3
"""
Authoritative Focused Grid Sweep for EXP_N009.
Uses authoritative_eval_fn directly to guarantee 100% metric fidelity.
Takes ~40-60 seconds total.
"""
import os
import sys
import time
import json
import itertools
from authoritative_eval_fn import run_authoritative_eval

script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
EXPERIMENTS_DIR = os.path.join(code_dir, "experiments")
VAL_NPZ = os.path.join(EXPERIMENTS_DIR, "val_pairs_n009.npz")
PROBS_LGB = os.path.join(EXPERIMENTS_DIR, "probs_n009_lgb.npy")
PROBS_XGB = os.path.join(EXPERIMENTS_DIR, "probs_n009_xgb.npy")

def main():
    t0 = time.time()
    print("=" * 70, flush=True)
    print("[EXP_N009] Running Authoritative Grid Sweep...", flush=True)
    print("=" * 70, flush=True)

    # High-impact targeted configurations centered around the verified optimum
    weights = [0.45, 0.50, 0.55]
    base_ths = [0.945, 0.950, 0.955, 0.960]
    dense_ths = [0.955, 0.962, 0.968]
    margins = [0.020, 0.025, 0.035]
    anchors = [0.970, 0.978]
    bridges = [0.930, 0.938]
    cutoffs = [75]

    configs = []
    for w in weights:
        for bt in base_ths:
            for dt in dense_ths:
                if dt <= bt:
                    continue
                for mg in margins:
                    for ag in anchors:
                        for br in bridges:
                            for dc in cutoffs:
                                configs.append((w, bt, dt, mg, ag, br, dc))

    print(f"Total configurations to evaluate: {len(configs)}", flush=True)

    best_B = 0.0
    best_res = None
    best_params = None

    for idx, (w, bt, dt, mg, ag, br, dc) in enumerate(configs):
        res = run_authoritative_eval(
            val_npz_path=VAL_NPZ,
            probs_lgb_path=PROBS_LGB,
            probs_xgb_path=PROBS_XGB,
            w_lgb=w,
            base_th=bt,
            dense_th=dt,
            margin=mg,
            anchor=ag,
            bridge=br,
            density_cutoff=dc,
            verbose=False,
        )
        b_f05 = res["B"]["macro_f_05"]
        if b_f05 > best_B:
            best_B = b_f05
            best_res = res
            best_params = {
                "w_lgb": w,
                "base_th": bt,
                "dense_th": dt,
                "margin": mg,
                "anchor": ag,
                "bridge": br,
                "density_cutoff": dc,
            }
            print(f"[{idx+1}/{len(configs)}] NEW BEST Metric B = {b_f05:.6f} (A={res['A']['macro_f_05']:.6f}) | Params: {best_params}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"Grid sweep finished in {time.time()-t0:.1f}s", flush=True)
    print(f"Champion Metric B = {best_B:.6f}", flush=True)
    print("Metrics summary:")
    print(f"  Metric A (Decision-only)   : {best_res['A']['macro_f_05']:.6f}")
    print(f"  Metric B (Complete System) : {best_res['B']['macro_f_05']:.6f}")
    print(f"  Precision B: {best_res['B']['macro_precision']:.6f}, Recall B: {best_res['B']['macro_recall']:.6f}")
    print(f"  FP: {best_res['errors']['fp']}, FN-model: {best_res['errors']['fn_model']}, FN-blocker: {best_res['errors']['fn_blocker']}")
    print(f"Best parameters: {json.dumps(best_params, indent=2)}", flush=True)

if __name__ == "__main__":
    main()
