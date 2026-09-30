#!/usr/bin/env python3
"""
EXP_N010: High-Precision Threshold Sweep to Reach Metric B >= 0.9950.

Performs vectorised grid search over:
- base_th (0.940 - 0.975)
- dense_th (0.955 - 0.985)
- margin (0.015 - 0.035)
- anchor (0.965 - 0.985)
- bridge (0.910 - 0.945)
- blend weight (LGB vs XGB)
"""
import sys
import os
sys.path.insert(0, 'scripts')
from authoritative_eval_fn import run_authoritative_eval

VAL_NPZ = 'experiments/val_pairs_n010.npz'
PROBS_LGB = 'experiments/probs_n010_lgb.npy'
PROBS_XGB = 'experiments/probs_n010_xgb.npy'

def main():
    print("=" * 70)
    print("[EXP_N010] Running Authoritative Metric B Evaluation & Threshold Tuning")
    print("=" * 70)

    best_score = 0.0
    best_config = None

    configs = [
        {"w_lgb": 0.45, "base_th": 0.950, "dense_th": 0.965, "margin": 0.025, "anchor": 0.970, "bridge": 0.930, "density_cutoff": 75},
        {"w_lgb": 0.40, "base_th": 0.955, "dense_th": 0.968, "margin": 0.020, "anchor": 0.975, "bridge": 0.935, "density_cutoff": 75},
        {"w_lgb": 0.50, "base_th": 0.952, "dense_th": 0.965, "margin": 0.025, "anchor": 0.972, "bridge": 0.932, "density_cutoff": 60},
        {"w_lgb": 0.35, "base_th": 0.960, "dense_th": 0.972, "margin": 0.020, "anchor": 0.978, "bridge": 0.938, "density_cutoff": 60},
        {"w_lgb": 0.45, "base_th": 0.962, "dense_th": 0.975, "margin": 0.018, "anchor": 0.980, "bridge": 0.940, "density_cutoff": 50},
        {"w_lgb": 0.40, "base_th": 0.948, "dense_th": 0.962, "margin": 0.025, "anchor": 0.968, "bridge": 0.925, "density_cutoff": 80},
    ]

    for cfg in configs:
        res = run_authoritative_eval(
            val_npz_path=VAL_NPZ,
            probs_lgb_path=PROBS_LGB,
            probs_xgb_path=PROBS_XGB,
            w_lgb=cfg["w_lgb"],
            label="EXP_N010",
            base_th=cfg["base_th"],
            dense_th=cfg["dense_th"],
            margin=cfg["margin"],
            anchor=cfg["anchor"],
            bridge=cfg["bridge"],
            density_cutoff=cfg["density_cutoff"],
            verbose=False,
        )
        mB = res["B"]["macro_f_05"]
        mA = res["A"]["macro_f_05"]
        prec = res["B"]["macro_precision"]
        rec = res["B"]["macro_recall"]
        errs = res["errors"]
        print(f"Config {cfg} -> Metric B: {mB:.6f} (Metric A: {mA:.6f}, P: {prec:.4f}, R: {rec:.4f}, FP: {errs['fp']}, FN-model: {errs['fn_model']}, FN-blocker: {errs['fn_blocker']})")

        if mB > best_score:
            best_score = mB
            best_config = (cfg, res)

    print("\n" + "=" * 70)
    print(f"BEST CONFIG FOR EXP_N010: Metric B = {best_score:.6f}")
    print(f"Details: {best_config}")
    print("=" * 70)

if __name__ == "__main__":
    main()
