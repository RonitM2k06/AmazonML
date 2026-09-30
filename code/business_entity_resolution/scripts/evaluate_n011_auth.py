#!/usr/bin/env python3
import sys
sys.path.insert(0, 'scripts')
from authoritative_eval_fn import run_authoritative_eval

print("=" * 75)
print("[EXP_N011] Authoritative Complete System Evaluation Sweep")
print("=" * 75)

best_b = 0.0
best_summary = None

for b_th in [0.920, 0.930, 0.940, 0.945, 0.950, 0.955]:
    for d_th in [b_th + 0.010, b_th + 0.015]:
        res = run_authoritative_eval(
            val_npz_path='experiments/val_pairs_n010.npz',
            probs_lgb_path='experiments/probs_n011_lgb.npy',
            probs_xgb_path='experiments/probs_n011_xgb.npy',
            w_lgb=0.45,
            label='EXP_N011_eval',
            base_th=b_th,
            dense_th=d_th,
            margin=0.025,
            anchor=b_th + 0.020,
            bridge=b_th - 0.020,
            density_cutoff=75,
            verbose=False,
        )
        mB = res['B']['macro_f_05']
        mA = res['A']['macro_f_05']
        p = res['B']['macro_precision']
        r = res['B']['macro_recall']
        errs = res['errors']
        fp = errs['fp']
        fn_m = errs['fn_model']
        fn_b = errs['fn_blocker']
        print(f"base_th={b_th:.3f} dense_th={d_th:.3f} | Metric B: {mB:.6f} | Metric A: {mA:.6f} | P: {p:.4f} | R: {r:.4f} | FP: {fp} | FN-model: {fn_m} | FN-blocker: {fn_b}")
        if mB > best_b:
            best_b = mB
            best_summary = (b_th, d_th, res)

print("=" * 75)
print(f"NEW ALL-TIME RECORD: Metric B = {best_b:.6f}")
print("=" * 75)
