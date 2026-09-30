#!/usr/bin/env python3
import sys, os
sys.path.insert(0, 'scripts')
from authoritative_eval_fn import run_authoritative_eval

res = run_authoritative_eval(
    val_npz_path='experiments/val_pairs_n009.npz',
    probs_lgb_path='experiments/probs_n009_lgb.npy',
    probs_xgb_path='experiments/probs_n009_xgb.npy',
    w_lgb=0.45,
    label='EXP_N009_eval',
    base_th=0.950,
    dense_th=0.965,
    margin=0.025,
    anchor=0.970,
    bridge=0.930,
    density_cutoff=75,
)
print('--- Results ---')
print(f"Metric A (Decision-only): {res['A']['macro_f_05']:.6f}")
print(f"Metric B (Complete System): {res['B']['macro_f_05']:.6f}")
print(f"Precision B: {res['B']['macro_precision']:.6f}, Recall B: {res['B']['macro_recall']:.6f}")
print(f"FP: {res['errors']['fp']}, FN-model: {res['errors']['fn_model']}, FN-blocker: {res['errors']['fn_blocker']}")
