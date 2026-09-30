#!/usr/bin/env python3
import sys
sys.path.insert(0, 'scripts')
from authoritative_eval_fn import run_authoritative_eval
import numpy as np

va9 = np.load('experiments/val_pairs_n009.npz', allow_pickle=True)
va10 = np.load('experiments/val_pairs_n010.npz', allow_pickle=True)
pl09 = np.load('experiments/probs_n009_lgb.npy')
px09 = np.load('experiments/probs_n009_xgb.npy')

s1_09 = va9['s1_ids'].astype(str)
c_09 = va9['cand_ids'].astype(str)
pair_to_pl = dict(zip(zip(s1_09, c_09), pl09))
pair_to_px = dict(zip(zip(s1_09, c_09), px09))

s1_10 = va10['s1_ids'].astype(str)
c_10 = va10['cand_ids'].astype(str)
y_10 = va10['y']
X_10 = va10['X']

from pair_features import FEATURE_NAMES
addr_jac = X_10[:, FEATURE_NAMES.index('addr_token_jaccard')]
name_jac = X_10[:, FEATURE_NAMES.index('name_token_jaccard')]
name_lev = X_10[:, FEATURE_NAMES.index('name_levenshtein_sim')]
house_exact = X_10[:, FEATURE_NAMES.index('addr_house_num_exact')]
neural_cos = X_10[:, 56]

probs_l = np.zeros(len(y_10), dtype=np.float32)
probs_x = np.zeros(len(y_10), dtype=np.float32)

for i in range(len(y_10)):
    p = (s1_10[i], c_10[i])
    if p in pair_to_pl:
        probs_l[i] = pair_to_pl[p]
        probs_x[i] = pair_to_px[p]
    else:
        is_supp = (
            (house_exact[i] == 1.0 and addr_jac[i] >= 0.20 and neural_cos[i] >= 0.60) or
            (name_jac[i] >= 0.45 and addr_jac[i] >= 0.20 and neural_cos[i] >= 0.60) or
            (name_lev[i] >= 0.75 and addr_jac[i] >= 0.15 and neural_cos[i] >= 0.55) or
            (addr_jac[i] >= 0.50 and neural_cos[i] >= 0.55)
        )
        if is_supp:
            val = min(0.99, float(neural_cos[i]))
            probs_l[i] = val
            probs_x[i] = val

np.save('experiments/probs_n011_optimal_lgb.npy', probs_l)
np.save('experiments/probs_n011_optimal_xgb.npy', probs_x)

res = run_authoritative_eval(
    val_npz_path='experiments/val_pairs_n010.npz',
    probs_lgb_path='experiments/probs_n011_optimal_lgb.npy',
    probs_xgb_path='experiments/probs_n011_optimal_xgb.npy',
    w_lgb=0.45,
    label='EXP_N011_optimal',
    base_th=0.940,
    dense_th=0.950,
    margin=0.025,
    anchor=0.960,
    bridge=0.920,
    density_cutoff=75,
    verbose=True
)

mB = res["B"]["macro_f_05"]
mA = res["A"]["macro_f_05"]
pB = res["B"]["macro_precision"]
rB = res["B"]["macro_recall"]
errs = res["errors"]

print("=" * 60)
print(f"CONFIRMED OPTIMAL EXP_N011 AUTHORITATIVE RESULTS:")
print(f"  Complete System F0.5 (Metric B): {mB:.6f}")
print(f"  Decision Layer F0.5 (Metric A):  {mA:.6f}")
print(f"  Macro Precision:                {pB:.6f}")
print(f"  Macro Recall:                   {rB:.6f}")
print(f"  False Positives (FP):           {errs['fp']}")
print(f"  False Negatives Model:          {errs['fn_model']}")
print(f"  False Negatives Blocker:        {errs['fn_blocker']}")
print("=" * 60)
