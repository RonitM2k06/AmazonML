import sys
from pathlib import Path
sys.path.insert(0, str(Path("scripts")))
import numpy as np
import pandas as pd
from sim_n034_candidate import (
    unique_s1, c_s, bs_s, aj_s, neural_cos_s, p_s, hn_s,
    gt_full, training_edges, evaluate_authoritative, val_s1_set, recovered_map
)
from run_n035 import pred_J

pred_n036 = pred_J.copy()
added_tp = 0
added_fp = 0
for i in range(len(unique_s1)):
    s1 = unique_s1[i]
    if np.sum(pred_J[i]) > 0:
        continue
    true_s1 = gt_full.get(s1, set())
    for r in range(0, 3):
        if p_s[i, r] >= 0.940 and bs_s[i, r] == 0:
            pred_n036[i, r] = True
            if c_s[i, r] in true_s1:
                added_tp += 1
            else:
                added_fp += 1

res = evaluate_authoritative(pred_n036, unique_s1, c_s, gt_full, val_s1_set, recovered_map)
print(f"N036 Candidate Score:")
print(f"Added TP: {added_tp}, Added FP: {added_fp}")
print(f"Metric B: {res['macro_f_05']:.6f} | P: {res['macro_precision']:.6f} | R: {res['macro_recall']:.6f} | FP={res['fp']} | FN={res['total_fn']}")
