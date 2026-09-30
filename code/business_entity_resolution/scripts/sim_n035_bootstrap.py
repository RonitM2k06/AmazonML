import sys
from pathlib import Path
sys.path.insert(0, str(Path("scripts")))
import numpy as np
import pandas as pd
from sim_clean_n034 import (
    pred_clean, unique_s1, c_s, bs_s, aj_s, neural_cos_s,
    gt_full, training_edges, evaluate_authoritative, val_s1_set, recovered_map
)

# Add cross-source mutual pair rule for unanchored entities (cos >= 0.70 & aj >= 0.35, r <= 1)
pred_n035_cand = pred_clean.copy()
added_tp_cross = 0
added_fp_cross = 0

for i in range(len(unique_s1)):
    s1 = unique_s1[i]
    if np.sum(pred_clean[i]) > 0:
        continue
    true_s1 = gt_full.get(s1, set())
    cands_i = c_s[i]
    for r1 in range(0, 2):
        c1 = cands_i[r1]
        for r2 in range(r1 + 1, 2):
            c2 = cands_i[r2]
            if c1[:2] != c2[:2] and c2 in training_edges.get(c1, set()):
                # cross-source mutual connection!
                for r_cur, c_cur in [(r1, c1), (r2, c2)]:
                    if neural_cos_s[i, r_cur] >= 0.70 and aj_s[i, r_cur] >= 0.35:
                        pred_n035_cand[i, r_cur] = True
                        if c_cur in true_s1:
                            added_tp_cross += 1
                        else:
                            added_fp_cross += 1

res = evaluate_authoritative(pred_n035_cand, unique_s1, c_s, gt_full, val_s1_set, recovered_map)
print(f"N035 Simulation with Cross-Source Mutual Anchor Bootstrap:")
print(f"Added Cross TP: {added_tp_cross}, Added Cross FP: {added_fp_cross}")
print(f"Metric B: {res['macro_f_05']:.6f} | P: {res['macro_precision']:.6f} | R: {res['macro_recall']:.6f} | FP={res['fp']} | FN={res['total_fn']}")
