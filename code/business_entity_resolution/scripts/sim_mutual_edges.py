import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
import sys

CODE_DIR = Path(".")
sys.path.insert(0, str(CODE_DIR / "src"))
sys.path.insert(0, str(CODE_DIR / "scripts"))
from authoritative_eval_fn import _load_val_s1_ids, _load_gt
from run_n032 import evaluate_authoritative

DATA_DIR = Path("../../dataset/train")
EXP_DIR = CODE_DIR / "experiments"

from sim_n034_candidate import (
    unique_s1, c_s, bs_s, aj_s, neural_cos_s, p_s, hn_s,
    s1_acc_map, training_edges, gt_full, pred_N033, val_s1_set, recovered_map
)

# Build N034 predictions
pred_N034 = pred_N033.copy()
for i in range(len(unique_s1)):
    s1 = unique_s1[i]
    acc_sibs = s1_acc_map.get(s1, set())
    if not acc_sibs:
        continue
    for r in range(1, 80):
        if not pred_N034[i, r]:
            cid = c_s[i, r]
            conn_count = sum(1 for sib in acc_sibs if cid in training_edges.get(sib, set()))
            rule_b = (conn_count >= 1) and (bs_s[i, r] > 0)
            rule_m3 = (conn_count >= 3) and (bs_s[i, r] == 0)
            if rule_b or rule_m3:
                pred_N034[i, r] = True

res_n034 = evaluate_authoritative(pred_N034, unique_s1, c_s, gt_full, val_s1_set, recovered_map)
print(f"N034 Verified: Metric B={res_n034['macro_f_05']:.6f} | FP={res_n034['fp']} | FN={res_n034['total_fn']}")

# Search unanchored S1 entities: candidates connected to EACH OTHER in the training graph!
triangles_tp = []
triangles_fp = []

for i in range(len(unique_s1)):
    s1 = unique_s1[i]
    # Check if this S1 has 0 accepted predictions currently
    if np.sum(pred_N034[i]) > 0:
        continue
    true_s1 = gt_full.get(s1, set())
    cands_i = c_s[i]
    
    # Check top-20 candidates for mutual training edges
    for r1 in range(0, 20):
        c1 = cands_i[r1]
        for r2 in range(r1 + 1, 20):
            c2 = cands_i[r2]
            if c2 in training_edges.get(c1, set()):
                # c1 and c2 are mutually connected in the training graph!
                rec1 = {"s1": s1, "cid": c1, "r": r1, "p": float(p_s[i, r1]), "cos": float(neural_cos_s[i, r1]), "aj": float(aj_s[i, r1]), "hn": float(hn_s[i, r1]), "partner": c2}
                rec2 = {"s1": s1, "cid": c2, "r": r2, "p": float(p_s[i, r2]), "cos": float(neural_cos_s[i, r2]), "aj": float(aj_s[i, r2]), "hn": float(hn_s[i, r2]), "partner": c1}
                for rec, is_tp in [(rec1, c1 in true_s1), (rec2, c2 in true_s1)]:
                    if is_tp:
                        triangles_tp.append(rec)
                    else:
                        triangles_fp.append(rec)

df_ttp = pd.DataFrame(triangles_tp).drop_duplicates(subset=["s1", "cid"])
df_tfp = pd.DataFrame(triangles_fp).drop_duplicates(subset=["s1", "cid"])
print(f"\nMutual Candidate Edges for Unanchored Entities: TPs={len(df_ttp)}, FPs={len(df_tfp)}")
if len(df_tfp) > 0:
    print("FP sample:")
    print(df_tfp.head(10))
if len(df_ttp) > 0:
    print("TP sample:")
    print(df_ttp.head(10))
