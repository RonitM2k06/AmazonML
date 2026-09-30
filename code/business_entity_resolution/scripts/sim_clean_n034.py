import sys
from pathlib import Path
sys.path.insert(0, str(Path("scripts")))
from sim_n034_candidate import (
    unique_s1, c_s, bs_s, aj_s, neural_cos_s, gt_full,
    pred_N033, training_edges, s1_acc_map, evaluate_authoritative,
    val_s1_set, recovered_map
)

pred_clean = pred_N033.copy()
added_tp = 0
added_fp = 0
for i in range(len(unique_s1)):
    s1 = unique_s1[i]
    acc_sibs = s1_acc_map.get(s1, set())
    if not acc_sibs:
        continue
    true_s1 = gt_full.get(s1, set())
    for r in range(1, 80):
        if not pred_clean[i, r]:
            cid = c_s[i, r]
            conn_count = sum(1 for sib in acc_sibs if cid in training_edges.get(sib, set()))
            rule_branch_conn = (conn_count >= 1) and (bs_s[i, r] > 0)
            rule_multi_anchor_3 = (conn_count >= 3) and (bs_s[i, r] == 0)
            if rule_branch_conn or rule_multi_anchor_3:
                pred_clean[i, r] = True
                if cid in true_s1:
                    added_tp += 1
                else:
                    added_fp += 1

res = evaluate_authoritative(pred_clean, unique_s1, c_s, gt_full, val_s1_set, recovered_map)
print(f"CLEAN N034 (rule_b + rule_m3):")
print(f"Added TP: {added_tp}, Added FP: {added_fp}")
print(f"Metric B: {res['macro_f_05']:.6f} | P: {res['macro_precision']:.6f} | R: {res['macro_recall']:.6f} | FP={res['fp']} | FN={res['total_fn']}")
