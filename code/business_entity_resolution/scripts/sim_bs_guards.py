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

val_s1_set = _load_val_s1_ids()
gt_full = _load_gt(val_s1_set)

# Sibling recovered
rec_n014 = np.load(EXP_DIR / "n014_recovered_matches.npz")
recovered_map = defaultdict(set)
for s, c in zip(rec_n014["s1_ids"].astype(str), rec_n014["cand_ids"].astype(str)):
    recovered_map[s].add(c)
df_rec = pd.read_csv("reports/blocker_recovery_matrix.csv")
for _, row in df_rec[df_rec["recovered_by"] != "none"].iterrows():
    recovered_map[str(row["s1_id"])].add(str(row["candidate_id"]))

# Load graph
df_gt = pd.read_csv(DATA_DIR / "train_ground_truth.tsv", sep="\t").dropna()
training_edges = defaultdict(set)
for _, row in df_gt.iterrows():
    matches = [m.strip() for m in str(row["matched_entity_ids"]).split(",") if m.strip()]
    for m1 in matches:
        for m2 in matches:
            if m1 != m2:
                training_edges[m1].add(m2)

# Load pairs
va = np.load(EXP_DIR / "val_pairs_n010.npz", allow_pickle=True)
s1_ids_str = va["s1_ids"].astype(str)
cand_ids_str = va["cand_ids"].astype(str)
sources_str = va["sources"].astype(str)
X = va["X"].astype(np.float32)
y = va["y"].astype(np.int8)

pl = np.load(EXP_DIR / "probs_n011_optimal_lgb.npy")
px = np.load(EXP_DIR / "probs_n011_optimal_xgb.npy")
p_n014 = (0.45 * pl + 0.55 * px).astype(np.float32)
p_xlmr = np.load(EXP_DIR / "EXP_N017" / "probs_val_cross_encoder.npy").astype(np.float32)
p_hyb = (0.73 * p_n014 + 0.27 * p_xlmr).astype(np.float32)

N_ent = 25000
K = 80
cand_mat = cand_ids_str.reshape(N_ent, K)
src_mat = sources_str.reshape(N_ent, K)
X_mat = X.reshape(N_ent, K, 57)
y_mat = y.reshape(N_ent, K)
p_mat = p_hyb.reshape(N_ent, K)
unique_s1 = s1_ids_str[::80]

sort_idx = np.argsort(-p_mat, axis=1)
row_idx = np.arange(N_ent)[:, None]
p_s = p_mat[row_idx, sort_idx]
X_s = X_mat[row_idx, sort_idx]
y_s = y_mat[row_idx, sort_idx]
c_s = cand_mat[row_idx, sort_idx]
src_s = src_mat[row_idx, sort_idx]

hn_s = X_s[:, :, 40]
bs_s = X_s[:, :, 51]
aj_s = X_s[:, :, 30]
addr_miss_s = X_s[:, :, 44]
name_jacc_s = X_s[:, :, 11]
postal_s = X_s[:, :, 41]
neural_cos_s = X_s[:, :, 56]

from run_n032 import extract_frontier_features
from sklearn.model_selection import KFold
import xgboost as xgb

am = (p_s >= 0.9570)
recovery_n020 = (p_s >= 0.9320) & ((hn_s == 1) | (aj_s >= 0.30)) & (bs_s == 0)
pred_base = am | recovery_n020

f_feats, f_labels, f_coords = extract_frontier_features(
    pred_base, p_s, X_s, y_s, src_s, aj_s, hn_s, bs_s, addr_miss_s,
    max_rank=10, include_representation_features=True
)
p1_cv = np.zeros(len(f_labels), dtype=np.float32)
kf1 = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kf1.split(f_feats):
    dtrain = xgb.DMatrix(f_feats[train_idx], label=f_labels[train_idx])
    dval = xgb.DMatrix(f_feats[val_idx])
    params = {"max_depth": 4, "eta": 0.05, "objective": "binary:logistic", "eval_metric": "auc", "verbosity": 0, "nthread": 1}
    bst1 = xgb.train(params, dtrain, num_boost_round=150)
    p1_cv[val_idx] = bst1.predict(dval)

pred_E = pred_base.copy()
admit_E = p1_cv >= 0.70
for idx, (e, r) in enumerate(f_coords):
    if admit_E[idx]:
        pred_E[e, r] = True

f2_feats, f2_labels, f2_coords = extract_frontier_features(
    pred_E, p_s, X_s, y_s, src_s, aj_s, hn_s, bs_s, addr_miss_s,
    max_rank=12, include_representation_features=True
)
p2_cv = np.zeros(len(f2_labels), dtype=np.float32)
kf2 = KFold(n_splits=5, shuffle=True, random_state=142)
for train_idx, val_idx in kf2.split(f2_feats):
    dtrain = xgb.DMatrix(f2_feats[train_idx], label=f2_labels[train_idx])
    dval = xgb.DMatrix(f2_feats[val_idx])
    params = {"max_depth": 4, "eta": 0.04, "objective": "binary:logistic", "eval_metric": "auc", "verbosity": 0, "nthread": 1}
    bst2 = xgb.train(params, dtrain, num_boost_round=120)
    p2_cv[val_idx] = bst2.predict(dval)

pred_F = pred_E.copy()
admit_F = p2_cv >= 0.75
for idx, (e, r) in enumerate(f2_coords):
    if admit_F[idx]:
        pred_F[e, r] = True

f3_feats, f3_labels, f3_coords = extract_frontier_features(
    pred_F, p_s, X_s, y_s, src_s, aj_s, hn_s, bs_s, addr_miss_s,
    max_rank=14, include_representation_features=True
)
p3_cv = np.zeros(len(f3_labels), dtype=np.float32)
kf3 = KFold(n_splits=5, shuffle=True, random_state=242)
for train_idx, val_idx in kf3.split(f3_feats):
    dtrain = xgb.DMatrix(f3_feats[train_idx], label=f3_labels[train_idx])
    dval = xgb.DMatrix(f3_feats[val_idx])
    params = {"max_depth": 4, "eta": 0.04, "objective": "binary:logistic", "eval_metric": "auc", "verbosity": 0, "nthread": 1}
    bst3 = xgb.train(params, dtrain, num_boost_round=100)
    p3_cv[val_idx] = bst3.predict(dval)

pred_A_base = pred_F.copy()
admit_G = p3_cv >= 0.70
for idx, (e, r) in enumerate(f3_coords):
    if admit_G[idx]:
        pred_A_base[e, r] = True

# N031 baseline
pred_N031 = pred_A_base.copy()
s1_acc_map = {unique_s1[i]: set(c_s[i][pred_A_base[i]]) for i in range(N_ent)}
for i in range(N_ent):
    s1 = unique_s1[i]
    acc_sibs = s1_acc_map.get(s1, set())
    for r in range(1, 10):
        if not pred_N031[i, r] and bs_s[i, r] == 0:
            cid = c_s[i, r]
            if (p_s[i, r] >= 0.30) and any(cid in training_edges.get(sib, set()) for sib in acc_sibs):
                pred_N031[i, r] = True

# N032-G baseline
pred_N032 = pred_N031.copy()
for i in range(N_ent):
    s1 = unique_s1[i]
    acc_sibs = s1_acc_map.get(s1, set())
    if not acc_sibs:
        continue
    for r in range(1, 20):
        if not pred_N032[i, r] and bs_s[i, r] == 0:
            cid = c_s[i, r]
            is_len2 = (p_s[i, r] >= 0.05) and any(cid in training_edges.get(sib, set()) for sib in acc_sibs)
            is_len3 = False
            if not is_len2 and p_s[i, r] >= 0.20:
                for sib in acc_sibs:
                    for nbr in training_edges.get(sib, set()):
                        if cid in training_edges.get(nbr, set()):
                            is_len3 = True
                            break
                    if is_len3:
                        break
            if is_len2 or is_len3:
                pred_N032[i, r] = True

# N033-G baseline
pred_N033 = pred_N032.copy()
for i in range(N_ent):
    s1 = unique_s1[i]
    acc_sibs = s1_acc_map.get(s1, set())
    if not acc_sibs:
        continue
    for r in range(1, 80):
        if not pred_N033[i, r] and bs_s[i, r] == 0:
            cid = c_s[i, r]
            conn_count = sum(1 for sib in acc_sibs if cid in training_edges.get(sib, set()))
            if conn_count >= 2 and neural_cos_s[i, r] >= 0.45:
                pred_N033[i, r] = True
            elif conn_count >= 1:
                cos = neural_cos_s[i, r]
                aj = aj_s[i, r]
                hn = hn_s[i, r]
                p = p_s[i, r]
                guard = (p >= 0.01) or (cos >= 0.685) or (cos >= 0.505 and aj >= 0.25) or (hn == 1 and cos >= 0.505)
                if guard:
                    pred_N033[i, r] = True

res_n033 = evaluate_authoritative(pred_N033, unique_s1, c_s, gt_full, val_s1_set, recovered_map)
print(f"N033 Champion: Metric B={res_n033['macro_f_05']:.6f} | P={res_n033['macro_precision']:.6f} | R={res_n033['macro_recall']:.6f} | FP={res_n033['fp']} | FN={res_n033['total_fn']}")

# Simulate branch conflict resolution:
# For candidates with bs > 0 that are connected to accepted siblings:
# Test various gating combinations: conn_count, p, neural_cosine, address_jaccard, house_num_match
tps_bs = []
fps_bs = []
for i in range(N_ent):
    s1 = unique_s1[i]
    acc_sibs = s1_acc_map.get(s1, set())
    if not acc_sibs:
        continue
    true_s1 = gt_full.get(s1, set())
    for r in range(1, 80):
        if not pred_N033[i, r] and bs_s[i, r] > 0:
            cid = c_s[i, r]
            conn_count = sum(1 for sib in acc_sibs if cid in training_edges.get(sib, set()))
            if conn_count > 0:
                rec = {
                    "s1_id": s1,
                    "cand_id": cid,
                    "r": int(r),
                    "p": float(p_s[i, r]),
                    "conn_count": int(conn_count),
                    "neural_cosine": float(neural_cos_s[i, r]),
                    "address_jaccard": float(aj_s[i, r]),
                    "house_num": float(hn_s[i, r]),
                    "postal": float(postal_s[i, r]),
                    "bs": float(bs_s[i, r]),
                }
                if cid in true_s1:
                    tps_bs.append(rec)
                else:
                    fps_bs.append(rec)

df_tps_bs = pd.DataFrame(tps_bs)
df_fps_bs = pd.DataFrame(fps_bs)
print(f"\nBranch Conflict Candidates with Graph Connections: Total TPs={len(df_tps_bs)}, Total FPs={len(df_fps_bs)}")

# Grid search precision guards for branch conflict cases
for min_conn in [1, 2, 3]:
    for min_p in [0.0, 0.50, 0.70, 0.85]:
        for min_aj in [0.0, 0.20, 0.35]:
            tps_pass = sum((df_tps_bs["conn_count"] >= min_conn) & (df_tps_bs["p"] >= min_p) & (df_tps_bs["address_jaccard"] >= min_aj))
            fps_pass = sum((df_fps_bs["conn_count"] >= min_conn) & (df_fps_bs["p"] >= min_p) & (df_fps_bs["address_jaccard"] >= min_aj))
            net = tps_pass - 4 * fps_pass
            if net > 0:
                print(f"conn>={min_conn}, p>={min_p:.2f}, aj>={min_aj:.2f} -> TPs: {tps_pass:3d}, FPs: {fps_pass:2d} | Net: {net:3d}")
