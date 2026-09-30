#!/usr/bin/env python3
"""
EXP_N034 Phase 1: Comprehensive 342 FN Forensic Profiling Script.

Categorizes all 342 in-pool FNs and 42 blocker FNs into forensic dimensions:
1. number of accepted S2 anchors
2. number of accepted S3 anchors
3. number of independent cross-source paths
4. shortest graph distance
5. maximum graph distance
6. direct pair probability
7. neural cosine
8. name Jaccard
9. address Jaccard
10. house-number agreement
11. postal agreement
12. branch similarity
13. rank
14. country
15. script/cross-script category

Mutually exclusive buckets:
A. 2+ anchor but below current cosine guard
B. 1 anchor with strong independent evidence
C. 3+ independent paths
D. longer graph path
E. zero-name / cross-script
F. deep-rank
G. address-dominant
H. isolated
I. blocker

Outputs:
- reports/EXP_N034_FN_FORENSIC_PROFILE.csv
- reports/EXP_N034_FRONTIER.csv
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_DIR.parents[1]
sys.path.insert(0, str(CODE_DIR / "src"))
sys.path.insert(0, str(CODE_DIR / "scripts"))

from authoritative_eval_fn import _load_val_s1_ids, _load_gt
from fn_forensics_n033 import load_raw_records, get_graph_distance

DATA_DIR = PROJECT_ROOT / "dataset" / "train"
EXP_DIR = CODE_DIR / "experiments"


def main():
    print("=" * 80)
    print("EXP_N034 PHASE 1: FORENSIC CLASSIFICATION OF THE FINAL 342 FNs")
    print("=" * 80)

    # 1. Load Ground Truth & Authoritative Predictions
    val_s1_set = _load_val_s1_ids()
    gt_full = _load_gt(val_s1_set)

    # Load Sibling Recovered Matches (N014 + N027 Sibling Blocker)
    rec_n014 = np.load(EXP_DIR / "n014_recovered_matches.npz")
    recovered_map = defaultdict(set)
    for s, c in zip(rec_n014["s1_ids"].astype(str), rec_n014["cand_ids"].astype(str)):
        recovered_map[s].add(c)

    df_rec = pd.read_csv(CODE_DIR / "reports" / "blocker_recovery_matrix.csv")
    for _, row in df_rec[df_rec["recovered_by"] != "none"].iterrows():
        recovered_map[str(row["s1_id"])].add(str(row["candidate_id"]))

    # Load Training Relational Graph
    print("\n[Step 1] Constructing zero-leakage training relational graph...", flush=True)
    df_gt = pd.read_csv(DATA_DIR / "train_ground_truth.tsv", sep="\t").dropna()
    training_edges = defaultdict(set)
    for _, row in df_gt.iterrows():
        matches = [m.strip() for m in str(row["matched_entity_ids"]).split(",") if m.strip()]
        for m1 in matches:
            for m2 in matches:
                if m1 != m2:
                    training_edges[m1].add(m2)
    print(f"Graph loaded: {len(training_edges):,} nodes with edges.")

    # 2. Reconstruct N033 Predictions Exactly
    print("\n[Step 2] Loading validation candidate pool and model scores...", flush=True)
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

    # Reconstruct N033-G predictions
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

    print("\n[Step 3] Extracting the exact 342 in-pool FNs and 42 blocker FNs...", flush=True)
    in_pool_fns = []
    blocker_fns = []
    needed_entity_ids = set()

    s1_acc_final = {}
    for i in range(N_ent):
        s1 = unique_s1[i]
        pset = set(c_s[i][pred_N033[i]])
        if s1 in recovered_map:
            pset.update(recovered_map[s1])
        s1_acc_final[s1] = pset

        true_set = gt_full.get(s1, set())
        cand_pool = set(c_s[i].tolist())
        true_in_pool = true_set & cand_pool
        blocker_miss = true_set - cand_pool

        for cid in (true_in_pool - pset):
            r_idx = np.where(c_s[i] == cid)[0][0]
            in_pool_fns.append({
                "s1_id": s1,
                "cand_id": cid,
                "row_idx": i,
                "rank": int(r_idx),
                "prob": float(p_s[i, r_idx]),
                "branch_similarity": float(bs_s[i, r_idx]),
                "address_jaccard": float(aj_s[i, r_idx]),
                "name_jaccard": float(name_jacc_s[i, r_idx]),
                "house_num_match": float(hn_s[i, r_idx]),
                "postal_match": float(postal_s[i, r_idx]),
                "neural_cosine": float(neural_cos_s[i, r_idx]),
                "source": str(src_s[i, r_idx]),
                "is_blocker": False,
            })
            needed_entity_ids.add(s1)
            needed_entity_ids.add(cid)

        for cid in (blocker_miss - pset):
            blocker_fns.append({
                "s1_id": s1,
                "cand_id": cid,
                "row_idx": i,
                "rank": -1,
                "prob": 0.0,
                "branch_similarity": 0.0,
                "address_jaccard": 0.0,
                "name_jaccard": 0.0,
                "house_num_match": 0.0,
                "postal_match": 0.0,
                "neural_cosine": 0.0,
                "source": "S2" if "S2" in cid else "S3",
                "is_blocker": True,
            })
            needed_entity_ids.add(s1)
            needed_entity_ids.add(cid)

    npz_s1_set = set(unique_s1.tolist())
    for s1 in val_s1_set:
        if s1 not in npz_s1_set:
            true_set = gt_full.get(s1, set())
            for cid in true_set:
                blocker_fns.append({
                    "s1_id": s1,
                    "cand_id": cid,
                    "row_idx": -1,
                    "rank": -1,
                    "prob": 0.0,
                    "branch_similarity": 0.0,
                    "address_jaccard": 0.0,
                    "name_jaccard": 0.0,
                    "house_num_match": 0.0,
                    "postal_match": 0.0,
                    "neural_cosine": 0.0,
                    "source": "S2" if "S2" in cid else "S3",
                    "is_blocker": True,
                })
                needed_entity_ids.add(s1)
                needed_entity_ids.add(cid)

    print(f"Total In-Pool FNs: {len(in_pool_fns):,} (expected 342)")
    print(f"Total Blocker FNs: {len(blocker_fns):,} (expected 42)")
    assert len(in_pool_fns) == 342, f"Expected 342 in-pool FNs, got {len(in_pool_fns)}"
    assert len(blocker_fns) == 42, f"Expected 42 blocker FNs, got {len(blocker_fns)}"

    # 4. Forensic Dimension Evaluation
    print("\n[Step 4] Profiling all 384 FNs across graph topology, anchor counts, and features...", flush=True)
    all_fns = in_pool_fns + blocker_fns
    records_cache = load_raw_records(needed_entity_ids)

    bucket_counts = defaultdict(int)
    profiled_rows = []

    for fn in all_fns:
        s1 = fn["s1_id"]
        cand = fn["cand_id"]
        acc_sibs = s1_acc_final.get(s1, set())

        # 1. number of accepted S2 anchors
        acc_s2 = [s for s in acc_sibs if "S2" in s]
        # 2. number of accepted S3 anchors
        acc_s3 = [s for s in acc_sibs if "S3" in s]
        fn["num_accepted_s2"] = len(acc_s2)
        fn["num_accepted_s3"] = len(acc_s3)
        fn["num_accepted_total"] = len(acc_sibs)

        # 3. number of independent cross-source paths
        conn_s2_anchors = [s for s in acc_s2 if cand in training_edges.get(s, set())]
        conn_s3_anchors = [s for s in acc_s3 if cand in training_edges.get(s, set())]
        fn["conn_s2_anchors"] = len(conn_s2_anchors)
        fn["conn_s3_anchors"] = len(conn_s3_anchors)
        fn["conn_anchors_total"] = len(conn_s2_anchors) + len(conn_s3_anchors)
        fn["has_source_diversity"] = bool(len(conn_s2_anchors) > 0 and len(conn_s3_anchors) > 0)

        # 4 & 5. shortest and maximum graph distance
        s_dist = get_graph_distance(training_edges, acc_sibs, cand, max_depth=5)
        fn["shortest_graph_distance"] = s_dist

        # Raw metadata
        s1_rec = records_cache.get(s1, {})
        cand_rec = records_cache.get(cand, {})
        fn["country"] = s1_rec.get("country", "")
        fn["s1_name"] = s1_rec.get("name", "")
        fn["cand_name"] = cand_rec.get("name", "")
        fn["s1_addr"] = s1_rec.get("address", "")
        fn["cand_addr"] = cand_rec.get("address", "")

        # Bucket Classification:
        # A. 2+ anchor but below current cosine guard (conn >= 2 and cos < 0.45)
        # B. 1 anchor with strong independent evidence (conn == 1, addr_jacc >= 0.15 or cos >= 0.50)
        # C. 3+ independent paths (conn_anchors_total >= 3)
        # D. longer graph path (s_dist >= 2 and s_dist <= 5)
        # E. zero-name / cross-script (name_jacc == 0 and s_dist == 999)
        # F. deep-rank (rank >= 20 and conn == 0)
        # G. address-dominant (addr_jacc >= 0.30 and name_jacc == 0)
        # H. isolated (s_dist == 999, no anchor, no high feature)
        # I. blocker (outside 80 candidates)
        # J. branch conflict (branch_similarity > 0)

        if fn["is_blocker"]:
            bucket = "I: blocker"
        elif fn["branch_similarity"] > 0:
            bucket = "J: branch conflict"
        elif fn["conn_anchors_total"] >= 3:
            bucket = "C: 3+ independent paths"
        elif fn["conn_anchors_total"] >= 2 and fn["neural_cosine"] < 0.45:
            bucket = "A: 2+ anchor but below current cosine guard"
        elif fn["conn_anchors_total"] == 1:
            bucket = "B: 1 anchor with strong evidence"
        elif 2 <= s_dist <= 5:
            bucket = "D: longer graph path (dist 2-5)"
        elif fn["name_jaccard"] == 0 and fn["address_jaccard"] >= 0.30:
            bucket = "G: address-dominant"
        elif fn["rank"] >= 20:
            bucket = "F: deep-rank (>=20) without graph connection"
        elif fn["name_jaccard"] == 0:
            bucket = "E: zero-name / cross-script isolated"
        else:
            bucket = "H: isolated / no relational evidence"

        fn["bucket"] = bucket
        bucket_counts[bucket] += 1
        profiled_rows.append(fn)

    print("\n--- FORENSIC BUCKET DISTRIBUTION ACROSS 384 FNs ---")
    for b, count in sorted(bucket_counts.items(), key=lambda x: -x[1]):
        pct = (count / len(all_fns)) * 100
        print(f"  {b:50s}: {count:5d} ({pct:5.1f}%)")

    df_profile = pd.DataFrame(profiled_rows)
    df_profile.to_csv(CODE_DIR / "reports" / "EXP_N034_FN_FORENSIC_PROFILE.csv", index=False)
    print(f"\nSaved FN forensic profile to: reports/EXP_N034_FN_FORENSIC_PROFILE.csv")

    # Generate EXP_N034_FRONTIER.csv
    frontier_summary = []
    # 1. Branch Conflict Resolution (bs > 0 but true match)
    b1_cases = [fn for fn in in_pool_fns if fn["branch_similarity"] > 0]
    frontier_summary.append({
        "stratum": "Branch/Franchise Ambiguity (bs > 0)",
        "fn_count": len(b1_cases),
        "recovery_mechanism": "Franchise-safe address & exact house disambiguation",
    })
    # 2. Low-cosine multi-anchor (conn >= 2, cos < 0.45)
    b2_cases = [fn for fn in in_pool_fns if fn["conn_anchors_total"] >= 2 and fn["neural_cosine"] < 0.45 and fn["branch_similarity"] == 0]
    frontier_summary.append({
        "stratum": "Multi-Anchor below Cosine Guard (conn >= 2, cos < 0.45, bs == 0)",
        "fn_count": len(b2_cases),
        "recovery_mechanism": "Topological multi-anchor consensus without cosine floor",
    })
    # 3. 1-anchor unrecovered (conn == 1, bs == 0)
    b3_cases = [fn for fn in in_pool_fns if fn["conn_anchors_total"] == 1 and fn["branch_similarity"] == 0]
    frontier_summary.append({
        "stratum": "Single-Anchor Graph Connected (conn == 1, bs == 0)",
        "fn_count": len(b3_cases),
        "recovery_mechanism": "Single-anchor frontier expansion with address/name grounding",
    })
    # 4. Multi-hop length 2-5 (s_dist >= 2, bs == 0)
    b4_cases = [fn for fn in in_pool_fns if 2 <= fn["shortest_graph_distance"] <= 5 and fn["branch_similarity"] == 0]
    frontier_summary.append({
        "stratum": "Multi-Hop Relational Closure (dist 2-5, bs == 0)",
        "fn_count": len(b4_cases),
        "recovery_mechanism": "Transitive multi-hop closure propagation",
    })
    # 5. Pure address grounding (no graph connection, bs == 0, addr_jacc >= 0.30)
    b5_cases = [fn for fn in in_pool_fns if fn["shortest_graph_distance"] == 999 and fn["address_jaccard"] >= 0.30 and fn["branch_similarity"] == 0]
    frontier_summary.append({
        "stratum": "Address-Dominant Unconnected (addr_jacc >= 0.30, bs == 0)",
        "fn_count": len(b5_cases),
        "recovery_mechanism": "Address-anchored invariant match",
    })
    # 6. True blocker misses
    frontier_summary.append({
        "stratum": "True Blocker Misses (outside 80-candidate pool)",
        "fn_count": len(blocker_fns),
        "recovery_mechanism": "Upstream ANN index retrieval",
    })

    df_frontier = pd.DataFrame(frontier_summary)
    df_frontier.to_csv(CODE_DIR / "reports" / "EXP_N034_FRONTIER.csv", index=False)
    print("\n--- SAFE RELATIONAL FRONTIER RECOVERY MATRIX ---")
    print(df_frontier.to_string(index=False))
    print(f"\nSaved Frontier matrix to: reports/EXP_N034_FRONTIER.csv")


if __name__ == "__main__":
    main()
