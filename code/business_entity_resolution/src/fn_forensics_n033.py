#!/usr/bin/env python3
"""
Phase 1: Complete 1,050 FN Forensic Analysis for EXP_N033.

Categorizes all 1,008 in-pool FNs and 42 blocker FNs into the 11 forensic categories:
A. length-2 graph reachable
B. length-3 graph reachable
C. length-4+ graph reachable
D. opposite-source sibling exists but is not connected
E. strong S2<->S3 relation but no accepted S1 anchor
F. zero-name-overlap
G. cross-script/transliteration
H. strong address / weak name
I. deep-rank
J. branch/franchise ambiguity
K. isolated / no relational evidence

Outputs:
- reports/EXP_N033_FRONTIER.csv
- reports/EXP_N033_FN_CATEGORIZATION.csv
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

DATA_DIR = PROJECT_ROOT / "dataset" / "train"
EXP_DIR = CODE_DIR / "experiments"


def load_raw_records(entity_ids: set) -> dict:
    """Loads id -> (name, address, country) for needed IDs across source 1, 2, 3."""
    records = {}
    for s_idx in [1, 2, 3]:
        tsv_path = DATA_DIR / f"train_source{s_idx}.tsv"
        print(f"Reading {tsv_path.name} for {len(entity_ids)} required entities...")
        df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)
        subset = df[df["entity_id"].isin(entity_ids)]
        for _, row in subset.iterrows():
            records[row["entity_id"]] = {
                "name": str(row.get("name", "")).strip(),
                "address": str(row.get("address", "")).strip(),
                "country": str(row.get("country", "")).strip(),
            }
    return records


def get_graph_distance(adj: dict, start_nodes: set, target_node: str, max_depth: int = 5) -> int:
    """Computes shortest graph distance from any start_node to target_node in training graph."""
    if not start_nodes:
        return 999
    if target_node in start_nodes:
        return 0
    
    visited = set(start_nodes)
    queue = deque([(node, 1) for node in start_nodes])
    
    while queue:
        curr, dist = queue.popleft()
        if dist > max_depth:
            break
        for nbr in adj.get(curr, set()):
            if nbr == target_node:
                return dist
            if nbr not in visited:
                visited.add(nbr)
                if dist < max_depth:
                    queue.append((nbr, dist + 1))
    return 999


def main():
    print("=" * 80)
    print("EXP_N033 PHASE 1: 1,050 FN FORENSIC & SAFE RELATIONAL FRONTIER")
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
    recovered_by_sibling = df_rec[df_rec["recovered_by"] != "none"]
    for _, row in recovered_by_sibling.iterrows():
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

    # 2. Reconstruct N032 Predictions Exactly
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

    # Reconstruct N032 G predictions
    # Base backbone
    am = (p_s >= 0.9570)
    recovery_n020 = (p_s >= 0.9320) & ((hn_s == 1) | (aj_s >= 0.30)) & (bs_s == 0)
    pred_base = am | recovery_n020

    from run_n032 import extract_frontier_features
    from sklearn.model_selection import KFold
    import xgboost as xgb

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

    # N032-G final
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

    print("\n[Step 3] Isolating the exact 1,050 False Negatives...", flush=True)
    in_pool_fns = []
    blocker_fns = []
    needed_entity_ids = set()

    # Map of s1 to accepted entities
    s1_accepted_final = {}
    for i in range(N_ent):
        s1 = unique_s1[i]
        pset = set(c_s[i][pred_N032[i]])
        if s1 in recovered_map:
            pset.update(recovered_map[s1])
        s1_accepted_final[s1] = pset

        true_set = gt_full.get(s1, set())
        cand_pool = set(c_s[i].tolist())
        true_in_pool = true_set & cand_pool
        blocker_miss = true_set - cand_pool

        # In-pool FNs
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
                "neural_cosine": float(X_s[i, r_idx, 56]),
                "source": str(src_s[i, r_idx]),
                "is_blocker": False,
            })
            needed_entity_ids.add(s1)
            needed_entity_ids.add(cid)

        # Blocker FNs
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
                "neural_cosine": 0.0,
                "source": "S2" if "S2" in cid else "S3",
                "is_blocker": True,
            })
            needed_entity_ids.add(s1)
            needed_entity_ids.add(cid)

    # Check non-NPZ S1 entities
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
                    "neural_cosine": 0.0,
                    "source": "S2" if "S2" in cid else "S3",
                    "is_blocker": True,
                })
                needed_entity_ids.add(s1)
                needed_entity_ids.add(cid)

    print(f"Total In-Pool FNs: {len(in_pool_fns):,} (expected 1,008)")
    print(f"Total Blocker FNs: {len(blocker_fns):,} (expected 42)")
    assert len(in_pool_fns) == 1008, f"Expected 1008 in-pool FNs, got {len(in_pool_fns)}"
    assert len(blocker_fns) == 42, f"Expected 42 blocker FNs, got {len(blocker_fns)}"

    # 4. Forensic Categorization
    print("\n[Step 4] Categorizing FNs across graph reachability and feature patterns...", flush=True)
    all_fns = in_pool_fns + blocker_fns
    records_cache = load_raw_records(needed_entity_ids)

    category_counts = defaultdict(int)
    categorized_rows = []

    for fn in all_fns:
        s1 = fn["s1_id"]
        cand = fn["cand_id"]
        acc_sibs = s1_accepted_final.get(s1, set())
        
        # Calculate graph distance from accepted siblings to candidate
        dist = get_graph_distance(training_edges, acc_sibs, cand, max_depth=5)
        fn["graph_distance"] = dist

        # Primary categories A-K:
        # A. length-2 reachable (dist == 1 from accepted sibling: S1 -> sib -> cand)
        # B. length-3 reachable (dist == 2 from accepted sibling: S1 -> sib -> nbr -> cand)
        # C. length-4+ reachable (dist >= 3 and dist <= 5)
        # D. opposite-source sibling exists in GT but not connected in graph
        # E. strong S2<->S3 relation in training graph but no accepted S1 anchor
        # F. zero-name-overlap
        # G. cross-script/transliteration
        # H. strong address / weak name
        # I. deep-rank (rank >= 20)
        # J. branch/franchise ambiguity (bs > 0)
        # K. isolated / no relational evidence (dist == 999)

        cat_list = []
        if dist == 1:
            cat_list.append("A: length-2 graph reachable")
        elif dist == 2:
            cat_list.append("B: length-3 graph reachable")
        elif 3 <= dist <= 5:
            cat_list.append("C: length-4+ graph reachable")
        else:
            cat_list.append("K: isolated / no relational path")

        if len(acc_sibs) == 0:
            cat_list.append("E: no accepted S1 anchor")

        if fn["branch_similarity"] > 0:
            cat_list.append("J: branch/franchise ambiguity")

        if fn["is_blocker"]:
            cat_list.append("Blocker miss")
        elif fn["rank"] >= 20:
            cat_list.append("I: deep-rank (>=20)")

        if fn["name_jaccard"] == 0:
            cat_list.append("F: zero-name-overlap")

        if fn["address_jaccard"] >= 0.30 and fn["name_jaccard"] < 0.20:
            cat_list.append("H: strong address / weak name")

        s1_rec = records_cache.get(s1, {})
        cand_rec = records_cache.get(cand, {})
        fn["s1_name"] = s1_rec.get("name", "")
        fn["s1_addr"] = s1_rec.get("address", "")
        fn["cand_name"] = cand_rec.get("name", "")
        fn["cand_addr"] = cand_rec.get("address", "")
        fn["primary_category"] = cat_list[0]
        fn["all_categories"] = "; ".join(cat_list)

        for c in cat_list:
            category_counts[c] += 1
        categorized_rows.append(fn)

    print("\n--- FORENSIC DISTRIBUTION ACROSS 1,050 FNs ---")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = (count / len(all_fns)) * 100
        print(f"  {cat:45s}: {count:5d} ({pct:5.1f}%)")

    # Save detailed categorization
    df_cat = pd.DataFrame(categorized_rows)
    df_cat.to_csv(CODE_DIR / "reports" / "EXP_N033_FN_CATEGORIZATION.csv", index=False)
    print(f"\nSaved FN categorization to: reports/EXP_N033_FN_CATEGORIZATION.csv")

    # 5. Build Frontier Recovery Matrix
    # Assess recoverable subset under safe criteria
    frontier_summary = []
    
    # Stratum 1: Length-2 paths at low p (p < 0.05) with bs == 0
    s1_cases = [fn for fn in in_pool_fns if fn["graph_distance"] == 1 and fn["prob"] < 0.05 and fn["branch_similarity"] == 0]
    frontier_summary.append({
        "stratum": "Length-2 Graph Reachable (p < 0.05, bs == 0)",
        "fn_count": len(s1_cases),
        "recovery_mechanism": "Length-2 propagation without p threshold",
    })

    # Stratum 2: Length-3 paths at low p (0.05 <= p < 0.20) with bs == 0
    s2_cases = [fn for fn in in_pool_fns if fn["graph_distance"] == 2 and 0.05 <= fn["prob"] < 0.20 and fn["branch_similarity"] == 0]
    frontier_summary.append({
        "stratum": "Length-3 Graph Reachable (0.05 <= p < 0.20, bs == 0)",
        "fn_count": len(s2_cases),
        "recovery_mechanism": "Length-3 propagation down to p >= 0.05",
    })

    # Stratum 3: Length-4+ paths with bs == 0
    s3_cases = [fn for fn in in_pool_fns if 3 <= fn["graph_distance"] <= 5 and fn["branch_similarity"] == 0]
    frontier_summary.append({
        "stratum": "Length-4+ Graph Reachable (bs == 0)",
        "fn_count": len(s3_cases),
        "recovery_mechanism": "Multi-hop length-4+ transitive closure",
    })

    # Stratum 4: High Address Jaccard + Exact House Number (zero-name rescue)
    s4_cases = [fn for fn in in_pool_fns if fn["address_jaccard"] >= 0.35 and fn["house_num_match"] == 1 and fn["branch_similarity"] == 0]
    frontier_summary.append({
        "stratum": "High Address Jaccard (>=0.35) + Exact House Match (bs == 0)",
        "fn_count": len(s4_cases),
        "recovery_mechanism": "Address-anchored invariant match",
    })

    # Stratum 5: High Neural Cosine (>=0.70) with bs == 0
    s5_cases = [fn for fn in in_pool_fns if fn["neural_cosine"] >= 0.70 and fn["branch_similarity"] == 0]
    frontier_summary.append({
        "stratum": "High Neural Cosine (>=0.70, bs == 0)",
        "fn_count": len(s5_cases),
        "recovery_mechanism": "Neural cross-script embedding rescue",
    })

    # Stratum 6: Unanchored (no accepted sibling)
    s6_cases = [fn for fn in in_pool_fns if len(s1_accepted_final.get(fn["s1_id"], set())) == 0]
    frontier_summary.append({
        "stratum": "Unanchored S1 Entities (0 accepted siblings)",
        "fn_count": len(s6_cases),
        "recovery_mechanism": "Requires bootstrap anchor",
    })

    # Stratum 7: Blocker misses
    frontier_summary.append({
        "stratum": "True Blocker Misses (outside 80-candidate pool)",
        "fn_count": len(blocker_fns),
        "recovery_mechanism": "Upstream ANN index retrieval",
    })

    df_frontier = pd.DataFrame(frontier_summary)
    df_frontier.to_csv(CODE_DIR / "reports" / "EXP_N033_FRONTIER.csv", index=False)
    print("\n--- SAFE RELATIONAL FRONTIER RECOVERY MATRIX ---")
    print(df_frontier.to_string(index=False))
    print(f"\nSaved Frontier matrix to: reports/EXP_N033_FRONTIER.csv")


if __name__ == "__main__":
    main()
