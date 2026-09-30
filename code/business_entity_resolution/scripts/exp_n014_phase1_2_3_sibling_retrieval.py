#!/usr/bin/env python3
"""
EXP_N014 — Phase 1, 2, 3: Bidirectional Sibling Retrieval on the 804 Missing True Pairs.

Evaluates:
- Querying S3 target index from accepted S2 sibling (S2 alone & S1+S2 composite)
- Querying S2 target index from accepted S3 sibling (S3 alone & S1+S3 composite)
- Sweep of top_k in [120, 200, 300, 500]
- Sibling lexical & address retrieval
- Exact recovery matrix for the 804 blocker misses
- Verification of new candidate ceiling on the complete 86,232 true pairs
"""

from __future__ import annotations

import os
import sys
import time
import re
import csv
import gc
from collections import defaultdict
from pathlib import Path

import numpy as np
import faiss
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONUNBUFFERED"] = "1"

# Paths
CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_DIR.parents[1]
sys.path.insert(0, str(CODE_DIR / "src"))
sys.path.insert(0, str(CODE_DIR / "scripts"))

from authoritative_eval_fn import _load_val_s1_ids, _load_gt
from two_tower_encoder import load_two_tower, RecordTokenizer, encode_records_fused

DATA_DIR = PROJECT_ROOT / "dataset" / "train"
EXP_DIR = CODE_DIR / "experiments"
MODEL_DIR = CODE_DIR / "models"


def main():
    t_start = time.time()
    print("=" * 75, flush=True)
    print("[EXP_N014] Phase 1, 2, 3: Targeted Bidirectional Sibling Retrieval", flush=True)
    print("=" * 75, flush=True)

    # 1. Load Ground Truth for 25k Validation S1
    val_s1_set = _load_val_s1_ids()
    gt_full = _load_gt(val_s1_set)
    total_true_matches = sum(len(mids) for mids in gt_full.values())
    print(f"Validation Population: {len(val_s1_set):,} entities. Total Ground Truth Matches: {total_true_matches:,}", flush=True)

    # 2. Identify Candidates in N010/N011 pool and Graph-Recovered matches from N013
    va = np.load(EXP_DIR / "val_pairs_n010.npz", allow_pickle=True)
    s1_10 = va["s1_ids"].astype(str)
    c_10 = va["cand_ids"].astype(str)
    existing_pool = set(zip(s1_10, c_10))

    # Load N011 probabilities to get accepted anchors
    pl = np.load(EXP_DIR / "probs_n011_optimal_lgb.npy")
    px = np.load(EXP_DIR / "probs_n011_optimal_xgb.npy")
    probs = 0.45 * pl + 0.55 * px

    unique_s1, s1_codes = np.unique(s1_10, return_inverse=True)
    sort_key = np.lexsort((-probs, s1_codes))
    s1_codes_s = s1_codes[sort_key]
    probs_s = probs[sort_key]
    c_ids_s = c_10[sort_key]
    sources_s = va["sources"].astype(str)[sort_key]

    boundaries = np.where(np.diff(s1_codes_s) != 0)[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(probs_s)]])

    base_th, dense_th, margin, anchor, bridge, density_cutoff = 0.940, 0.950, 0.025, 0.960, 0.920, 75
    sg = base_th * 0.85

    accepted_s2 = defaultdict(list)
    accepted_s3 = defaultdict(list)
    base_accepted_pool = set()

    for k in range(len(unique_s1)):
        sl = slice(starts[k], ends[k])
        ps = probs_s[sl]
        cids = c_ids_s[sl]
        s3 = sources_s[sl] == "S3"
        N_ = len(ps)
        if N_ == 0 or ps[0] < sg:
            continue
        top = ps[0]
        et = dense_th if N_ > density_cutoff else base_th
        am = (ps >= et) & ((top - ps) <= margin)
        anm = ps >= anchor
        if np.any(anm & am):
            if not bool(np.any(am & s3)):
                am = am | ((~am) & s3 & (ps >= bridge))
            if not bool(np.any(am & ~s3)):
                am = am | ((~am) & ~s3 & (ps >= bridge))
        
        s1_id = unique_s1[k]
        for cid, accepted in zip(cids[am], am[am]):
            base_accepted_pool.add((s1_id, cid))
            if cid.startswith("S2"):
                accepted_s2[s1_id].append(cid)
            else:
                accepted_s3[s1_id].append(cid)

    # 3. Stream Raw Records to get Graph-Recovered from N013 (+202 matches)
    # Collect all 1,006 blocker misses
    all_blocker_misses = [(s1, m) for s1, mids in gt_full.items() for m in mids if (s1, m) not in existing_pool]
    print(f"Total Blocker Misses before Graph Closure: {len(all_blocker_misses)}", flush=True)

    needed_ids = set()
    for s1, m in all_blocker_misses:
        needed_ids.add(s1)
        needed_ids.add(m)
        for p in accepted_s2.get(s1, []):
            needed_ids.add(p)
        for p in accepted_s3.get(s1, []):
            needed_ids.add(p)

    recs = {}
    with open(DATA_DIR / "train_source1.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in needed_ids:
                recs[p[0]] = (p[0], p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "", p[3] if len(p) > 3 else "")

    for sf in ["train_source2.tsv", "train_source3.tsv"]:
        with open(DATA_DIR / sf, encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if p[0] in needed_ids:
                    recs[p[0]] = (p[0], p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "", p[3] if len(p) > 3 else "")

    def clean_txt(s: str) -> str:
        return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split()) if s else ""

    def extract_hn(addr: str) -> str:
        m = re.match(r"^(\d+[\w\-]*)", addr.strip())
        return m.group(1).lower() if m else ""

    def tokens(s: str) -> set:
        return set(clean_txt(s).split())

    def jac(s1: set, s2: set) -> float:
        return len(s1 & s2) / max(1, len(s1 | s2))

    # N013 Graph Closure matches
    n013_graph_matches = set()
    for s1, m in all_blocker_misses:
        r_miss = recs.get(m, ("", "", "", ""))
        r_s1 = recs.get(s1, ("", "", "", ""))
        ad_m = clean_txt(r_miss[2])
        hn_m = extract_hn(r_miss[2])
        nm_m = clean_txt(r_miss[1])

        if ad_m and ad_m == clean_txt(r_s1[2]):
            n013_graph_matches.add((s1, m))
            continue

        sibs = accepted_s2.get(s1, []) + accepted_s3.get(s1, [])
        for sib in sibs:
            r_sib = recs.get(sib, ("", "", "", ""))
            ad_sib = clean_txt(r_sib[2])
            hn_sib = extract_hn(r_sib[2])
            t_ad_m = tokens(r_miss[2])
            t_ad_sib = tokens(r_sib[2])
            t_nm_m = tokens(r_miss[1])
            t_nm_sib = tokens(r_sib[1])

            if ad_m and ad_m == ad_sib:
                n013_graph_matches.add((s1, m))
                break
            if hn_m and hn_m == hn_sib and jac(t_ad_m, t_ad_sib) >= 0.40 and jac(t_nm_m, t_nm_sib) >= 0.25:
                n013_graph_matches.add((s1, m))
                break

    print(f"N013 Graph-Recovered Matches: {len(n013_graph_matches)}", flush=True)

    # 4. Extract the Exact Remaining 804 Blocker Misses
    remaining_804 = [(s1, m) for s1, m in all_blocker_misses if (s1, m) not in n013_graph_matches]
    print(f"Target Set: Exactly {len(remaining_804)} remaining blocker misses to recover.\n", flush=True)

    # 5. Encode Needed Records with Two-Tower Model on GPU
    print("Encoding S1, accepted siblings, and target records on GPU...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    two_tower = load_two_tower(str(MODEL_DIR / "two_tower_stageC.pt"), device=device)
    tok = RecordTokenizer()

    ids_list, vecs = encode_records_fused(two_tower, tok, recs, batch_size=256, device=device)
    id2vec = {id_: vecs[i] for i, id_ in enumerate(ids_list)}

    # 6. Stream Country and ID Index for Target Sets
    print("Indexing target corpus by country and source...", flush=True)
    # Target embeddings file
    tgt_emb = np.load(EXP_DIR / "embeddings_n010_target.npz", allow_pickle=True)
    tgt_ids = tgt_emb["ids"]
    tgt_vecs = tgt_emb["vecs"]
    N_tgt = len(tgt_ids)

    # Target countries
    tgt_country = {}
    for sf in ["train_source2.tsv", "train_source3.tsv"]:
        with open(DATA_DIR / sf, encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if len(p) > 3:
                    tgt_country[p[0]] = p[3]

    # Split targets into: (S2_India, S2_US, S3_India, S3_US)
    # Boundary: S2 is 0..5034615, S3 is 5034616..10320218
    s2_india_idx = []
    s2_us_idx = []
    s3_india_idx = []
    s3_us_idx = []

    for i in range(N_tgt):
        tid = tgt_ids[i]
        c = tgt_country.get(tid, "")
        if i < 5034616:
            if c == "India": s2_india_idx.append(i)
            elif c == "US": s2_us_idx.append(i)
        else:
            if c == "India": s3_india_idx.append(i)
            elif c == "US": s3_us_idx.append(i)

    print(f"Target Sub-Index Sizes: S2_India={len(s2_india_idx):,}, S2_US={len(s2_us_idx):,}, S3_India={len(s3_india_idx):,}, S3_US={len(s3_us_idx):,}", flush=True)

    # 7. Sibling Query Execution
    # For each missed target in remaining_804:
    # If missed target is S3: query S3 sub-index using accepted S2 (alone, normalized, and composite S1+S2)
    # If missed target is S2: query S2 sub-index using accepted S3 (alone, normalized, and composite S1+S3)
    target_set_by_type = defaultdict(list)
    for s1, m in remaining_804:
        r_m = recs.get(m, ("", "", "", ""))
        ctry = r_m[3] if len(r_m) > 3 and r_m[3] else "India"
        is_s3 = m.startswith("S3")
        target_set_by_type[(is_s3, ctry)].append((s1, m))

    top_k_thresholds = [120, 200, 300, 500]
    recovered_by_topk_s2_to_s3 = defaultdict(set)
    recovered_by_topk_s3_to_s2 = defaultdict(set)
    recovered_by_topk_composite = defaultdict(set)

    # Process each sub-index one by one to keep RAM low
    sub_configs = [
        (True, "India", s3_india_idx, "S3_India"),
        (True, "US", s3_us_idx, "S3_US"),
        (False, "India", s2_india_idx, "S2_India"),
        (False, "US", s2_us_idx, "S2_US"),
    ]

    for is_s3_target, ctry, idx_list, label in sub_configs:
        pairs = target_set_by_type.get((is_s3_target, ctry), [])
        if not pairs:
            continue
        print(f"\nBuilding FAISS Index for {label} ({len(idx_list):,} records) to evaluate {len(pairs)} missed pairs...", flush=True)
        t0 = time.time()
        sub_vecs = tgt_vecs[idx_list].astype(np.float32)
        idx_faiss = faiss.IndexFlatIP(256)
        idx_faiss.add(sub_vecs)
        sub_tids = [tgt_ids[x] for x in idx_list]
        tid_to_subpos = {tid: pos for pos, tid in enumerate(sub_tids)}
        print(f"  FAISS index built in {time.time()-t0:.1f}s", flush=True)

        # Prepare queries
        for s1, m in pairs:
            v_s1 = id2vec.get(s1)
            v_m = id2vec.get(m)
            true_pos = tid_to_subpos.get(m)
            if true_pos is None:
                continue

            # Query 1: From Accepted Sibling in opposite source
            sibs = accepted_s2.get(s1, []) if is_s3_target else accepted_s3.get(s1, [])
            for sib in sibs:
                v_sib = id2vec.get(sib)
                if v_sib is None:
                    continue
                # Query sibling alone
                q_sib = v_sib.reshape(1, -1).astype(np.float32)
                _, I_sib = idx_faiss.search(q_sib, 500)
                rank_sib = np.where(I_sib[0] == true_pos)[0]
                if len(rank_sib) > 0:
                    r = rank_sib[0] + 1
                    for k_th in top_k_thresholds:
                        if r <= k_th:
                            if is_s3_target:
                                recovered_by_topk_s2_to_s3[k_th].add((s1, m))
                            else:
                                recovered_by_topk_s3_to_s2[k_th].add((s1, m))

                # Query Composite S1 + Sibling
                if v_s1 is not None:
                    v_comp = (v_s1 + v_sib) / np.linalg.norm(v_s1 + v_sib)
                    q_comp = v_comp.reshape(1, -1).astype(np.float32)
                    _, I_comp = idx_faiss.search(q_comp, 500)
                    rank_comp = np.where(I_comp[0] == true_pos)[0]
                    if len(rank_comp) > 0:
                        r = rank_comp[0] + 1
                        for k_th in top_k_thresholds:
                            if r <= k_th:
                                recovered_by_topk_composite[k_th].add((s1, m))

        del idx_faiss, sub_vecs, sub_tids, tid_to_subpos
        gc.collect()

    # 8. Report the Complete Recovery Matrix
    print("\n" + "=" * 75)
    print("EXP_N014 TARGETED RECOVERY MATRIX ON THE 804 BLOCKER MISSES")
    print("=" * 75)
    print(f"{'top_k Depth':15s} | {'S2->S3 Misses':18s} | {'S3->S2 Misses':18s} | {'Composite S1+Sib':18s} | {'Union Recovered':16s} | {'% of 804 Recovered'}")
    print("-" * 115)

    best_union_recovered = set()
    best_candidate_recall = 0.0

    for k_th in top_k_thresholds:
        s2_s3 = recovered_by_topk_s2_to_s3[k_th]
        s3_s2 = recovered_by_topk_s3_to_s2[k_th]
        comp = recovered_by_topk_composite[k_th]
        union_rec = s2_s3 | s3_s2 | comp
        pct = len(union_rec) / len(remaining_804) * 100
        print(f"top_k = {k_th:<6d} | {len(s2_s3):<18d} | {len(s3_s2):<18d} | {len(comp):<18d} | {len(union_rec):<16d} | {pct:6.2f}%")
        if k_th == 500:
            best_union_recovered = union_rec

    # 9. Calculate the True Unified Multi-Path Candidate Ceiling
    total_pool_true_matches = 85226 + len(n013_graph_matches) + len(best_union_recovered)
    candidate_ceiling = total_pool_true_matches / total_true_matches
    remaining_unrecoverable = total_true_matches - total_pool_true_matches

    print("\n" + "=" * 75)
    print("CRITICAL CANDIDATE CEILING GATE (PHASE 7)")
    print("=" * 75)
    print(f"Total Ground Truth Matches:              {total_true_matches:,}")
    print(f"N010/N011 Starting Candidate Pool:       85,226 (98.833%)")
    print(f"N013 S1-S2-S3 Graph Closure:             +{len(n013_graph_matches)} matches")
    print(f"N014 Bidirectional Sibling ANN:          +{len(best_union_recovered)} matches")
    print(f"TOTAL RECOVERED CANDIDATE MATCHES:       {total_pool_true_matches:,} / {total_true_matches:,}")
    print(f"UNIFIED CANDIDATE RECALL CEILING:        {candidate_ceiling*100:.4f}%")
    print(f"Remaining Blocker Misses:                {remaining_unrecoverable:,} ({remaining_unrecoverable/total_true_matches*100:.4f}%)")
    print("=" * 75)

    if candidate_ceiling >= 0.9950:
        print("[GATE RESULT] SUCCESS: Unified Candidate Ceiling >= 99.50% achieved! Proceed to final entity reranker.", flush=True)
    else:
        print(f"[GATE RESULT] Candidate ceiling ({candidate_ceiling*100:.4f}%) is below 99.50%. Residual analysis required.", flush=True)


if __name__ == "__main__":
    main()
