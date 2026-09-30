#!/usr/bin/env python3
"""
EXP_N015 — Phase 1: Comprehensive Forensic Audit of the Remaining 404 Blocker Misses.
"""

from __future__ import annotations

import os
import sys
import time
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONUNBUFFERED"] = "1"

# Paths
CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_DIR.parents[1]
sys.path.insert(0, str(CODE_DIR / "src"))
sys.path.insert(0, str(CODE_DIR / "scripts"))

from authoritative_eval_fn import _load_val_s1_ids, _load_gt

DATA_DIR = PROJECT_ROOT / "dataset" / "train"
EXP_DIR = CODE_DIR / "experiments"
MODEL_DIR = CODE_DIR / "models"


def main():
    t_start = time.time()
    print("=" * 75, flush=True)
    print("[EXP_N015] Phase 1: Forensic Audit of the Remaining 404 Blocker Misses", flush=True)
    print("=" * 75, flush=True)

    val_s1_set = _load_val_s1_ids()
    gt_full = _load_gt(val_s1_set)

    va = np.load(EXP_DIR / "val_pairs_n010.npz", allow_pickle=True)
    pl = np.load(EXP_DIR / "probs_n011_optimal_lgb.npy")
    px = np.load(EXP_DIR / "probs_n011_optimal_xgb.npy")
    probs = 0.45 * pl + 0.55 * px

    s1_ids_str = va["s1_ids"].astype(str)
    cand_ids_str = va["cand_ids"].astype(str)
    sources_str = va["sources"].astype(str)

    unique_s1, s1_codes = np.unique(s1_ids_str, return_inverse=True)
    sort_key = np.lexsort((-probs, s1_codes))
    s1_codes_s = s1_codes[sort_key]
    probs_s = probs[sort_key]
    cand_ids_s = cand_ids_str[sort_key]
    sources_s = sources_str[sort_key]

    boundaries = np.where(np.diff(s1_codes_s) != 0)[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(probs_s)]])

    base_th, dense_th, margin, anchor, bridge, density_cutoff = 0.940, 0.950, 0.025, 0.960, 0.920, 75
    sg = base_th * 0.85

    base_preds = {}
    accepted_s2 = defaultdict(list)
    accepted_s3 = defaultdict(list)

    for k in range(len(unique_s1)):
        sl = slice(starts[k], ends[k])
        ps = probs_s[sl]
        cids = cand_ids_s[sl]
        s3 = sources_s[sl] == "S3"
        N_ = len(ps)
        if N_ == 0 or ps[0] < sg:
            pred_set = set()
        else:
            top = ps[0]
            et = dense_th if N_ > density_cutoff else base_th
            am = (ps >= et) & ((top - ps) <= margin)
            anm = ps >= anchor
            if np.any(anm & am):
                if not bool(np.any(am & s3)):
                    am = am | ((~am) & s3 & (ps >= bridge))
                if not bool(np.any(am & ~s3)):
                    am = am | ((~am) & ~s3 & (ps >= bridge))
            pred_set = set(cids[am])

        s1_id = unique_s1[k]
        base_preds[s1_id] = pred_set
        for cid in pred_set:
            if cid.startswith("S2"):
                accepted_s2[s1_id].append(cid)
            else:
                accepted_s3[s1_id].append(cid)

    # N010/N011 pool
    existing_pool = set(zip(s1_ids_str, cand_ids_str))
    all_blocker_misses = [(s1, m) for s1, mids in gt_full.items() for m in mids if (s1, m) not in existing_pool]

    # Stream raw records for all needed entities
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

    # N013 Graph Closure matches (+94 matches)
    n013_graph_matches = defaultdict(set)
    for s1, m in all_blocker_misses:
        r_miss = recs.get(m, ("", "", "", ""))
        r_s1 = recs.get(s1, ("", "", "", ""))
        ad_m = clean_txt(r_miss[2])
        hn_m = extract_hn(r_miss[2])

        if ad_m and ad_m == clean_txt(r_s1[2]):
            n013_graph_matches[s1].add(m)
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
                n013_graph_matches[s1].add(m)
                break
            if hn_m and hn_m == hn_sib and jac(t_ad_m, t_ad_sib) >= 0.40 and jac(t_nm_m, t_nm_sib) >= 0.25:
                n013_graph_matches[s1].add(m)
                break

    # EXP_N014 Sibling ANN matches (+508 matches)
    remaining_after_graph = [(s1, m) for s1, m in all_blocker_misses if m not in n013_graph_matches.get(s1, set())]

    import torch
    from two_tower_encoder import load_two_tower, RecordTokenizer, encode_records_fused
    import faiss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    two_tower = load_two_tower(str(MODEL_DIR / "two_tower_stageC.pt"), device=device)
    tok = RecordTokenizer()

    ids_list, vecs = encode_records_fused(two_tower, tok, recs, batch_size=256, device=device)
    id2vec = {id_: vecs[i] for i, id_ in enumerate(ids_list)}

    tgt_emb = np.load(EXP_DIR / "embeddings_n010_target.npz", allow_pickle=True)
    tgt_ids = tgt_emb["ids"]
    tgt_vecs = tgt_emb["vecs"]
    N_tgt = len(tgt_ids)

    tgt_country = {}
    for sf in ["train_source2.tsv", "train_source3.tsv"]:
        with open(DATA_DIR / sf, encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if len(p) > 3:
                    tgt_country[p[0]] = p[3]

    s2_india_idx = [i for i in range(5034616) if tgt_country.get(tgt_ids[i]) == "India"]
    s2_us_idx = [i for i in range(5034616) if tgt_country.get(tgt_ids[i]) == "US"]
    s3_india_idx = [i for i in range(5034616, N_tgt) if tgt_country.get(tgt_ids[i]) == "India"]
    s3_us_idx = [i for i in range(5034616, N_tgt) if tgt_country.get(tgt_ids[i]) == "US"]

    target_set_by_type = defaultdict(list)
    for s1, m in remaining_after_graph:
        r_m = recs.get(m, ("", "", "", ""))
        ctry = r_m[3] if len(r_m) > 3 and r_m[3] else "India"
        is_s3 = m.startswith("S3")
        target_set_by_type[(is_s3, ctry)].append((s1, m))

    recovered_n014_matches = set()
    sub_configs = [
        (True, "India", s3_india_idx),
        (True, "US", s3_us_idx),
        (False, "India", s2_india_idx),
        (False, "US", s2_us_idx),
    ]

    for is_s3_target, ctry, idx_list in sub_configs:
        pairs = target_set_by_type.get((is_s3_target, ctry), [])
        if not pairs:
            continue
        sub_vecs = tgt_vecs[idx_list].astype(np.float32)
        idx_faiss = faiss.IndexFlatIP(256)
        idx_faiss.add(sub_vecs)
        sub_tids = [tgt_ids[x] for x in idx_list]
        tid_to_subpos = {tid: pos for pos, tid in enumerate(sub_tids)}

        for s1, m in pairs:
            v_s1 = id2vec.get(s1)
            v_m = id2vec.get(m)
            true_pos = tid_to_subpos.get(m)
            if true_pos is None:
                continue

            sibs = accepted_s2.get(s1, []) if is_s3_target else accepted_s3.get(s1, [])
            for sib in sibs:
                v_sib = id2vec.get(sib)
                if v_sib is None:
                    continue
                q_sib = v_sib.reshape(1, -1).astype(np.float32)
                _, I_sib = idx_faiss.search(q_sib, 500)
                if true_pos in I_sib[0]:
                    recovered_n014_matches.add((s1, m))
                    break

                if v_s1 is not None:
                    v_comp = (v_s1 + v_sib) / np.linalg.norm(v_s1 + v_sib)
                    q_comp = v_comp.reshape(1, -1).astype(np.float32)
                    _, I_comp = idx_faiss.search(q_comp, 500)
                    if true_pos in I_comp[0]:
                        recovered_n014_matches.add((s1, m))
                        break

        del idx_faiss, sub_vecs, sub_tids, tid_to_subpos
        import gc
        gc.collect()

    # EXACT 404 REMAINING BLOCKER MISSES
    remaining_404 = [(s1, m) for s1, m in remaining_after_graph if (s1, m) not in recovered_n014_matches]
    print(f"\nIdentified EXACTLY {len(remaining_404)} remaining blocker misses.", flush=True)

    # FORENSIC CATEGORIZATION
    by_country = defaultdict(int)
    by_source = defaultdict(int)
    has_opposite_sib = 0
    has_same_sib = 0
    is_pure_singleton = 0
    by_name_sim = defaultdict(int)
    by_addr_sim = defaultdict(int)
    by_hn_agreement = defaultdict(int)
    neural_cos_list = []

    for s1, m in remaining_404:
        r_m = recs.get(m, ("", "", "", ""))
        r_s1 = recs.get(s1, ("", "", "", ""))
        ctry = r_m[3] if len(r_m) > 3 and r_m[3] else "India"
        src = "S3" if m.startswith("S3") else "S2"
        other_src = "S2" if src == "S3" else "S3"
        by_country[ctry] += 1
        by_source[src] += 1

        preds = base_preds.get(s1, set())
        has_opp = any(p.startswith(other_src) for p in preds)
        has_sm = any(p.startswith(src) for p in preds)
        if has_opp: has_opposite_sib += 1
        if has_sm: has_same_sib += 1
        if not has_opp and not has_sm: is_pure_singleton += 1

        t_nm_s1 = tokens(r_s1[1])
        t_nm_m = tokens(r_m[1])
        t_ad_s1 = tokens(r_s1[2])
        t_ad_m = tokens(r_m[2])
        hn_s1 = extract_hn(r_s1[2])
        hn_m = extract_hn(r_m[2])

        nm_jac = jac(t_nm_s1, t_nm_m)
        ad_jac = jac(t_ad_s1, t_ad_m)

        if nm_jac >= 0.70: by_name_sim[">= 0.70 (High)"] += 1
        elif nm_jac >= 0.30: by_name_sim["0.30 - 0.70 (Medium)"] += 1
        elif nm_jac > 0.0: by_name_sim["0.01 - 0.30 (Low)"] += 1
        else: by_name_sim["0.00 (Zero name overlap)"] += 1

        if ad_jac >= 0.70: by_addr_sim[">= 0.70 (High)"] += 1
        elif ad_jac >= 0.30: by_addr_sim["0.30 - 0.70 (Medium)"] += 1
        elif ad_jac > 0.0: by_addr_sim["0.01 - 0.30 (Low)"] += 1
        else: by_addr_sim["0.00 (Zero addr overlap)"] += 1

        if hn_s1 and hn_m:
            if hn_s1 == hn_m: by_hn_agreement["Exact HN Match"] += 1
            else: by_hn_agreement["Different HN"] += 1
        else:
            by_hn_agreement["Missing HN in either"] += 1

        v_s1 = id2vec.get(s1)
        v_m = id2vec.get(m)
        if v_s1 is not None and v_m is not None:
            neural_cos_list.append(float(np.dot(v_s1, v_m)))

    cos_arr = np.array(neural_cos_list)

    print("\n" + "=" * 75)
    print("EXP_N015 FORENSIC AUDIT OF THE 404 REMAINING BLOCKER MISSES")
    print("=" * 75)
    print(f"Total Remaining Blocker Misses: {len(remaining_404):,}")
    print("\n1. Breakdown by Source:")
    for k, v in sorted(by_source.items()):
        print(f"   {k:15s}: {v:4d} ({v/len(remaining_404)*100:5.2f}%)")

    print("\n2. Breakdown by Country:")
    for k, v in sorted(by_country.items()):
        print(f"   {k:15s}: {v:4d} ({v/len(remaining_404)*100:5.2f}%)")

    print("\n3. Sibling Graph Context:")
    print(f"   Has Accepted Opposite-Source Sibling: {has_opposite_sib:4d} ({has_opposite_sib/len(remaining_404)*100:5.2f}%)")
    print(f"   Has Accepted Same-Source Sibling:     {has_same_sib:4d} ({has_same_sib/len(remaining_404)*100:5.2f}%)")
    print(f"   Pure Singleton Miss (Zero siblings):  {is_pure_singleton:4d} ({is_pure_singleton/len(remaining_404)*100:5.2f}%)")

    print("\n4. Name-Token Overlap with S1:")
    for k, v in sorted(by_name_sim.items()):
        print(f"   {k:30s}: {v:4d} ({v/len(remaining_404)*100:5.2f}%)")

    print("\n5. Address-Token Overlap with S1:")
    for k, v in sorted(by_addr_sim.items()):
        print(f"   {k:30s}: {v:4d} ({v/len(remaining_404)*100:5.2f}%)")

    print("\n6. House-Number Agreement:")
    for k, v in sorted(by_hn_agreement.items()):
        print(f"   {k:30s}: {v:4d} ({v/len(remaining_404)*100:5.2f}%)")

    print("\n7. Direct S1 Neural Cosine:")
    print(f"   Mean Cosine:   {cos_arr.mean():.4f}")
    print(f"   Median Cosine: {np.median(cos_arr):.4f}")
    print(f"   Cosine >= 0.50: {(cos_arr >= 0.50).sum():4d} ({(cos_arr >= 0.50).sum()/len(cos_arr)*100:5.2f}%)")
    print(f"   Cosine >= 0.40: {(cos_arr >= 0.40).sum():4d} ({(cos_arr >= 0.40).sum()/len(cos_arr)*100:5.2f}%)")
    print("=" * 75)


if __name__ == "__main__":
    main()
