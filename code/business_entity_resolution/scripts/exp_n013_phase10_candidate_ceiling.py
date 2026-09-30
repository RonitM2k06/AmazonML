#!/usr/bin/env python3
"""
EXP_N013 — Phase 10: Critical Unified Candidate Ceiling & Blocker Miss Audit.

Measures candidate recall across all retrieval layers on the frozen 25k validation set:
1. N012 Lexical Candidate Recall
2. N010/N011 Dense Neural ANN Candidate Recall
3. Cross-Source S2 <-> S3 Graph Transitive Recall
4. Unified Multi-Path Ceiling (Lexical + Neural + Graph)
5. Fine-grained forensic breakdown of the remaining blocker misses.
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
from normalization import normalize_business_name, normalize_address

DATA_DIR = PROJECT_ROOT / "dataset" / "train"
EXP_DIR = CODE_DIR / "experiments"


def main():
    t_start = time.time()
    print("=" * 75, flush=True)
    print("[EXP_N013] Phase 10: Critical Unified Candidate Ceiling Analysis", flush=True)
    print("=" * 75, flush=True)

    # 1. Load Ground Truth for 25k Validation S1
    val_s1_set = _load_val_s1_ids()
    gt_full = _load_gt(val_s1_set)
    total_true_matches = sum(len(mids) for mids in gt_full.values())
    total_s1 = len(val_s1_set)
    print(f"Validation Population: {total_s1:,} S1 entities. Total Ground Truth Matches: {total_true_matches:,}", flush=True)

    # 2. Candidate Recall in N009 (Lexical only)
    print("\n--- Measuring N009 Lexical Candidate Recall ---", flush=True)
    v9 = np.load(EXP_DIR / "val_pairs_n009.npz", allow_pickle=True)
    s1_09 = v9["s1_ids"].astype(str)
    c_09 = v9["cand_ids"].astype(str)
    y_09 = v9["y"]
    lexical_hits = int(y_09.sum())
    lexical_recall = lexical_hits / total_true_matches
    print(f"N009 Lexical Blocker: {lexical_hits:,} / {total_true_matches:,} true matches ({lexical_recall*100:.2f}% recall, misses: {total_true_matches - lexical_hits:,})", flush=True)

    # 3. Candidate Recall in N010/N011/N012 (Lexical + Neural ANN)
    print("\n--- Measuring N010/N011 Unified Candidate Recall ---", flush=True)
    va = np.load(EXP_DIR / "val_pairs_n010.npz", allow_pickle=True)
    s1_10 = va["s1_ids"].astype(str)
    c_10 = va["cand_ids"].astype(str)
    y_10 = va["y"]
    union_hits = int(y_10.sum())
    union_recall = union_hits / total_true_matches
    union_misses = total_true_matches - union_hits
    print(f"N010/N011 Candidate Pool: {union_hits:,} / {total_true_matches:,} true matches ({union_recall*100:.2f}% recall, misses: {union_misses:,})", flush=True)

    # 4. Extract Blocker Misses
    existing_pool = set(zip(s1_10, c_10))
    blocker_miss_list = []  # (s1, missed_target, source)
    for s1, mids in gt_full.items():
        for m in mids:
            if (s1, m) not in existing_pool:
                src = "S3" if m.startswith("S3") else "S2"
                blocker_miss_list.append((s1, m, src))

    assert len(blocker_miss_list) == union_misses, f"Expected {union_misses} misses, got {len(blocker_miss_list)}"
    print(f"\nExtracted exactly {len(blocker_miss_list)} blocker misses to analyze for Graph & Multi-Path recovery.", flush=True)

    # 5. Load High-Confidence Predictions from N011 to anchor Graph
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

    accepted_preds = {}
    for k in range(len(unique_s1)):
        sl = slice(starts[k], ends[k])
        ps = probs_s[sl]
        cids = c_ids_s[sl]
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
        accepted_preds[unique_s1[k]] = pred_set

    # 6. Stream raw records for all entities involved in blocker misses
    needed_s1 = set(s1 for s1, _, _ in blocker_miss_list)
    needed_targets = set(m for _, m, _ in blocker_miss_list)
    for s1, _, _ in blocker_miss_list:
        for p in accepted_preds.get(s1, set()):
            needed_targets.add(p)

    recs = {}
    with open(DATA_DIR / "train_source1.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in needed_s1:
                recs[p[0]] = {
                    "name": p[1] if len(p) > 1 else "",
                    "address": p[2] if len(p) > 2 else "",
                    "country": p[3] if len(p) > 3 else ""
                }

    for sf in ["train_source2.tsv", "train_source3.tsv"]:
        with open(DATA_DIR / sf, encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if p[0] in needed_targets:
                    recs[p[0]] = {
                        "name": p[1] if len(p) > 1 else "",
                        "address": p[2] if len(p) > 2 else "",
                        "country": p[3] if len(p) > 3 else ""
                    }

    def clean_txt(s):
        return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split()) if s else ""

    def tokens(s):
        return set(clean_txt(s).split())

    def jac(s1, s2):
        return len(s1 & s2) / max(1, len(s1 | s2))

    def extract_house_num(addr):
        m = re.match(r"^(\d+[\w\-]*)", addr.strip())
        return m.group(1).lower() if m else ""

    # 7. Forensic Breakdown of Blocker Misses
    print("\n" + "=" * 75)
    print("DETAILED FORENSIC AUDIT OF THE 1,006 BLOCKER MISSES")
    print("=" * 75)

    by_country = defaultdict(int)
    by_source = defaultdict(int)
    by_recovery_path = defaultdict(int)
    by_name_sim = defaultdict(int)
    by_addr_sim = defaultdict(int)
    recovered_set = set()

    for idx, (s1, missed_id, missed_src) in enumerate(blocker_miss_list):
        r_miss = recs.get(missed_id, {})
        r_s1 = recs.get(s1, {})
        ctry = r_miss.get("country") or r_s1.get("country") or "Unknown"
        by_country[ctry] += 1
        by_source[missed_src] += 1

        t_nm_s1 = tokens(r_s1.get("name", ""))
        t_nm_m = tokens(r_miss.get("name", ""))
        t_ad_s1 = tokens(r_s1.get("address", ""))
        t_ad_m = tokens(r_miss.get("address", ""))

        nm_jac = jac(t_nm_s1, t_nm_m)
        ad_jac = jac(t_ad_s1, t_ad_m)

        # Name similarity bucket
        if nm_jac >= 0.70:
            by_name_sim[">= 0.70 (High overlap)"] += 1
        elif nm_jac >= 0.40:
            by_name_sim["0.40 - 0.70 (Medium overlap)"] += 1
        elif nm_jac > 0.0:
            by_name_sim["0.01 - 0.40 (Low overlap)"] += 1
        else:
            by_name_sim["0.00 (Zero lexical name overlap)"] += 1

        # Address similarity bucket
        if ad_jac >= 0.70:
            by_addr_sim[">= 0.70 (High address match)"] += 1
        elif ad_jac >= 0.40:
            by_addr_sim["0.40 - 0.70 (Medium address match)"] += 1
        elif ad_jac > 0.0:
            by_addr_sim["0.01 - 0.40 (Low address match)"] += 1
        else:
            by_addr_sim["0.00 (Zero lexical address overlap)"] += 1

        # Graph / Multi-path Recovery Checks
        preds = accepted_preds.get(s1, set())
        norm_ad_m = clean_txt(r_miss.get("address", ""))
        norm_nm_m = clean_txt(r_miss.get("name", ""))
        hn_m = extract_house_num(r_miss.get("address", ""))

        # Path 1: Direct exact address with S1
        if norm_ad_m and norm_ad_m == clean_txt(r_s1.get("address", "")):
            by_recovery_path["Direct Exact Address with S1"] += 1
            recovered_set.add(idx)

        # Path 2: Exact address with accepted sibling
        found_sib_exact_addr = False
        found_sib_high_overlap = False
        found_sib_hn_match = False

        for sib_id in preds:
            r_sib = recs.get(sib_id, {})
            norm_ad_sib = clean_txt(r_sib.get("address", ""))
            norm_nm_sib = clean_txt(r_sib.get("name", ""))
            t_ad_sib = tokens(r_sib.get("address", ""))
            t_nm_sib = tokens(r_sib.get("name", ""))
            hn_sib = extract_house_num(r_sib.get("address", ""))

            if norm_ad_m and norm_ad_m == norm_ad_sib:
                found_sib_exact_addr = True
            if hn_m and hn_m == hn_sib and jac(t_ad_m, t_ad_sib) >= 0.40:
                found_sib_hn_match = True
            if jac(t_ad_m, t_ad_sib) >= 0.50 and jac(t_nm_m, t_nm_sib) >= 0.40:
                found_sib_high_overlap = True

        if found_sib_exact_addr:
            by_recovery_path["Transitive Exact Address with Sibling"] += 1
            recovered_set.add(idx)
        if found_sib_hn_match:
            by_recovery_path["Transitive House-Number + Street Tokens with Sibling"] += 1
            recovered_set.add(idx)
        if found_sib_high_overlap:
            by_recovery_path["Transitive High Dual Name+Address Overlap with Sibling"] += 1
            recovered_set.add(idx)

    print(f"Total Blocker Misses: {len(blocker_miss_list):,}")
    print("\nBreakdown by Target Source:")
    for k, v in sorted(by_source.items()):
        print(f"  {k:15s}: {v:5d} ({v/len(blocker_miss_list)*100:5.2f}%)")

    print("\nBreakdown by Country:")
    for k, v in sorted(by_country.items()):
        print(f"  {k:15s}: {v:5d} ({v/len(blocker_miss_list)*100:5.2f}%)")

    print("\nName Token Overlap with S1:")
    for k, v in sorted(by_name_sim.items()):
        print(f"  {k:35s}: {v:5d} ({v/len(blocker_miss_list)*100:5.2f}%)")

    print("\nAddress Token Overlap with S1:")
    for k, v in sorted(by_addr_sim.items()):
        print(f"  {k:35s}: {v:5d} ({v/len(blocker_miss_list)*100:5.2f}%)")

    print("\nGraph Recovery Mechanisms:")
    for k, v in sorted(by_recovery_path.items()):
        print(f"  {k:45s}: {v:5d} ({v/len(blocker_miss_list)*100:5.2f}%)")

    print("\n" + "=" * 75)
    print("UNIFIED CANDIDATE CEILING REPORT")
    print("=" * 75)
    print(f"Ground Truth Total Matches:           {total_true_matches:,}")
    print(f"N009 Lexical Candidate Recall:         {lexical_hits:,} ({lexical_recall*100:.2f}%)")
    print(f"N010/N011 Lexical + ANN Recall:        {union_hits:,} ({union_recall*100:.2f}%)")
    
    total_unified_recovered = union_hits + len(recovered_set)
    unified_ceiling = total_unified_recovered / total_true_matches
    print(f"Recovered via S1-S2-S3 Graph Closure:  +{len(recovered_set):,} matches")
    print(f"UNIFIED MULTI-PATH CANDIDATE CEILING:  {total_unified_recovered:,} / {total_true_matches:,} ({unified_ceiling*100:.4f}% recall)")
    print(f"Remaining Missed Matches:              {total_true_matches - total_unified_recovered:,} ({100 - unified_ceiling*100:.4f}%)")
    print("=" * 75)


if __name__ == "__main__":
    main()
