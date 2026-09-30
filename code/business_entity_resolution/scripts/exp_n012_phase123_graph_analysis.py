#!/usr/bin/env python3
"""
EXP_N012 — Phases 1, 2, 3: High-Confidence Graph Extraction & Blocker Miss Recovery Analysis.
"""

import os
import sys
import time
import random
from collections import defaultdict
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1. Load validation set and ground truth
sys.path.insert(0, "scripts")
from authoritative_eval_fn import _load_val_s1_ids, _load_gt

print("=" * 70)
print("[EXP_N012] Phase 1, 2 & 3: Graph Construction & Blocker Recovery Audit")
print("=" * 70)

val_s1_set = _load_val_s1_ids()
gt_full = _load_gt(val_s1_set)
print(f"Loaded {len(val_s1_set)} validation S1 entities. Ground truth entities: {len(gt_full)}")

# 2. Load N011 validation predictions and probabilities
print("\nLoading N010 validation pairs and N011 calibrated probabilities...")
va = np.load("experiments/val_pairs_n010.npz", allow_pickle=True)
probs_lgb = np.load("experiments/probs_n011_optimal_lgb.npy").astype(np.float32)
probs_xgb = np.load("experiments/probs_n011_optimal_xgb.npy").astype(np.float32)
probs = 0.45 * probs_lgb + 0.55 * probs_xgb

s1_ids = va["s1_ids"].astype(str)
cand_ids = va["cand_ids"].astype(str)
sources = va["sources"].astype(str)
y = va["y"].astype(np.int8)

# Grouping
unique_s1, s1_codes = np.unique(s1_ids, return_inverse=True)
sort_key = np.lexsort((-probs, s1_codes))
s1_codes_s = s1_codes[sort_key]
probs_s = probs[sort_key]
cand_ids_s = cand_ids[sort_key]
sources_s = sources[sort_key]
y_s = y[sort_key]

boundaries = np.where(np.diff(s1_codes_s) != 0)[0] + 1
starts = np.concatenate([[0], boundaries])
ends = np.concatenate([boundaries, [len(probs_s)]])

base_th = 0.940
dense_th = 0.950
margin = 0.025
anchor = 0.960
bridge = 0.920
density_cutoff = 75
sg = base_th * 0.85

accepted_preds = {} # s1 -> set(cids)
high_conf_preds = {} # s1 -> set(cids) with prob >= 0.96
fn_model_list = [] # (s1, cid, source, prob)
fn_blocker_list = [] # (s1, cid, source)
fp_list = [] # (s1, cid, source, prob)

for k, s1_id in enumerate(unique_s1):
    sl = slice(starts[k], ends[k])
    ps = probs_s[sl]
    cids = cand_ids_s[sl]
    s3 = sources_s[sl] == "S3"
    ys = y_s[sl]

    true_full = gt_full.get(s1_id, set())
    cand_pool = set(cids.tolist())
    true_in_pool = true_full & cand_pool
    blocker_miss = true_full - cand_pool

    # Record blocker misses
    for bm in blocker_miss:
        fn_blocker_list.append((s1_id, bm, "S3" if bm.startswith("S3") else "S2"))

    # Decision rule
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

    accepted_preds[s1_id] = pred_set
    high_conf_preds[s1_id] = set(cids[ps >= 0.96])

    # Record model FNs and FPs
    for cid_val, p_val, y_val, src_val in zip(cids, ps, ys, sources_s[sl]):
        if y_val == 1 and cid_val not in pred_set:
            fn_model_list.append((s1_id, cid_val, src_val, float(p_val)))
        if cid_val in pred_set and cid_val not in true_full:
            fp_list.append((s1_id, cid_val, src_val, float(p_val)))

# Entities not in NPZ
for s1_id in val_s1_set:
    if s1_id not in accepted_preds:
        true_full = gt_full.get(s1_id, set())
        for bm in true_full:
            fn_blocker_list.append((s1_id, bm, "S3" if bm.startswith("S3") else "S2"))

print(f"\nVerification of Baseline Discrepancies:")
print(f"  Accepted Predictions Total: {sum(len(v) for v in accepted_preds.values())}")
print(f"  False Positives (FP): {len(fp_list)} (expected 112)")
print(f"  False Negatives Model (FN-model): {len(fn_model_list)} (expected 3323)")
print(f"  False Negatives Blocker (FN-blocker): {len(fn_blocker_list)} (expected 1006)")

# 3. Collect Raw Records
needed_s1 = set(s1 for s1, _, _ in fn_blocker_list) | set(s1 for s1, _, _, _ in fn_model_list) | set(s1 for s1, _, _, _ in fp_list)
needed_targets = set(cid for _, cid, _ in fn_blocker_list) | set(cid for _, cid, _, _ in fn_model_list) | set(cid for _, cid, _, _ in fp_list)

# Also need raw records of siblings for S1 entities that have blocker misses or model misses
for s1 in set(s1 for s1, _, _ in fn_blocker_list) | set(s1 for s1, _, _, _ in fn_model_list):
    # matched siblings
    for cid in accepted_preds.get(s1, set()):
        needed_targets.add(cid)
    # ground truth siblings
    for cid in gt_full.get(s1, set()):
        needed_targets.add(cid)

print(f"\nStreaming raw text records for {len(needed_s1)} S1s and {len(needed_targets)} Targets...")

raw_records = {}

# S1
with open("../../dataset/train/train_source1.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        if p[0] in needed_s1:
            raw_records[p[0]] = {
                "name": p[1] if len(p) > 1 else "",
                "address": p[2] if len(p) > 2 else "",
                "country": p[3] if len(p) > 3 else ""
            }

# S2
with open("../../dataset/train/train_source2.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        if p[0] in needed_targets:
            raw_records[p[0]] = {
                "name": p[1] if len(p) > 1 else "",
                "address": p[2] if len(p) > 2 else "",
                "country": p[3] if len(p) > 3 else ""
            }

# S3
with open("../../dataset/train/train_source3.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        if p[0] in needed_targets:
            raw_records[p[0]] = {
                "name": p[1] if len(p) > 1 else "",
                "address": p[2] if len(p) > 2 else "",
                "country": p[3] if len(p) > 3 else ""
            }

print(f"Loaded {len(raw_records)} raw records.")

# Helper normalization functions
import re
def norm_text(s):
    if not s: return ""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())

def extract_tokens(s):
    return set(norm_text(s).split())

def jaccard(set1, set2):
    if not set1 or not set2: return 0.0
    return len(set1 & set2) / len(set1 | set2)

def extract_house_num(addr):
    m = re.match(r"^(\d+[\w\-]*)", addr.strip())
    return m.group(1).lower() if m else ""

# 4. Phase 3 Audit: Recovering the 1,006 Blocker Misses
print("\n" + "=" * 70)
print("PHASE 3: RECOVERING THE 1,006 BLOCKER MISSES AUDIT")
print("=" * 70)

# Categories
mech_counts = defaultdict(int)
recovered_any = set()
recovered_by_mech = defaultdict(set)

for idx, (s1, missed_id, missed_src) in enumerate(fn_blocker_list):
    other_src = "S2" if missed_src == "S3" else "S3"
    rec_missed = raw_records.get(missed_id, {})
    rec_s1 = raw_records.get(s1, {})
    
    missed_name_tokens = extract_tokens(rec_missed.get("name", ""))
    missed_addr_tokens = extract_tokens(rec_missed.get("address", ""))
    missed_norm_addr = norm_text(rec_missed.get("address", ""))
    missed_norm_name = norm_text(rec_missed.get("name", ""))
    missed_hn = extract_house_num(rec_missed.get("address", ""))
    
    # 1. Direct address match between S1 and missed_id
    s1_norm_addr = norm_text(rec_s1.get("address", ""))
    s1_addr_tokens = extract_tokens(rec_s1.get("address", ""))
    if missed_norm_addr and s1_norm_addr == missed_norm_addr:
        mech_counts["D_direct_norm_addr_match_with_s1"] += 1
        recovered_any.add(idx)
        recovered_by_mech["D"].add(idx)

    # 2. Check through accepted sibling matches for this S1
    matched_siblings = accepted_preds.get(s1, set())
    has_accepted_other_src = any(cid.startswith(other_src) for cid in matched_siblings)
    has_accepted_same_src = any(cid.startswith(missed_src) for cid in matched_siblings)
    
    if has_accepted_other_src:
        mech_counts["A_has_accepted_other_source_sibling"] += 1
    if has_accepted_same_src:
        mech_counts["has_accepted_same_source_sibling"] += 1
        
    found_graph_link = False
    found_shared_addr = False
    found_shared_hn = False
    found_shared_name = False
    
    for sib_id in matched_siblings:
        sib_rec = raw_records.get(sib_id, {})
        sib_norm_addr = norm_text(sib_rec.get("address", ""))
        sib_norm_name = norm_text(sib_rec.get("name", ""))
        sib_addr_tokens = extract_tokens(sib_rec.get("address", ""))
        sib_name_tokens = extract_tokens(sib_rec.get("name", ""))
        sib_hn = extract_house_num(sib_rec.get("address", ""))
        
        # Check transitive edge conditions
        addr_jac = jaccard(missed_addr_tokens, sib_addr_tokens)
        name_jac = jaccard(missed_name_tokens, sib_name_tokens)
        
        # D. shared normalized address
        if sib_norm_addr and sib_norm_addr == missed_norm_addr:
            found_shared_addr = True
            
        # E. shared postal + house number / high address jaccard + house number
        if sib_hn and sib_hn == missed_hn and addr_jac >= 0.40:
            found_shared_hn = True
            
        # Transitive strong link
        if (sib_norm_addr == missed_norm_addr and name_jac >= 0.30) or \
           (sib_norm_name == missed_norm_name and addr_jac >= 0.30) or \
           (addr_jac >= 0.60 and name_jac >= 0.40) or \
           (sib_hn and sib_hn == missed_hn and addr_jac >= 0.40 and name_jac >= 0.30):
            found_graph_link = True

    if found_graph_link:
        mech_counts["B_or_C_transitive_graph_link"] += 1
        recovered_any.add(idx)
        recovered_by_mech["graph"].add(idx)
        
    if found_shared_addr:
        mech_counts["D_shared_norm_addr_with_sibling"] += 1
        recovered_any.add(idx)
        recovered_by_mech["shared_addr"].add(idx)
        
    if found_shared_hn:
        mech_counts["E_shared_hn_and_addr_tokens"] += 1
        recovered_any.add(idx)
        recovered_by_mech["shared_hn"].add(idx)

total_bm = len(fn_blocker_list)
print(f"Total Blocker Misses: {total_bm}")
for k, v in sorted(mech_counts.items()):
    print(f"  {k:40s}: {v:5d} / {total_bm} ({v/total_bm*100:5.2f}%)")
print(f"  --> ANY of above recovery mechanisms: {len(recovered_any):5d} / {total_bm} ({len(recovered_any)/total_bm*100:5.2f}%)")

# 5. Model False Negatives (3,323) Audit
print("\n" + "=" * 70)
print("PHASE 7: MODEL FALSE NEGATIVES (3,323) CATEGORIZATION")
print("=" * 70)

p_bands = defaultdict(int)
has_sibling_acc = 0
strong_graph_support = 0

for s1, cid, src, prob in fn_model_list:
    # Bin probability
    if prob >= 0.90:
        p_bands["0.90 <= p < 0.94 (Near Decision Threshold)"] += 1
    elif prob >= 0.80:
        p_bands["0.80 <= p < 0.90 (Medium-High Score)"] += 1
    elif prob >= 0.50:
        p_bands["0.50 <= p < 0.80 (Medium Score)"] += 1
    elif prob >= 0.10:
        p_bands["0.10 <= p < 0.50 (Low Score)"] += 1
    else:
        p_bands["p < 0.10 (Extremely Low Score)"] += 1
        
    # Check if there is an accepted sibling
    matched_siblings = accepted_preds.get(s1, set())
    if len(matched_siblings) > 0:
        has_sibling_acc += 1
        # Check graph support with accepted sibling
        rec_cand = raw_records.get(cid, {})
        cand_name_tokens = extract_tokens(rec_cand.get("name", ""))
        cand_addr_tokens = extract_tokens(rec_cand.get("address", ""))
        cand_hn = extract_house_num(rec_cand.get("address", ""))
        cand_norm_addr = norm_text(rec_cand.get("address", ""))
        cand_norm_name = norm_text(rec_cand.get("name", ""))
        
        for sib_id in matched_siblings:
            sib_rec = raw_records.get(sib_id, {})
            sib_norm_addr = norm_text(sib_rec.get("address", ""))
            sib_norm_name = norm_text(sib_rec.get("name", ""))
            addr_jac = jaccard(cand_addr_tokens, extract_tokens(sib_rec.get("address", "")))
            name_jac = jaccard(cand_name_tokens, extract_tokens(sib_rec.get("name", "")))
            sib_hn = extract_house_num(sib_rec.get("address", ""))
            
            if (sib_norm_addr == cand_norm_addr and name_jac >= 0.25) or \
               (sib_norm_name == cand_norm_name and addr_jac >= 0.25) or \
               (addr_jac >= 0.50 and name_jac >= 0.35) or \
               (sib_hn and sib_hn == cand_hn and addr_jac >= 0.35 and name_jac >= 0.25):
                strong_graph_support += 1
                break

print("Probability distribution of the 3,323 model false negatives:")
for k, v in sorted(p_bands.items()):
    print(f"  {k:50s}: {v:5d} ({v/len(fn_model_list)*100:5.2f}%)")
print(f"\nModel FNs with at least 1 accepted sibling match: {has_sibling_acc} / {len(fn_model_list)} ({has_sibling_acc/len(fn_model_list)*100:5.2f}%)")
print(f"Model FNs with strong graph support from accepted sibling: {strong_graph_support} / {len(fn_model_list)} ({strong_graph_support/len(fn_model_list)*100:5.2f}%)")

# 6. False Positives (112) Audit
print("\n" + "=" * 70)
print("PHASE 7: FALSE POSITIVES (112) GRAPH CONTRADICTION AUDIT")
print("=" * 70)

fp_has_sibling = 0
fp_graph_contradicts = 0
fp_ann_only = 0

for s1, cid, src, prob in fp_list:
    matched_siblings = [c for c in accepted_preds.get(s1, set()) if c != cid]
    if len(matched_siblings) > 0:
        fp_has_sibling += 1
        rec_fp = raw_records.get(cid, {})
        fp_addr_tokens = extract_tokens(rec_fp.get("address", ""))
        fp_name_tokens = extract_tokens(rec_fp.get("name", ""))
        fp_hn = extract_house_num(rec_fp.get("address", ""))
        
        # Check if it contradicts all siblings (e.g. completely different house number or address jaccard < 0.1)
        contradicts_all = True
        for sib_id in matched_siblings:
            sib_rec = raw_records.get(sib_id, {})
            sib_hn = extract_house_num(sib_rec.get("address", ""))
            addr_jac = jaccard(fp_addr_tokens, extract_tokens(sib_rec.get("address", "")))
            if (sib_hn and fp_hn and sib_hn == fp_hn) or addr_jac >= 0.40:
                contradicts_all = False
                break
        if contradicts_all:
            fp_graph_contradicts += 1

print(f"Total False Positives: {len(fp_list)}")
print(f"  FPs with accepted sibling matches: {fp_has_sibling}")
print(f"  FPs that contradict sibling address/house-number: {fp_graph_contradicts} ({fp_graph_contradicts/len(fp_list)*100:5.2f}%)")
print("=" * 70)
