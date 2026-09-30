import os
import sys
import time
import csv
from collections import Counter
import numpy as np

# Ensure UTF-8 output on Windows console safely
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

print("=== STEP 1: INITIALIZING SCRIPT ===", flush=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(CODE_DIR, "..", ".."))
SRC_DIR = os.path.join(CODE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from normalization import normalize_name, normalize_address, tokenize
from blocking import get_consonant_skeleton

REPORTS_DIR = os.path.join(CODE_DIR, "reports")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "dataset", "train")

print("=== STEP 2: LOADING ERROR CSVS ===", flush=True)

def load_csv_simple(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

fp_rows = load_csv_simple(os.path.join(REPORTS_DIR, "exp_n008_errors_fp.csv"))
fn_m_rows = load_csv_simple(os.path.join(REPORTS_DIR, "exp_n008_errors_fn_model.csv"))
fn_b_rows = load_csv_simple(os.path.join(REPORTS_DIR, "exp_n008_errors_fn_blocker.csv"))

print(f"Loaded: FP={len(fp_rows):,}, FN_model={len(fn_m_rows):,}, FN_blocker={len(fn_b_rows):,}", flush=True)

s1_needed = set()
cand_needed = set()

for r in fp_rows + fn_m_rows + fn_b_rows:
    if "s1_id" in r:
        s1_needed.add(r["s1_id"])
    if "cand_id" in r:
        cand_needed.add(r["cand_id"])

print(f"Unique entities required: S1={len(s1_needed):,}, Candidates={len(cand_needed):,}", flush=True)

print("=== STEP 3: STREAMING TSV FILES ===", flush=True)

def stream_records(path, needed_set):
    res = {}
    if not os.path.exists(path):
        print(f"Missing {path}", flush=True)
        return res
    t0 = time.time()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row.get("entity_id")
            if eid in needed_set:
                res[eid] = {
                    "name": row.get("name", ""),
                    "address": row.get("address", ""),
                    "country": row.get("country", "")
                }
                if len(res) == len(needed_set):
                    break
    print(f"Read {os.path.basename(path)}: found {len(res):,}/{len(needed_set):,} in {time.time()-t0:.1f}s", flush=True)
    return res

s1_data = stream_records(os.path.join(TRAIN_DIR, "train_source1.tsv"), s1_needed)

s2_needed = {c for c in cand_needed if c.startswith("S2")}
s3_needed = {c for c in cand_needed if c.startswith("S3")}

s2_data = stream_records(os.path.join(TRAIN_DIR, "train_source2.tsv"), s2_needed)
s3_data = stream_records(os.path.join(TRAIN_DIR, "train_source3.tsv"), s3_needed)

cand_data = {**s2_data, **s3_data}

print("=== STEP 4: CATEGORIZING FALSE POSITIVES (699) ===", flush=True)
fp_cats = Counter()
analysis_rows = []

for r in fp_rows:
    s1 = s1_data.get(r["s1_id"], {})
    cand = cand_data.get(r["cand_id"], {})
    s1_name = s1.get("name", "")
    cand_name = cand.get("name", "")
    s1_addr = s1.get("address", "")
    cand_addr = cand.get("address", "")

    n1 = normalize_name(s1_name)
    n2 = normalize_name(cand_name)
    t1 = set(tokenize(n1))
    t2 = set(tokenize(n2))
    name_overlap = len(t1 & t2) / max(1, min(len(t1), len(t2)))

    try:
        addr_jacc = float(r.get("addr_token_jaccard", 0.0))
        prob = float(r.get("prob", 0.0))
    except Exception:
        addr_jacc = 0.0
        prob = 0.0

    if n1 == n2 and addr_jacc < 0.2:
        cat = "A_same_name_diff_biz"
    elif addr_jacc > 0.5 and name_overlap < 0.3:
        cat = "B_same_addr_diff_biz"
    elif min(len(t1), len(t2)) <= 1 and name_overlap == 1.0:
        cat = "C_short_name_collision"
    elif "branch" in s1_addr.lower() or "branch" in cand_addr.lower() or name_overlap > 0.8:
        cat = "F_branch_or_chain_collision"
    elif prob < 0.965:
        cat = "H_marginal_threshold_error"
    else:
        cat = "I_other_semantic_error"

    fp_cats[cat] += 1
    analysis_rows.append({
        "error_type": "FP", "category": cat, "s1_id": r["s1_id"], "cand_id": r["cand_id"],
        "source": r.get("source", ""), "country": s1.get("country", ""),
        "s1_name": s1_name, "s1_addr": s1_addr, "cand_name": cand_name, "cand_addr": cand_addr,
        "prob": prob, "rank": r.get("rank", 0), "cosine_sim": r.get("cosine_sim", 0.0),
        "recovery_strategy": ""
    })

print("FP Categories:")
for k, v in fp_cats.most_common():
    print(f"  {k:30s}: {v:4d} ({v/len(fp_rows)*100:5.1f}%)")

print("=== STEP 5: CATEGORIZING MODEL FALSE NEGATIVES (1,788) ===", flush=True)
fn_m_cats = Counter()
for r in fn_m_rows:
    s1 = s1_data.get(r["s1_id"], {})
    cand = cand_data.get(r["cand_id"], {})
    s1_name = s1.get("name", "")
    cand_name = cand.get("name", "")
    s1_addr = s1.get("address", "")
    cand_addr = cand.get("address", "")

    try:
        prob = float(r.get("prob", 0.0))
        top_minus_p = float(r.get("top_minus_prob", 0.0))
        s1_miss = float(r.get("is_s1_addr_missing", 0.0))
        cand_miss = float(r.get("is_cand_addr_missing", 0.0))
        addr_jacc = float(r.get("addr_token_jaccard", 0.0))
        name_jacc = float(r.get("name_token_jaccard", 0.0))
    except Exception:
        prob = top_minus_p = s1_miss = cand_miss = addr_jacc = name_jacc = 0.0

    if prob < 0.50:
        cat = "B_low_model_score"
    elif prob >= 0.956 and top_minus_p > 0.025:
        cat = "C_margin_pruned"
    elif s1_miss > 0.5 or cand_miss > 0.5:
        cat = "G_missing_address_penalty"
    elif addr_jacc < 0.1:
        cat = "F_address_divergence"
    elif name_jacc < 0.3:
        cat = "E_name_variation"
    else:
        cat = "D_bridge_subthreshold"

    fn_m_cats[cat] += 1
    analysis_rows.append({
        "error_type": "FN_MODEL", "category": cat, "s1_id": r["s1_id"], "cand_id": r["cand_id"],
        "source": r.get("source", ""), "country": s1.get("country", ""),
        "s1_name": s1_name, "s1_addr": s1_addr, "cand_name": cand_name, "cand_addr": cand_addr,
        "prob": prob, "rank": r.get("rank", 0), "cosine_sim": r.get("cosine_sim", 0.0),
        "recovery_strategy": ""
    })

print("Model FN Categories:")
for k, v in fn_m_cats.most_common():
    print(f"  {k:30s}: {v:4d} ({v/len(fn_m_rows)*100:5.1f}%)")

print("=== STEP 6: CATEGORIZING BLOCKER MISSES (3,066) & RECOVERY ANALYSIS ===", flush=True)
fn_b_cats = Counter()
rec_counter = Counter()

# Precompute stopwords for address matching
ADDR_STOPS = {"india", "road", "rd", "street", "st", "lane", "nagar", "colony", "near", "opp", "opposite", "floor", "fl", "building", "bldg", "plot", "area", "bazar", "city"}

for r in fn_b_rows:
    s1 = s1_data.get(r["s1_id"], {})
    cand = cand_data.get(r["cand_id"], {})
    s1_name = s1.get("name", "")
    cand_name = cand.get("name", "")
    s1_addr = s1.get("address", "")
    cand_addr = cand.get("address", "")

    n1 = normalize_name(s1_name)
    n2 = normalize_name(cand_name)
    t1 = tokenize(n1)
    t2 = tokenize(n2)

    t_a1 = set(tokenize(normalize_address(s1_addr))) - ADDR_STOPS
    t_a2 = set(tokenize(normalize_address(cand_addr))) - ADDR_STOPS
    addr_overlap = t_a1 & t_a2

    # Check recovery mechanisms
    recoveries = []
    if n1 and n1 == n2:
        recoveries.append("exact_norm")
    
    # Rare or single token unconstrained
    common_tokens = [tok for tok in set(t1) & set(t2) if len(tok) >= 4]
    if common_tokens:
        recoveries.append("shared_token_len4plus")
    elif set(t1) & set(t2):
        recoveries.append("shared_token_short")

    # Phonetic consonant skeleton
    sk1 = [get_consonant_skeleton(tok) for tok in t1 if len(tok) >= 3]
    sk2 = [get_consonant_skeleton(tok) for tok in t2 if len(tok) >= 3]
    common_skels = [s for s in set(sk1) & set(sk2) if len(s) >= 3]
    if common_skels:
        recoveries.append("phonetic_skeleton_match")

    # Address anchors
    if len(addr_overlap) >= 2:
        recoveries.append("loose_address_overlap_2plus")
    elif len(addr_overlap) == 1:
        recoveries.append("single_distinctive_address_token")

    # Character ngram 4
    if len(n1) >= 4 and len(n2) >= 4:
        ng1 = {n1[i:i+4] for i in range(len(n1)-3)}
        ng2 = {n2[i:i+4] for i in range(len(n2)-3)}
        if len(ng1 & ng2) >= 2:
            recoveries.append("char_ngram4_shared")

    # Categorize cause
    if not s1_name or not cand_name:
        cat = "A_missing_name"
    elif any(ord(c) > 127 for c in s1_name) or any(ord(c) > 127 for c in cand_name):
        cat = "B_non_latin_script"
    elif len(set(t1) & set(t2)) == 0:
        if len(addr_overlap) >= 1:
            cat = "C_zero_name_overlap_address_match"
        else:
            cat = "D_no_surface_overlap_severe_noise"
    elif len(t1) <= 1 or len(t2) <= 1:
        cat = "E_short_name_unmatched"
    else:
        cat = "F_complex_variant"

    fn_b_cats[cat] += 1
    rec_str = "|".join(recoveries) if recoveries else "NONE_LEXICAL"
    for rec in recoveries:
        rec_counter[rec] += 1
    if not recoveries:
        rec_counter["UNRECOVERABLE_LEXICAL"] += 1

    analysis_rows.append({
        "error_type": "FN_BLOCKER", "category": cat, "s1_id": r["s1_id"], "cand_id": r["cand_id"],
        "source": r.get("source", ""), "country": s1.get("country", ""),
        "s1_name": s1_name, "s1_addr": s1_addr, "cand_name": cand_name, "cand_addr": cand_addr,
        "prob": 0.0, "rank": 999, "cosine_sim": 0.0,
        "recovery_strategy": rec_str
    })

print("Blocker Miss Categories:")
for k, v in fn_b_cats.most_common():
    print(f"  {k:35s}: {v:4d} ({v/len(fn_b_rows)*100:5.1f}%)")

print("\nRecovery Pathways for 3,066 Blocker Misses:")
for k, v in rec_counter.most_common():
    print(f"  {k:35s}: {v:4d} ({v/len(fn_b_rows)*100:5.1f}%)")

print("=== STEP 7: SAVING REPORTS ===", flush=True)

out_csv = os.path.join(REPORTS_DIR, "EXP_N008_ERROR_ANALYSIS.csv")
fieldnames = ["error_type", "category", "s1_id", "cand_id", "source", "country", "s1_name", "s1_addr", "cand_name", "cand_addr", "prob", "rank", "cosine_sim", "recovery_strategy"]

with open(out_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(analysis_rows)

print(f"Saved: {out_csv} ({len(analysis_rows):,} rows)", flush=True)

summary_md = os.path.join(REPORTS_DIR, "EXP_N008_ERROR_SUMMARY.md")
with open(summary_md, "w", encoding="utf-8") as f:
    f.write("# EXP_N008 Comprehensive Error Forensics & Blocker Recovery Report\n\n")
    f.write("## 1. High-Level Performance Metrics\n")
    f.write("- **Metric A (Decision Layer on Captured Candidates)**: Macro F0.5 = 0.986154\n")
    f.write("- **Metric B (Complete End-to-End System)**: Macro F0.5 = 0.972440\n")
    f.write("- **Target**: Complete System Macro F0.5 >= 0.9950 (Gap: +0.02256)\n")
    f.write("- **Total S1 Evaluated**: 25,000\n")
    f.write("- **Total True Ground Truth Matches**: 86,232\n")
    f.write("- **Total False Positives**: 699\n")
    f.write("- **Total False Negatives**: 4,854\n")
    f.write("  - Model missed from candidates: 1,788 (36.8% of FN)\n")
    f.write("  - Blocker never retrieved: 3,066 (63.2% of FN)\n\n")

    f.write("## 2. False Positive (FP) Breakdown (Total: 699)\n")
    for cat, count in fp_cats.most_common():
        f.write(f"- **{cat}**: {count:,} ({count/len(fp_rows)*100:.1f}%)\n")
    f.write("\n")

    f.write("## 3. False Negative — Model Miss Breakdown (Total: 1,788)\n")
    for cat, count in fn_m_cats.most_common():
        f.write(f"- **{cat}**: {count:,} ({count/len(fn_m_rows)*100:.1f}%)\n")
    f.write("\n")

    f.write("## 4. False Negative — Blocker Miss Breakdown (Total: 3,066)\n")
    for cat, count in fn_b_cats.most_common():
        f.write(f"- **{cat}**: {count:,} ({count/len(fn_b_rows)*100:.1f}%)\n")
    f.write("\n")

    f.write("## 5. Potential Recovery Pathways for 3,066 Blocker Misses\n")
    for rec, count in rec_counter.most_common():
        f.write(f"- **{rec}**: {count:,} ({count/len(fn_b_rows)*100:.1f}%)\n")
    f.write("\n")

print(f"Saved: {summary_md}", flush=True)
print("ALL ERROR ANALYSIS COMPLETE!", flush=True)
