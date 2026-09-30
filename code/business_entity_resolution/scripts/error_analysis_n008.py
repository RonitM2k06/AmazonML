#!/usr/bin/env python3
"""
Error Analysis for EXP_N008.
Analyzes:
1. False Positives (699)
2. False Negatives from Model (1,788)
3. False Negatives from Blocker (3,066)
Generates: reports/EXP_N008_ERROR_ANALYSIS.csv and reports/EXP_N008_ERROR_SUMMARY.md
"""

import os
import sys
import time
import csv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONUNBUFFERED"] = "1"
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

print("Starting error_analysis_n008.py...", flush=True)

from collections import Counter, defaultdict
import pandas as pd
import numpy as np

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(CODE_DIR, "..", ".."))
SRC_DIR = os.path.join(CODE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from normalization import normalize_name, normalize_address, tokenize
from blocking import transliterate_indic, get_consonant_skeleton

REPORTS_DIR = os.path.join(CODE_DIR, "reports")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "dataset", "train")

def load_record_map(tsv_path, ids_needed):
    """Load records for only requested entity IDs."""
    data = {}
    if not os.path.exists(tsv_path):
        print(f"Warning: {tsv_path} not found")
        return data
    print(f"Reading {os.path.basename(tsv_path)} for {len(ids_needed):,} IDs...")
    with open(tsv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row.get("entity_id")
            if eid in ids_needed:
                data[eid] = row
    return data

def main():
    print("=" * 70)
    print("EXP_N008 Targeted Error Analysis")
    print("=" * 70)

    fp_csv = os.path.join(REPORTS_DIR, "exp_n008_errors_fp.csv")
    fn_m_csv = os.path.join(REPORTS_DIR, "exp_n008_errors_fn_model.csv")
    fn_b_csv = os.path.join(REPORTS_DIR, "exp_n008_errors_fn_blocker.csv")

    df_fp = pd.read_csv(fp_csv) if os.path.exists(fp_csv) else pd.DataFrame()
    df_fn_m = pd.read_csv(fn_m_csv) if os.path.exists(fn_m_csv) else pd.DataFrame()
    df_fn_b = pd.read_csv(fn_b_csv) if os.path.exists(fn_b_csv) else pd.DataFrame()

    print(f"Loaded errors: FP={len(df_fp):,}, FN_model={len(df_fn_m):,}, FN_blocker={len(df_fn_b):,}")

    # Gather entity IDs
    s1_ids = set(df_fp["s1_id"]).union(df_fn_m["s1_id"]).union(df_fn_b["s1_id"])
    s2_ids = set()
    s3_ids = set()

    for df in [df_fp, df_fn_m, df_fn_b]:
        if not df.empty and "cand_id" in df.columns:
            for cid in df["cand_id"]:
                if str(cid).startswith("S2"):
                    s2_ids.add(cid)
                elif str(cid).startswith("S3"):
                    s3_ids.add(cid)

    print(f"Entities needed: S1={len(s1_ids):,}, S2={len(s2_ids):,}, S3={len(s3_ids):,}")

    s1_map = load_record_map(os.path.join(TRAIN_DIR, "train_source1.tsv"), s1_ids)
    s2_map = load_record_map(os.path.join(TRAIN_DIR, "train_source2.tsv"), s2_ids)
    s3_map = load_record_map(os.path.join(TRAIN_DIR, "train_source3.tsv"), s3_ids)

    cand_map = {**s2_map, **s3_map}

    analysis_rows = []

    # 1. Analyze False Positives (699)
    print("\nCategorizing False Positives...")
    fp_cats = Counter()
    for _, row in df_fp.iterrows():
        s1 = s1_map.get(row["s1_id"], {})
        cand = cand_map.get(row["cand_id"], {})
        s1_name = str(s1.get("name", ""))
        cand_name = str(cand.get("name", ""))
        s1_addr = str(s1.get("address", ""))
        cand_addr = str(cand.get("address", ""))

        n_s1 = normalize_name(s1_name)
        n_cand = normalize_name(cand_name)
        t_s1 = set(tokenize(n_s1))
        t_cand = set(tokenize(n_cand))
        overlap = len(t_s1 & t_cand) / max(1, min(len(t_s1), len(t_cand)))

        addr_jacc = row.get("addr_token_jaccard", 0.0)
        prob = row.get("prob", 0.0)

        # Categorize
        if n_s1 == n_cand and addr_jacc < 0.2:
            cat = "A_same_name_diff_biz"
        elif addr_jacc > 0.6 and overlap < 0.3:
            cat = "B_same_addr_diff_biz"
        elif min(len(t_s1), len(t_cand)) <= 1 and overlap == 1.0:
            cat = "C_short_name_collision"
        elif "branch" in s1_addr.lower() or "branch" in cand_addr.lower() or overlap > 0.8:
            cat = "F_branch_or_similar_chain"
        elif prob < 0.965:
            cat = "H_marginal_threshold_error"
        else:
            cat = "I_other_semantic_confusion"

        fp_cats[cat] += 1
        analysis_rows.append({
            "error_type": "FP",
            "category": cat,
            "s1_id": row["s1_id"],
            "cand_id": row["cand_id"],
            "source": row.get("source", ""),
            "country": s1.get("country", ""),
            "s1_name": s1_name,
            "s1_addr": s1_addr,
            "cand_name": cand_name,
            "cand_addr": cand_addr,
            "prob": prob,
            "rank": row.get("rank", 0),
            "cosine_sim": row.get("cosine_sim", 0.0),
            "recovery_strategy": ""
        })

    # 2. Analyze False Negatives from Model (1,788)
    print("Categorizing Model False Negatives...")
    fn_m_cats = Counter()
    for _, row in df_fn_m.iterrows():
        s1 = s1_map.get(row["s1_id"], {})
        cand = cand_map.get(row["cand_id"], {})
        s1_name = str(s1.get("name", ""))
        cand_name = str(cand.get("name", ""))
        s1_addr = str(s1.get("address", ""))
        cand_addr = str(cand.get("address", ""))

        prob = row.get("prob", 0.0)
        top_minus_prob = row.get("top_minus_prob", 0.0)
        s1_addr_missing = row.get("is_s1_addr_missing", 0.0)
        cand_addr_missing = row.get("is_cand_addr_missing", 0.0)

        if prob < 0.50:
            cat = "B_low_model_score"
        elif prob >= 0.956 and top_minus_prob > 0.025:
            cat = "C_margin_pruned"
        elif s1_addr_missing or cand_addr_missing:
            cat = "G_missing_address_penalty"
        elif row.get("addr_token_jaccard", 0.0) < 0.1:
            cat = "F_address_divergence"
        elif row.get("name_token_jaccard", 0.0) < 0.3:
            cat = "E_name_variation"
        else:
            cat = "D_bridge_subthreshold"

        fn_m_cats[cat] += 1
        analysis_rows.append({
            "error_type": "FN_MODEL",
            "category": cat,
            "s1_id": row["s1_id"],
            "cand_id": row["cand_id"],
            "source": row.get("source", ""),
            "country": s1.get("country", ""),
            "s1_name": s1_name,
            "s1_addr": s1_addr,
            "cand_name": cand_name,
            "cand_addr": cand_addr,
            "prob": prob,
            "rank": row.get("rank", 0),
            "cosine_sim": row.get("cosine_sim", 0.0),
            "recovery_strategy": ""
        })

    # 3. Analyze False Negatives from Blocker (3,066)
    print("Categorizing Blocker Misses and Evaluating Potential Recoveries...")
    fn_b_cats = Counter()
    rec_counter = Counter()

    for _, row in df_fn_b.iterrows():
        s1 = s1_map.get(row["s1_id"], {})
        cand = cand_map.get(row["cand_id"], {})
        s1_name = str(s1.get("name", ""))
        cand_name = str(cand.get("name", ""))
        s1_addr = str(s1.get("address", ""))
        cand_addr = str(cand.get("address", ""))

        n_s1 = normalize_name(s1_name)
        n_cand = normalize_name(cand_name)
        t_s1 = tokenize(n_s1)
        t_cand = tokenize(n_cand)

        # Recovery checks
        recoverable_by = []
        if n_s1 and n_s1 == n_cand:
            recoverable_by.append("exact_norm")
        if set(t_s1) & set(t_cand):
            recoverable_by.append("single_token_unconstrained")
        
        # Check phonetic / transliteration skeleton
        skel_s1 = [get_consonant_skeleton(t) for t in t_s1 if len(t) >= 3]
        skel_cand = [get_consonant_skeleton(t) for t in t_cand if len(t) >= 3]
        if set(skel_s1) & set(skel_cand):
            recoverable_by.append("single_phonetic_skeleton")

        # Check address token overlap
        t_a1 = set(tokenize(normalize_address(s1_addr)))
        t_a2 = set(tokenize(normalize_address(cand_addr)))
        addr_overlap = t_a1 & t_a2 - {"india", "road", "street", "nagar", "near", "opposite", "floor"}
        if len(addr_overlap) >= 2:
            recoverable_by.append("loose_address_overlap")
        elif len(addr_overlap) == 1:
            recoverable_by.append("single_address_token")

        # Categorize cause
        if not s1_name or not cand_name:
            cat = "A_missing_name"
        elif any(ord(c) > 127 for c in s1_name) or any(ord(c) > 127 for c in cand_name):
            cat = "B_non_latin_script"
        elif len(set(t_s1) & set(t_cand)) == 0:
            if len(addr_overlap) >= 1:
                cat = "C_zero_name_overlap_address_match"
            else:
                cat = "D_severe_corruption_no_surface_overlap"
        elif len(t_s1) <= 1 or len(t_cand) <= 1:
            cat = "E_short_name_unmatched"
        else:
            cat = "F_complex_variant"

        fn_b_cats[cat] += 1
        rec_str = "|".join(recoverable_by) if recoverable_by else "NONE"
        for r in recoverable_by:
            rec_counter[r] += 1
        if not recoverable_by:
            rec_counter["UNRECOVERABLE_LEXICAL"] += 1

        analysis_rows.append({
            "error_type": "FN_BLOCKER",
            "category": cat,
            "s1_id": row["s1_id"],
            "cand_id": row["cand_id"],
            "source": row.get("source", ""),
            "country": s1.get("country", ""),
            "s1_name": s1_name,
            "s1_addr": s1_addr,
            "cand_name": cand_name,
            "cand_addr": cand_addr,
            "prob": 0.0,
            "rank": 999,
            "cosine_sim": 0.0,
            "recovery_strategy": rec_str
        })

    # Save detailed CSV
    out_csv = os.path.join(REPORTS_DIR, "EXP_N008_ERROR_ANALYSIS.csv")
    pd.DataFrame(analysis_rows).to_csv(out_csv, index=False)
    print(f"\nDetailed analysis saved to: {out_csv}")

    # Generate Markdown Summary
    summary_md = os.path.join(REPORTS_DIR, "EXP_N008_ERROR_SUMMARY.md")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# EXP_N008 Comprehensive Error Forensics & Blocking Recovery Plan\n\n")
        f.write("## 1. High-Level Metrics Context\n")
        f.write("- **Metric A (Decision-layer on captured candidates)**: Macro F0.5 = 0.986154\n")
        f.write("- **Metric B (Complete System including Blocker Misses)**: Macro F0.5 = 0.972440\n")
        f.write("- **Total S1 Evaluated**: 25,000\n")
        f.write("- **Ground Truth Match Pairs**: 86,232\n")
        f.write("- **Total False Positives**: 699\n")
        f.write("- **Total False Negatives**: 4,854 (Model: 1,788 | Blocker: 3,066)\n\n")

        f.write("## 2. False Positive (FP) Breakdown (Total: 699)\n")
        for cat, count in fp_cats.most_common():
            f.write(f"- **{cat}**: {count:,} ({count/len(df_fp)*100:.1f}%)\n")
        f.write("\n")

        f.write("## 3. False Negative — Model Level (Total: 1,788)\n")
        for cat, count in fn_m_cats.most_common():
            f.write(f"- **{cat}**: {count:,} ({count/len(df_fn_m)*100:.1f}%)\n")
        f.write("\n")

        f.write("## 4. False Negative — Blocker Misses (Total: 3,066)\n")
        for cat, count in fn_b_cats.most_common():
            f.write(f"- **{cat}**: {count:,} ({count/len(df_fn_b)*100:.1f}%)\n")
        f.write("\n")

        f.write("## 5. Blocker Recovery Potential (3,066 Misses)\n")
        for rec, count in rec_counter.most_common():
            f.write(f"- **{rec}**: {count:,} pairs ({count/len(df_fn_b)*100:.1f}%)\n")
        f.write("\n")

    print(f"Summary saved to: {summary_md}")
    print("\n=== TOP BLOCKER RECOVERY STRATEGIES ===")
    for rec, count in rec_counter.most_common():
        print(f"  {rec:30s}: {count:5,d} ({count/len(df_fn_b)*100:5.1f}%)")

if __name__ == "__main__":
    main()
