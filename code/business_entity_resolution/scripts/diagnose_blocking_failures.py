#!/usr/bin/env python3
"""
Blocking Failure Diagnostics & Missed Pair Forensics.
Extracts all missed true pairs from the 25k validation benchmark under Union D,
classifies the root cause of every failure, checks bucket pruning impacts,
and saves missed_true_pairs.csv and missed_true_pair_analysis.md.
"""

import os
import sys
import csv
import json
import time
import math
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from normalization import (
    normalize_business_name,
    normalize_address,
    extract_postal_codes,
    extract_numbers,
    tokenize,
    token_jaccard,
    token_overlap,
    character_similarity,
    token_set_similarity,
    token_sort_similarity,
)
from blocking import BlockIndex, extract_blocking_keys
from evaluate_blocking import load_validation_sample


def diagnose_failures(dataset_dir, validation_ids_path, reports_dir):
    print("=" * 65)
    print("PHASE 4 — STEP 1, 2, 3: BLOCKING FAILURE FORENSICS")
    print("=" * 65)

    # 1. Load 25k validation sample
    s1_records, target_pool, s1_matches = load_validation_sample(
        dataset_dir, validation_ids_path, sample_size=25000, seed=42
    )

    union_d_keys = {
        "exact_norm", "sorted_tokens", "postal_name", "address_name",
        "char_ngram", "transliteration", "address_first"
    }

    print("\n[Step 1] Building Standard Union D Index (max_bucket=1000, max_cands=200)...")
    idx_std = BlockIndex(max_bucket_size=1000)
    target_keys_map = {}
    target_rec_map = {}

    for eid, name, addr, country in target_pool:
        target_rec_map[eid] = (name, addr, country)
        ks = extract_blocking_keys(name, addr, country, union_d_keys)
        target_keys_map[eid] = set(ks)
        idx_std.add(eid, ks)

    print("Querying Standard Union D Candidates...")
    cands_std = {}
    s1_keys_map = {}
    for s1_id, (eid, name, addr, country) in s1_records.items():
        ks = extract_blocking_keys(name, addr, country, union_d_keys)
        s1_keys_map[s1_id] = set(ks)
        cands_std[s1_id] = idx_std.query(ks, max_candidates=200)

    # Track standard recovered vs missed
    std_recovered = set()
    all_true_pairs = []
    for s1_id, mids in s1_matches.items():
        for mid in mids:
            all_true_pairs.append((s1_id, mid))
            if mid in cands_std.get(s1_id, set()):
                std_recovered.add((s1_id, mid))

    missed_pairs = [p for p in all_true_pairs if p not in std_recovered]
    total_true = len(all_true_pairs)
    print(f"Total True Pairs:     {total_true:,}")
    print(f"Standard Recovered:   {len(std_recovered):,} ({len(std_recovered)/total_true*100:.2f}%)")
    print(f"Standard Missed:      {len(missed_pairs):,} ({len(missed_pairs)/total_true*100:.2f}%)")

    # [Step 3] Run experiment with Unconstrained Pruning
    print("\n[Step 3] Testing Pruning Impact (max_bucket=50000, max_cands=5000)...")
    idx_unconstrained = BlockIndex(max_bucket_size=50000)
    for eid, (name, addr, country) in target_rec_map.items():
        idx_unconstrained.add(eid, list(target_keys_map[eid]))

    unconstrained_recovered = set()
    pruning_recovered_pairs = set()

    for s1_id, (eid, name, addr, country) in s1_records.items():
        ks = list(s1_keys_map[s1_id])
        cands_uncon = idx_unconstrained.query(ks, max_candidates=5000)
        for mid in s1_matches.get(s1_id, []):
            if mid in cands_uncon:
                unconstrained_recovered.add((s1_id, mid))
                if (s1_id, mid) not in std_recovered:
                    pruning_recovered_pairs.add((s1_id, mid))

    print(f"Unconstrained Recovered: {len(unconstrained_recovered):,} ({len(unconstrained_recovered)/total_true*100:.2f}%)")
    print(f"Recall lost strictly to pruning/caps: +{len(pruning_recovered_pairs):,} pairs (+{len(pruning_recovered_pairs)/total_true*100:.2f}%)")

    # Detailed Forensics on Missed Pairs
    print("\n[Step 2] Classifying and Extracting Missed True Pairs...")
    missed_rows = []
    failure_categories = Counter()
    pruning_keys_counter = Counter()

    for s1_id, mid in missed_pairs:
        s1_name, s1_addr, s1_country = s1_records[s1_id][1], s1_records[s1_id][2], s1_records[s1_id][3]
        m_name, m_addr, m_country = target_rec_map.get(mid, ("", "", s1_country))

        # Similarities
        name_sim = character_similarity(s1_name, m_name)
        addr_sim = character_similarity(s1_addr, m_addr) if (s1_addr and m_addr) else 0.0

        t1_name = tokenize(normalize_business_name(s1_name, strip_legal=True))
        t2_name = tokenize(normalize_business_name(m_name, strip_legal=True))
        name_jacc = token_jaccard(t1_name, t2_name)
        name_over = token_overlap(t1_name, t2_name)
        name_set_sim = token_set_similarity(s1_name, m_name)
        name_sort_sim = token_sort_similarity(s1_name, m_name)

        t1_addr = tokenize(normalize_address(s1_addr)) if s1_addr else []
        t2_addr = tokenize(normalize_address(m_addr)) if m_addr else []
        addr_over = token_overlap(t1_addr, t2_addr)

        num1 = extract_numbers(s1_addr)
        num2 = extract_numbers(m_addr)
        num_over = token_overlap(num1, num2) if (num1 and num2) else 0.0

        p1 = extract_postal_codes(s1_addr)
        p2 = extract_postal_codes(m_addr)
        postal_over = token_overlap(p1, p2) if (p1 and p2) else 0.0

        has_missing_addr = (not s1_addr or not m_addr)
        has_devanagari = bool(re.search(r"[\u0900-\u097F]", s1_name) or re.search(r"[\u0900-\u097F]", m_name))
        has_accents = bool(any(0x0300 <= ord(c) <= 0x036F for c in s1_name + m_name))

        # Check shared keys between S1 and target
        shared_keys = s1_keys_map[s1_id] & target_keys_map[mid]
        pruned_shared_keys = [k for k in shared_keys if k in idx_std.oversized_keys]

        # Automatic Root Cause Classification (Hierarchy)
        category = "Unknown"

        if (s1_id, mid) in pruning_recovered_pairs:
            category = "O_bucket_or_candidate_pruning"
            for pk in pruned_shared_keys:
                pruning_keys_counter[str(pk[:2])] += 1
        elif has_devanagari:
            category = "G_transliteration_script_mismatch"
        elif has_missing_addr:
            category = "H_missing_address_on_target"
        elif name_sim >= 0.80 and addr_sim < 0.40:
            category = "A_high_name_sim_but_divergent_address"
        elif addr_sim >= 0.80 and name_sim < 0.50:
            category = "B_high_addr_sim_but_divergent_name"
        elif name_over == 1.0 and name_jacc < 1.0:
            category = "E_token_insertion_deletion"
        elif name_sort_sim >= 0.95 and name_sim < 0.85:
            category = "F_word_order_transposition"
        elif len(p1) > 0 and len(p2) == 0:
            category = "I_missing_postal_code_on_target"
        elif len(num1) > 0 and len(num2) == 0:
            category = "J_missing_house_number_on_target"
        elif name_sim >= 0.60 and addr_sim >= 0.50:
            category = "C_moderate_name_and_address_similarity"
        elif len(s1_name) <= 6 or len(m_name) <= 6:
            category = "L_very_short_name_collision"
        elif has_accents:
            category = "M_accent_or_unicode_variation"
        else:
            category = "D_name_abbreviation_or_acronym"

        failure_categories[category] += 1

        missed_rows.append({
            "s1_id": s1_id,
            "matched_id": mid,
            "country": s1_country,
            "s1_name": s1_name,
            "target_name": m_name,
            "s1_address": s1_addr,
            "target_address": m_addr,
            "failure_category": category,
            "name_sim": round(name_sim, 3),
            "addr_sim": round(addr_sim, 3),
            "name_jaccard": round(name_jacc, 3),
            "name_token_overlap": round(name_over, 3),
            "name_set_sim": round(name_set_sim, 3),
            "addr_token_overlap": round(addr_over, 3),
            "numeric_overlap": round(num_over, 3),
            "postal_overlap": round(postal_over, 3),
            "has_missing_addr": int(has_missing_addr),
            "has_devanagari": int(has_devanagari),
            "shared_keys_count": len(shared_keys),
            "pruned_keys_count": len(pruned_shared_keys),
        })

    # Save CSV
    csv_path = os.path.join(reports_dir, "missed_true_pairs.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = list(missed_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(missed_rows)
    print(f"\nSaved {len(missed_rows):,} missed true pairs to {csv_path}")

    # Generate Markdown Report
    md_path = os.path.join(reports_dir, "missed_true_pair_analysis.md")
    write_forensics_markdown(
        md_path, total_true, len(std_recovered), len(missed_pairs),
        len(pruning_recovered_pairs), failure_categories, pruning_keys_counter, missed_rows
    )
    print(f"Saved failure analysis report to {md_path}")

    return missed_rows, failure_categories, len(pruning_recovered_pairs)


def write_forensics_markdown(
    md_path, total_true, n_recovered, n_missed, n_pruned, failure_categories, pruning_keys, missed_rows
):
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Blocking Failure Forensics & Root-Cause Classification\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write(f"- **Total True Match Pairs Evaluated:** {total_true:,}\n")
        f.write(f"- **Recovered by Union D (Standard):** {n_recovered:,} (**{n_recovered/total_true*100:.2f}%**)\n")
        f.write(f"- **Missed True Pairs:** {n_missed:,} (**{n_missed/total_true*100:.2f}%**)\n")
        f.write(f"- **Misses Caused by Pruning / Candidate Caps:** {n_pruned:,} (**{n_pruned/total_true*100:.2f}% of all true matches**)\n\n")

        f.write("## 2. Root-Cause Failure Classification\n\n")
        f.write("| Category | Description | Missed Pairs | % of Missed | % of Ground Truth |\n")
        f.write("|---|---|---|---|---|\n")

        descriptions = {
            "O_bucket_or_candidate_pruning": "Shared a blocking key, but bucket size exceeded cap (pruned as oversized)",
            "A_high_name_sim_but_divergent_address": "Name similarity ≥ 0.80, but address differs or is partial (no address block)",
            "H_missing_address_on_target": "Target record has empty address field; address blocks cannot fire",
            "C_moderate_name_and_address_similarity": "Both name and address moderately similar (name ~0.60–0.75, address ~0.50)",
            "E_token_insertion_deletion": "One name is a proper subset of the other with added/omitted descriptor",
            "D_name_abbreviation_or_acronym": "Severe abbreviation, acronym, or initials without common token",
            "B_high_addr_sim_but_divergent_name": "Address similarity ≥ 0.80, but business trade name differs significantly",
            "G_transliteration_script_mismatch": "Devanagari vs Latin script mismatch in India partition",
            "F_word_order_transposition": "Identical tokens in different syntactic order",
            "I_missing_postal_code_on_target": "S1 has PIN/ZIP code, but target address lacks postal code",
            "J_missing_house_number_on_target": "S1 has house number, but target has street name only",
            "L_very_short_name_collision": "Short name (≤ 5 chars) with high ambiguity",
            "M_accent_or_unicode_variation": "Accented or Unicode characters not normalized",
            "Unknown": "Unclassified residual failures",
        }

        for cat, cnt in failure_categories.most_common():
            pct_miss = round(cnt / n_missed * 100, 2)
            pct_gt = round(cnt / total_true * 100, 2)
            desc = descriptions.get(cat, cat)
            f.write(f"| `{cat}` | {desc} | **{cnt:,}** | **{pct_miss}%** | {pct_gt}% |\n")

        f.write("\n## 3. Pruning Impact Analysis (Step 3)\n\n")
        f.write(f"When `max_bucket_size` was raised from 1,000 to 50,000 and `max_candidates` was increased to 5,000:\n\n")
        f.write(f"- Exactly **{n_pruned:,} previously missed pairs were recovered** (+{n_pruned/total_true*100:.2f}% recall).\n")
        f.write("- **Keys most responsible for pruned true matches:**\n")
        for k, cnt in pruning_keys.most_common(10):
            f.write(f"  - `{k}`: {cnt:,} lost true matches\n")

        f.write("\n## 4. Concrete Examples of Top Failure Modes\n\n")
        by_cat = defaultdict(list)
        for r in missed_rows:
            if len(by_cat[r["failure_category"]]) < 3:
                by_cat[r["failure_category"]].append(r)

        for cat in ["A_high_name_sim_but_divergent_address", "H_missing_address_on_target", "C_moderate_name_and_address_similarity", "D_name_abbreviation_or_acronym"]:
            f.write(f"### Mode: `{cat}`\n\n")
            for i, ex in enumerate(by_cat.get(cat, []), 1):
                f.write(f"**Example {i} ({ex['country']}):**\n")
                f.write(f"- S1: `{ex['s1_id']}` | **{ex['s1_name']}** | {ex['s1_address']}\n")
                f.write(f"- Match: `{ex['matched_id']}` | **{ex['target_name']}** | {ex['target_address']}\n")
                f.write(f"- Metrics: Name Sim = `{ex['name_sim']}`, Addr Sim = `{ex['addr_sim']}`, Name Overlap = `{ex['name_token_overlap']}`\n\n")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    dataset_dir = os.path.join(project_root, "dataset")
    validation_ids_path = os.path.join(project_root, "code", "business_entity_resolution", "experiments", "validation_s1_ids.txt")
    reports_dir = os.path.join(project_root, "code", "business_entity_resolution", "reports")

    diagnose_failures(dataset_dir, validation_ids_path, reports_dir)


if __name__ == "__main__":
    main()
