#!/usr/bin/env python3
"""
Test Inference Pipeline for EXP_N039 Frozen Champion.
Generates:
  1. submission/matching_results.tsv (and output/matching_results.tsv)
  2. submission/candidate_pairs.tsv (and output/candidate_pairs.tsv)
  3. reports/FINAL_TEST_BLOCKER_AUDIT.json
  4. submission/FINAL_SUBMISSION_AUDIT.json
  5. submission/FINAL_SUBMISSION_MANIFEST.json
  6. submission/README.md

Zero-Leakage Invariants:
- Never accesses validation or test ground truth labels.
- Streams each country partition ('France', 'US', 'India') independently to guarantee sub-4GB RAM footprint.
- Applies frozen blocking rules:
    * Exact normalized business name
    * Sorted business name tokens
    * Address / postal token agreement filtering
    * Source-specific cardinality guards
- Writes format strictly compliant with ML Challenge 2026 specification:
    source1_entity_id\tmatched_entity_ids
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

# Ensure root paths
CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "dataset" / "test"
OUT_SUBMISSION_DIR = CODE_DIR / "submission"
OUT_ROOT_DIR = PROJECT_ROOT / "output"
REPORTS_DIR = CODE_DIR / "reports"

sys.path.insert(0, str(CODE_DIR / "src"))

from blocking import clean_business_name
from normalization import extract_numbers, tokenize


def get_file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def process_country_partition(
    country: str,
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
    matching_f,
    candidate_f,
    audit_dict: dict,
):
    print(f"\n{'='*70}\n[Processing Partition: {country}]\n{'='*70}", flush=True)
    t_start = time.time()

    # Step 1: Read all S1 records for this country
    print(f"[{country}] Step 1: Reading test S1 records...", flush=True)
    t0 = time.time()
    s1_order: List[str] = []
    s1_data: Dict[str, Tuple[str, str, Set[str], str]] = {}
    with open(s1_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            c = (row.get("country") or "").strip()
            if c != country:
                continue
            eid = row.get("entity_id", "").strip()
            nm = row.get("business_name", "") or ""
            addr = row.get("business_address", "") or ""
            norm_nm, toks = clean_business_name(nm)
            sort_toks = " ".join(sorted(toks)) if len(toks) >= 2 else norm_nm
            addr_toks = set(tokenize(addr.lower())) if addr else set()
            hns = extract_numbers(addr) if addr else []
            hn = hns[0] if hns else ""
            s1_order.append(eid)
            s1_data[eid] = (norm_nm, sort_toks, addr_toks, hn)

    n_s1 = len(s1_order)
    print(f"[{country}] Loaded {n_s1:,} S1 entities in {time.time()-t0:.2f}s", flush=True)

    # Step 2: Index S2 and S3 for this country
    print(f"[{country}] Step 2: Indexing S2 and S3 target records...", flush=True)
    t0 = time.time()
    norm_idx = defaultdict(list)
    sort_idx = defaultdict(list)
    target_addr: Dict[str, Tuple[Set[str], str]] = {}

    target_counts = {"S2": 0, "S3": 0}
    for tsv_path, src_lbl in [(s2_path, "S2"), (s3_path, "S3")]:
        t1 = time.time()
        with open(tsv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                c = (row.get("country") or "").strip()
                if c != country:
                    continue
                eid = row.get("entity_id", "").strip()
                nm = row.get("business_name", "") or ""
                addr = row.get("business_address", "") or ""
                norm_nm, toks = clean_business_name(nm)
                if norm_nm:
                    norm_idx[norm_nm].append(eid)
                if len(toks) >= 2:
                    sort_idx[" ".join(sorted(toks))].append(eid)
                addr_toks = set(tokenize(addr.lower())) if addr else set()
                hns = extract_numbers(addr) if addr else []
                hn = hns[0] if hns else ""
                target_addr[eid] = (addr_toks, hn)
                target_counts[src_lbl] += 1
        print(f"[{country}] Indexed {src_lbl} ({target_counts[src_lbl]:,} records) in {time.time()-t1:.2f}s", flush=True)

    print(f"[{country}] Target index built ({len(norm_idx):,} norm keys, {len(sort_idx):,} sort keys) in {time.time()-t0:.2f}s", flush=True)

    # Step 3: Candidate Generation & Matching
    print(f"[{country}] Step 3: Running Candidate Generation & Set Resolution...", flush=True)
    t0 = time.time()
    n_singletons = 0
    n_matched = 0
    total_candidates_partition = 0
    total_matches_partition = 0

    batch_size = 50000
    for idx, s1_id in enumerate(s1_order):
        norm_nm, sort_toks, a_toks, hn = s1_data[s1_id]

        cands_set = set()
        if norm_nm in norm_idx:
            cands_set.update(norm_idx[norm_nm])
        if sort_toks in sort_idx:
            cands_set.update(sort_idx[sort_toks])

        # Write candidates
        if cands_set:
            cands_list = sorted(cands_set)
            candidate_f.write(f"{s1_id}\t{','.join(cands_list)}\n")
            total_candidates_partition += len(cands_list)
        else:
            candidate_f.write(f"{s1_id}\t\n")

        # Rank and filter matches
        if not cands_set:
            matching_f.write(f"{s1_id}\t\n")
            n_singletons += 1
            continue

        scored_cands = []
        for cid in cands_set:
            ca_toks, chn = target_addr[cid]
            if a_toks and ca_toks:
                j = len(a_toks & ca_toks) / max(1, len(a_toks | ca_toks))
                hn_match = 1.0 if (hn and chn and hn == chn) else 0.0
                score = 0.5 * j + 0.5 * hn_match
            else:
                score = 0.5  # Uninformative address, rely on high-precision exact name match
            scored_cands.append((score, cid))

        scored_cands.sort(key=lambda x: -x[0])

        s2_selected = []
        s3_selected = []
        for sc, cid in scored_cands:
            # Drop low address overlap if other strong matches exist
            if sc < 0.20 and (s2_selected or s3_selected):
                continue
            if cid.startswith("S2-") and len(s2_selected) < 4:
                s2_selected.append(cid)
            elif cid.startswith("S3-") and len(s3_selected) < 4:
                s3_selected.append(cid)

        final_matches = s2_selected + s3_selected
        if final_matches:
            matching_f.write(f"{s1_id}\t{','.join(final_matches)}\n")
            n_matched += 1
            total_matches_partition += len(final_matches)
        else:
            matching_f.write(f"{s1_id}\t\n")
            n_singletons += 1

        if (idx + 1) % batch_size == 0 or (idx + 1) == n_s1:
            print(f"[{country}] Processed {idx+1:,} / {n_s1:,} entities ({time.time()-t0:.1f}s)...", flush=True)

    dt_part = time.time() - t_start
    print(f"[{country}] Completed in {dt_part:.2f}s | Singletons: {n_singletons:,} ({n_singletons/n_s1*100:.2f}%) | Matched: {n_matched:,} ({n_matched/n_s1*100:.2f}%)", flush=True)

    audit_dict[country] = {
        "s1_count": n_s1,
        "s2_targets": target_counts["S2"],
        "s3_targets": target_counts["S3"],
        "singletons": n_singletons,
        "matched_entities": n_matched,
        "total_matches": total_matches_partition,
        "mean_matches_per_entity": round(total_matches_partition / max(1, n_s1), 3),
        "total_candidates": total_candidates_partition,
        "mean_candidates_per_entity": round(total_candidates_partition / max(1, n_s1), 3),
        "elapsed_sec": round(dt_part, 2),
    }

    # Free memory
    del s1_order, s1_data, norm_idx, sort_idx, target_addr
    gc.collect()


def main():
    t_global = time.time()
    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026 — FINAL TEST INFERENCE PIPELINE")
    print("FROZEN CHAMPION: EXP_N039")
    print("=" * 80)

    s1_tsv = DATA_DIR / "test_source1.tsv"
    s2_tsv = DATA_DIR / "test_source2.tsv"
    s3_tsv = DATA_DIR / "test_source3.tsv"

    assert s1_tsv.exists(), f"Missing {s1_tsv}"
    assert s2_tsv.exists(), f"Missing {s2_tsv}"
    assert s3_tsv.exists(), f"Missing {s3_tsv}"

    OUT_SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    sub_matching_path = OUT_SUBMISSION_DIR / "matching_results.tsv"
    sub_candidate_path = OUT_SUBMISSION_DIR / "candidate_pairs.tsv"
    root_matching_path = OUT_ROOT_DIR / "matching_results.tsv"
    root_candidate_path = OUT_ROOT_DIR / "candidate_pairs.tsv"

    audit_stats = {"partitions": {}, "system": "EXP_N039_FROZEN_CHAMPION"}

    # Process all 3 partitions streaming directly to files
    with open(sub_matching_path, "w", encoding="utf-8", newline="\n") as f_match, \
         open(sub_candidate_path, "w", encoding="utf-8", newline="\n") as f_cand:

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for country in ["France", "US", "India"]:
            process_country_partition(
                country,
                s1_tsv,
                s2_tsv,
                s3_tsv,
                f_match,
                f_cand,
                audit_stats["partitions"],
            )

    print("\nCopying submission files to project root output/ directory...", flush=True)
    import shutil
    shutil.copyfile(sub_matching_path, root_matching_path)
    shutil.copyfile(sub_candidate_path, root_candidate_path)

    # Compute audits and hashes
    print("\nComputing SHA-256 digests and final audits...", flush=True)
    matching_sha = get_file_sha256(sub_matching_path)
    candidate_sha = get_file_sha256(sub_candidate_path)

    matching_size_bytes = os.path.getsize(sub_matching_path)
    candidate_size_bytes = os.path.getsize(sub_candidate_path)

    total_s1 = sum(p["s1_count"] for p in audit_stats["partitions"].values())
    total_singletons = sum(p["singletons"] for p in audit_stats["partitions"].values())
    total_matched = sum(p["matched_entities"] for p in audit_stats["partitions"].values())
    total_matches = sum(p["total_matches"] for p in audit_stats["partitions"].values())
    total_candidates = sum(p["total_candidates"] for p in audit_stats["partitions"].values())

    audit_stats["summary"] = {
        "total_test_s1_entities": total_s1,
        "singletons": total_singletons,
        "singleton_rate": round(total_singletons / total_s1, 4),
        "matched_entities": total_matched,
        "matched_rate": round(total_matched / total_s1, 4),
        "total_matches_predicted": total_matches,
        "mean_matches_per_s1": round(total_matches / total_s1, 3),
        "total_candidates": total_candidates,
        "mean_candidates_per_s1": round(total_candidates / total_s1, 3),
        "matching_results_sha256": matching_sha,
        "matching_results_bytes": matching_size_bytes,
        "candidate_pairs_sha256": candidate_sha,
        "candidate_pairs_bytes": candidate_size_bytes,
        "total_elapsed_sec": round(time.time() - t_global, 2),
    }

    # Save audits
    with open(REPORTS_DIR / "FINAL_TEST_BLOCKER_AUDIT.json", "w", encoding="utf-8") as f:
        json.dump(audit_stats, f, indent=2)

    with open(OUT_SUBMISSION_DIR / "FINAL_SUBMISSION_AUDIT.json", "w", encoding="utf-8") as f:
        json.dump(audit_stats, f, indent=2)

    manifest = {
        "challenge": "Amazon ML Challenge 2026",
        "task": "Business Entity Resolution",
        "frozen_champion": "EXP_N039",
        "champion_validation_macro_f05": 0.993320,
        "submission_files": [
            {
                "file": "matching_results.tsv",
                "sha256": matching_sha,
                "bytes": matching_size_bytes,
                "rows": total_s1,
            },
            {
                "file": "candidate_pairs.tsv",
                "sha256": candidate_sha,
                "bytes": candidate_size_bytes,
                "rows": total_s1,
            },
        ],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(OUT_SUBMISSION_DIR / "FINAL_SUBMISSION_MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Save submission README
    readme_text = f"""# ML Challenge 2026 — Final Submission Package (EXP_N039 Frozen Champion)

## 1. System Overview
- **Champion Experiment:** EXP_N039
- **Validation Complete-System Macro $F_{{0.5}}$:** **0.993320** (Macro Precision: 0.993207, Macro Recall: 0.994593, FP: 191, Total FN: 204)
- **Candidate Pool Recall Ceiling:** 99.9513%

## 2. Test Dataset Scale
- **Total Test S1 Entities:** {total_s1:,}
- **Singletons Identified:** {total_singletons:,} ({total_singletons/total_s1*100:.2f}%)
- **Entities with Matches:** {total_matched:,} ({total_matched/total_s1*100:.2f}%)
- **Total Cross-Source Matches:** {total_matches:,} (Average {total_matches/total_s1:.2f} matches/entity)
- **Total Candidate Pool Generated:** {total_candidates:,} (Average {total_candidates/total_s1:.2f} candidates/entity)

## 3. Cryptographic Verification (SHA-256)
- `matching_results.tsv`: `{matching_sha}` ({matching_size_bytes:,} bytes)
- `candidate_pairs.tsv`: `{candidate_sha}` ({candidate_size_bytes:,} bytes)
"""
    with open(OUT_SUBMISSION_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_text)

    print("\n" + "=" * 80)
    print("FINAL TEST INFERENCE COMPLETED SUCCESSFULLY")
    print(f"Total Test Entities Processed: {total_s1:,}")
    print(f"Matches Predicted: {total_matches:,} across {total_matched:,} entities")
    print(f"Singletons: {total_singletons:,}")
    print(f"matching_results.tsv SHA-256: {matching_sha}")
    print(f"Execution Time: {time.time()-t_global:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
