#!/usr/bin/env python3
"""
Ground Truth Inspector and Integrity Validator for Amazon ML Challenge 2026.
Analyzes the training ground truth and performs strict cross-source integrity checks.
"""

import os
import sys
import json
import time
from collections import Counter
import numpy as np

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def load_sorted_ids(path, prefix_len=3):
    """Load entity ID integer suffixes into a sorted numpy int64 array."""
    t0 = time.time()
    eids = []
    with open(path, "r", encoding="utf-8", errors="replace") as fp:
        next(fp)  # skip header
        for line in fp:
            eids.append(int(line.split("\t")[0][prefix_len:]))
    arr = np.array(eids, dtype=np.int64)
    arr.sort()
    dt = time.time() - t0
    return arr, dt


def load_s1_with_country(path):
    """Load S1 entity IDs and country codes into sorted numpy arrays."""
    t0 = time.time()
    eids = []
    countries = []
    country_map = {"US": 0, "India": 1}
    inv_map = {0: "US", 1: "India"}

    with open(path, "r", encoding="utf-8", errors="replace") as fp:
        next(fp)
        for line in fp:
            parts = line.rstrip("\r\n").split("\t")
            eids.append(int(parts[0][3:]))
            c = parts[3]
            if c not in country_map:
                country_map[c] = len(country_map)
                inv_map[country_map[c]] = c
            countries.append(country_map[c])

    eid_arr = np.array(eids, dtype=np.int64)
    c_arr = np.array(countries, dtype=np.int8)

    order = np.argsort(eid_arr)
    sorted_eids = eid_arr[order]
    sorted_countries = c_arr[order]
    dt = time.time() - t0
    return sorted_eids, sorted_countries, inv_map, dt


def inspect_ground_truth(dataset_dir, reports_dir):
    """Perform ground truth analysis and integrity verification."""
    print("=" * 60)
    print("PHASE 5 & 6: GROUND TRUTH INSPECTION & DATA INTEGRITY")
    print("=" * 60)

    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")

    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    test_s1_path = os.path.join(test_dir, "test_source1.tsv")

    print("\n[1/4] Indexing Source 1, Source 2, and Source 3 entities...")
    s1_ids, s1_countries, inv_country_map, t_s1 = load_s1_with_country(s1_path)
    print(f"  Loaded {len(s1_ids):,} Source 1 records in {t_s1:.2f}s")

    s2_ids, t_s2 = load_sorted_ids(s2_path)
    print(f"  Loaded {len(s2_ids):,} Source 2 records in {t_s2:.2f}s")

    s3_ids, t_s3 = load_sorted_ids(s3_path)
    print(f"  Loaded {len(s3_ids):,} Source 3 records in {t_s3:.2f}s")

    print("\n[2/4] Reading and analyzing train_ground_truth.tsv...")
    t_start = time.time()

    total_gt_rows = 0
    singletons = 0
    match_counts = Counter()
    max_matches = 0
    max_s1_id = None

    s2_only_count = 0
    s3_only_count = 0
    both_count = 0

    total_s2_matches = 0
    total_s3_matches = 0

    # Country-level tracking
    country_stats = {
        name: {
            "total_s1": 0,
            "singletons": 0,
            "match_counts": Counter(),
            "s2_only": 0,
            "s3_only": 0,
            "both": 0,
            "total_s2_matches": 0,
            "total_s3_matches": 0,
            "max_matches": 0,
        }
        for name in inv_country_map.values()
    }

    # Integrity trackers
    missing_s1_in_source = 0
    missing_s2_in_source = 0
    missing_s3_in_source = 0
    self_matches = 0
    intra_list_duplicates = 0
    unknown_prefixes = 0
    malformed_gt_rows = 0

    with open(gt_path, "r", encoding="utf-8", errors="replace") as fp:
        header = next(fp).rstrip("\r\n").split("\t")
        expected_cols = ["source1_entity_id", "matched_entity_ids"]
        unexpected_cols = header != expected_cols

        for line_num, line in enumerate(fp, start=2):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 2:
                malformed_gt_rows += 1
                continue

            total_gt_rows += 1
            s1_id_str, match_str = parts

            # Lookup S1 ID in train_source1
            s1_num = int(s1_id_str[3:])
            pos = np.searchsorted(s1_ids, s1_num)
            if pos >= len(s1_ids) or s1_ids[pos] != s1_num:
                missing_s1_in_source += 1
                country_name = "Unknown"
            else:
                c_idx = s1_countries[pos]
                country_name = inv_country_map[c_idx]

            c_dict = country_stats.get(country_name)
            if c_dict is not None:
                c_dict["total_s1"] += 1

            if not match_str:
                # Singleton
                singletons += 1
                match_counts[0] += 1
                if c_dict is not None:
                    c_dict["singletons"] += 1
                    c_dict["match_counts"][0] += 1
                continue

            # Non-singleton
            mids = match_str.split(",")
            k = len(mids)
            match_counts[k] += 1
            if k > max_matches:
                max_matches = k
                max_s1_id = s1_id_str

            if c_dict is not None:
                c_dict["match_counts"][k] += 1
                if k > c_dict["max_matches"]:
                    c_dict["max_matches"] = k

            # Check intra-list duplicates
            if len(mids) != len(set(mids)):
                intra_list_duplicates += 1

            has_s2 = False
            has_s3 = False

            for mid in mids:
                if mid.startswith("S1-"):
                    self_matches += 1
                elif mid.startswith("S2-"):
                    has_s2 = True
                    total_s2_matches += 1
                    if c_dict is not None:
                        c_dict["total_s2_matches"] += 1
                    mid_num = int(mid[3:])
                    p2 = np.searchsorted(s2_ids, mid_num)
                    if p2 >= len(s2_ids) or s2_ids[p2] != mid_num:
                        missing_s2_in_source += 1
                elif mid.startswith("S3-"):
                    has_s3 = True
                    total_s3_matches += 1
                    if c_dict is not None:
                        c_dict["total_s3_matches"] += 1
                    mid_num = int(mid[3:])
                    p3 = np.searchsorted(s3_ids, mid_num)
                    if p3 >= len(s3_ids) or s3_ids[p3] != mid_num:
                        missing_s3_in_source += 1
                else:
                    unknown_prefixes += 1

            if has_s2 and has_s3:
                both_count += 1
                if c_dict is not None:
                    c_dict["both"] += 1
            elif has_s2:
                s2_only_count += 1
                if c_dict is not None:
                    c_dict["s2_only"] += 1
            elif has_s3:
                s3_only_count += 1
                if c_dict is not None:
                    c_dict["s3_only"] += 1

    gt_elapsed = time.time() - t_start

    # Verify test_source1 requirement
    test_s1_count = 0
    with open(test_s1_path, "r", encoding="utf-8") as fp:
        next(fp)
        for _ in fp:
            test_s1_count += 1

    # Compute breakdown percentages
    pct_singleton = round((singletons / total_gt_rows) * 100, 2)
    pct_1_match = round((match_counts[1] / total_gt_rows) * 100, 2)
    pct_2_match = round((match_counts[2] / total_gt_rows) * 100, 2)
    pct_3_match = round((match_counts[3] / total_gt_rows) * 100, 2)
    count_4_plus = sum(match_counts[k] for k in match_counts if k >= 4)
    pct_4_plus = round((count_4_plus / total_gt_rows) * 100, 2)

    pct_s2_only = round((s2_only_count / total_gt_rows) * 100, 2)
    pct_s3_only = round((s3_only_count / total_gt_rows) * 100, 2)
    pct_both = round((both_count / total_gt_rows) * 100, 2)

    print("\n[3/4] Ground Truth Key Metrics:")
    print(f"  Total S1 Entities:     {total_gt_rows:,}")
    print(f"  Singleton Percentage:  {pct_singleton}% ({singletons:,} entities)")
    print(f"  1-Match Percentage:    {pct_1_match}% ({match_counts[1]:,} entities)")
    print(f"  2-Match Percentage:    {pct_2_match}% ({match_counts[2]:,} entities)")
    print(f"  3-Match Percentage:    {pct_3_match}% ({match_counts[3]:,} entities)")
    print(f"  4+ Match Percentage:   {pct_4_plus}% ({count_4_plus:,} entities)")
    print(f"  S2-Only Entities:      {pct_s2_only}% ({s2_only_count:,} entities)")
    print(f"  S3-Only Entities:      {pct_s3_only}% ({s3_only_count:,} entities)")
    print(f"  Both S2 & S3 Entities: {pct_both}% ({both_count:,} entities)")
    print(f"  Maximum Match Count:   {max_matches} (Entity: {max_s1_id})")

    print("\n[4/4] Data Integrity Audit:")
    print(f"  Every GT S1 ID in train_source1:     {'YES (0 missing)' if missing_s1_in_source == 0 else f'NO ({missing_s1_in_source} missing)'}")
    print(f"  Every matched S2 ID in train_source2: {'YES (0 missing)' if missing_s2_in_source == 0 else f'NO ({missing_s2_in_source} missing)'}")
    print(f"  Every matched S3 ID in train_source3: {'YES (0 missing)' if missing_s3_in_source == 0 else f'NO ({missing_s3_in_source} missing)'}")
    print(f"  Any S1 self-matches in ground truth:  {'NONE (0)' if self_matches == 0 else f'FOUND ({self_matches})'}")
    print(f"  Any intra-list duplicate IDs:        {'NONE (0)' if intra_list_duplicates == 0 else f'FOUND ({intra_list_duplicates})'}")
    print(f"  Any malformed GT rows:               {'NONE (0)' if malformed_gt_rows == 0 else f'FOUND ({malformed_gt_rows})'}")
    print(f"  Unexpected columns in GT header:     {'NONE' if not unexpected_cols else f'FOUND ({header})'}")
    print(f"  Test S1 entities to predict:         {test_s1_count:,} (All must appear in submission)")

    # Country details
    formatted_country_stats = {}
    for cname, cdata in country_stats.items():
        tot = cdata["total_s1"]
        c_pct_singleton = round((cdata["singletons"] / tot) * 100, 2) if tot > 0 else 0
        c_pct_1 = round((cdata["match_counts"][1] / tot) * 100, 2) if tot > 0 else 0
        c_pct_2 = round((cdata["match_counts"][2] / tot) * 100, 2) if tot > 0 else 0
        c_pct_3 = round((cdata["match_counts"][3] / tot) * 100, 2) if tot > 0 else 0
        c_4_plus = sum(cdata["match_counts"][k] for k in cdata["match_counts"] if k >= 4)
        c_pct_4_plus = round((c_4_plus / tot) * 100, 2) if tot > 0 else 0
        tot_matches = cdata["total_s2_matches"] + cdata["total_s3_matches"]
        avg_matches = round(tot_matches / tot, 2) if tot > 0 else 0

        formatted_country_stats[cname] = {
            "total_s1_entities": tot,
            "singletons": cdata["singletons"],
            "singleton_percentage": c_pct_singleton,
            "1_match_count": cdata["match_counts"][1],
            "1_match_percentage": c_pct_1,
            "2_match_count": cdata["match_counts"][2],
            "2_match_percentage": c_pct_2,
            "3_match_count": cdata["match_counts"][3],
            "3_match_percentage": c_pct_3,
            "4_plus_match_count": c_4_plus,
            "4_plus_match_percentage": c_pct_4_plus,
            "s2_only_count": cdata["s2_only"],
            "s2_only_percentage": round((cdata["s2_only"] / tot) * 100, 2) if tot > 0 else 0,
            "s3_only_count": cdata["s3_only"],
            "s3_only_percentage": round((cdata["s3_only"] / tot) * 100, 2) if tot > 0 else 0,
            "both_count": cdata["both"],
            "both_percentage": round((cdata["both"] / tot) * 100, 2) if tot > 0 else 0,
            "total_s2_matches": cdata["total_s2_matches"],
            "total_s3_matches": cdata["total_s3_matches"],
            "total_matches": tot_matches,
            "average_matches_per_entity": avg_matches,
            "max_matches": cdata["max_matches"],
        }

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_source1_entities": total_gt_rows,
        "singletons": {
            "count": singletons,
            "percentage": pct_singleton,
        },
        "match_cardinality": {
            "1_match": {"count": match_counts[1], "percentage": pct_1_match},
            "2_match": {"count": match_counts[2], "percentage": pct_2_match},
            "3_match": {"count": match_counts[3], "percentage": pct_3_match},
            "4_plus_match": {"count": count_4_plus, "percentage": pct_4_plus},
        },
        "source_coverage": {
            "s2_only": {"count": s2_only_count, "percentage": pct_s2_only},
            "s3_only": {"count": s3_only_count, "percentage": pct_s3_only},
            "both_s2_and_s3": {"count": both_count, "percentage": pct_both},
        },
        "total_matched_ids": total_s2_matches + total_s3_matches,
        "s2_matches": total_s2_matches,
        "s3_matches": total_s3_matches,
        "max_matches": max_matches,
        "max_matches_entity_id": max_s1_id,
        "country_level_statistics": formatted_country_stats,
        "integrity_audit": {
            "missing_s1_in_train_source1": missing_s1_in_source,
            "missing_s2_in_train_source2": missing_s2_in_source,
            "missing_s3_in_train_source3": missing_s3_in_source,
            "self_matches": self_matches,
            "intra_list_duplicates": intra_list_duplicates,
            "unknown_prefixes": unknown_prefixes,
            "malformed_gt_rows": malformed_gt_rows,
            "unexpected_columns": unexpected_cols,
            "test_source1_entities_required": test_s1_count,
        },
    }

    report_file = os.path.join(reports_dir, "ground_truth_report.json")
    with open(report_file, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)

    print(f"\nGround truth report saved to: {report_file}")
    return results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    dataset_dir = os.path.join(project_root, "dataset")
    reports_dir = os.path.abspath(os.path.join(script_dir, "..", "reports"))

    inspect_ground_truth(dataset_dir, reports_dir)


if __name__ == "__main__":
    main()
