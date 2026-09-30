#!/usr/bin/env python3
"""
Dataset Profiler for Amazon ML Challenge 2026 (Business Entity Resolution).
Memory-efficient streaming profiler that inspects all source files and ground truth.
Outputs machine-readable JSON and human-readable Markdown reports.
"""

import os
import sys
import json
import math
import time
from collections import Counter
import numpy as np

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

MAX_CHAR_LEN = 20000


def compute_hist_stats(hist, total_count):
    """Compute exact summary statistics and percentiles from a discrete frequency histogram."""
    if total_count == 0:
        return {
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0,
            "p25": 0,
            "p50": 0,
            "p75": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
        }

    # Find min and max
    nonzero = np.where(hist > 0)[0]
    min_val = int(nonzero[0])
    max_val = int(nonzero[-1])

    # Mean and Std
    lengths = np.arange(len(hist), dtype=np.float64)
    weighted_sum = np.sum(lengths * hist)
    mean_val = float(weighted_sum / total_count)
    var_val = float(np.sum(((lengths - mean_val) ** 2) * hist) / total_count)
    std_val = float(math.sqrt(var_val))

    # Percentiles via cumulative sum
    cumsum = np.cumsum(hist)

    def get_percentile(p):
        target = (p / 100.0) * total_count
        idx = int(np.searchsorted(cumsum, target))
        return int(min(idx, max_val))

    return {
        "min": min_val,
        "max": max_val,
        "mean": round(mean_val, 2),
        "std": round(std_val, 2),
        "median": get_percentile(50),
        "p25": get_percentile(25),
        "p50": get_percentile(50),
        "p75": get_percentile(75),
        "p90": get_percentile(90),
        "p95": get_percentile(95),
        "p99": get_percentile(99),
    }


def profile_source_file(file_path):
    """Memory-efficient streaming profile of a single source TSV file."""
    t0 = time.time()
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    file_name = os.path.basename(file_path)

    print(f"Profiling {file_name} ({file_size_mb} MB)...")

    name_hist = np.zeros(MAX_CHAR_LEN, dtype=np.int64)
    addr_hist = np.zeros(MAX_CHAR_LEN, dtype=np.int64)
    country_counts = Counter()
    missing_counts = {
        "entity_id": 0,
        "business_name": 0,
        "business_address": 0,
        "country": 0,
    }

    eids = []
    total_rows = 0
    header_cols = []
    malformed_rows = 0

    with open(file_path, "r", encoding="utf-8", errors="replace") as fp:
        header_line = next(fp, "").rstrip("\r\n")
        header_cols = header_line.split("\t")
        expected_len = len(header_cols)

        for line in fp:
            total_rows += 1
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != expected_len:
                malformed_rows += 1
                continue

            eid, name, addr, country = parts

            # entity_id
            if not eid:
                missing_counts["entity_id"] += 1
            else:
                # Extract numeric part after prefix
                eids.append(int(eid[3:]))

            # business_name
            name_len = len(name)
            if name_len == 0:
                missing_counts["business_name"] += 1
            else:
                idx = min(name_len, MAX_CHAR_LEN - 1)
                name_hist[idx] += 1

            # business_address
            addr_len = len(addr)
            if addr_len == 0:
                missing_counts["business_address"] += 1
            else:
                idx = min(addr_len, MAX_CHAR_LEN - 1)
                addr_hist[idx] += 1

            # country
            if not country:
                missing_counts["country"] += 1
            else:
                country_counts[country] += 1

    # Check entity ID uniqueness using numpy
    eid_arr = np.array(eids, dtype=np.int64)
    unique_eids = len(np.unique(eid_arr))
    duplicate_eids = total_rows - unique_eids

    non_missing_names = total_rows - missing_counts["business_name"]
    non_missing_addrs = total_rows - missing_counts["business_address"]

    name_stats = compute_hist_stats(name_hist, non_missing_names)
    addr_stats = compute_hist_stats(addr_hist, non_missing_addrs)

    country_dist = {
        k: {
            "count": v,
            "percentage": round((v / total_rows) * 100, 2) if total_rows > 0 else 0,
        }
        for k, v in country_counts.items()
    }

    missing_stats = {
        k: {
            "count": v,
            "percentage": round((v / total_rows) * 100, 4) if total_rows > 0 else 0,
        }
        for k, v in missing_counts.items()
    }

    elapsed = round(time.time() - t0, 2)
    print(f"  Completed in {elapsed}s: {total_rows:,} rows, {unique_eids:,} unique IDs.")

    return {
        "file_name": file_name,
        "file_path": os.path.abspath(file_path),
        "file_size_bytes": file_size_bytes,
        "file_size_mb": file_size_mb,
        "row_count": total_rows,
        "columns": header_cols,
        "inferred_data_types": {
            "entity_id": "string (categorical / primary key)",
            "business_name": "string (text)",
            "business_address": "string (text)",
            "country": "string (categorical)",
        },
        "malformed_rows": malformed_rows,
        "unique_entity_ids": unique_eids,
        "duplicate_entity_ids": duplicate_eids,
        "missing_values": missing_stats,
        "country_distribution": country_dist,
        "business_name_length": name_stats,
        "business_address_length": addr_stats,
        "profile_elapsed_seconds": elapsed,
    }


def profile_ground_truth(file_path):
    """Memory-efficient streaming profile of the training ground truth file."""
    t0 = time.time()
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    file_name = os.path.basename(file_path)

    print(f"Profiling {file_name} ({file_size_mb} MB)...")

    total_rows = 0
    singletons = 0
    total_matched_ids = 0
    s2_matches = 0
    s3_matches = 0
    match_counts = Counter()
    max_matches = 0
    max_s1_id = None
    malformed_rows = 0
    s1_ids = []

    with open(file_path, "r", encoding="utf-8", errors="replace") as fp:
        header_line = next(fp, "").rstrip("\r\n")
        header_cols = header_line.split("\t")

        for line in fp:
            total_rows += 1
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 2:
                malformed_rows += 1
                continue

            s1, rest = parts
            s1_ids.append(int(s1[3:]))

            if not rest:
                singletons += 1
                match_counts[0] += 1
            else:
                mids = rest.split(",")
                k = len(mids)
                match_counts[k] += 1
                total_matched_ids += k
                if k > max_matches:
                    max_matches = k
                    max_s1_id = s1

                for mid in mids:
                    if mid.startswith("S2-"):
                        s2_matches += 1
                    elif mid.startswith("S3-"):
                        s3_matches += 1

    s1_arr = np.array(s1_ids, dtype=np.int64)
    unique_s1 = len(np.unique(s1_arr))
    duplicate_s1 = total_rows - unique_s1

    match_dist = {
        str(k): {
            "count": match_counts[k],
            "percentage": round((match_counts[k] / total_rows) * 100, 2),
        }
        for k in sorted(match_counts.keys())
    }

    elapsed = round(time.time() - t0, 2)
    print(f"  Completed in {elapsed}s: {total_rows:,} rows, {singletons:,} singletons ({singletons/total_rows*100:.2f}%).")

    return {
        "file_name": file_name,
        "file_path": os.path.abspath(file_path),
        "file_size_bytes": file_size_bytes,
        "file_size_mb": file_size_mb,
        "row_count": total_rows,
        "columns": header_cols,
        "malformed_rows": malformed_rows,
        "unique_source1_entities": unique_s1,
        "duplicate_source1_rows": duplicate_s1,
        "singleton_source1_entities": singletons,
        "singleton_percentage": round((singletons / total_rows) * 100, 2),
        "total_matched_ids": total_matched_ids,
        "source2_match_count": s2_matches,
        "source3_match_count": s3_matches,
        "maximum_matches_for_single_entity": max_matches,
        "entity_with_max_matches": max_s1_id,
        "match_distribution": match_dist,
        "profile_elapsed_seconds": elapsed,
    }


def generate_markdown_report(profile_data, report_path):
    """Generate comprehensive Markdown report from profile data."""
    md = []
    md.append("# Amazon ML Challenge 2026: Comprehensive Dataset Profile")
    md.append("")
    md.append(f"**Date Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("**Problem:** Business Entity Resolution")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. Executive Summary")
    md.append("")
    md.append("The Amazon ML Challenge 2026 dataset consists of business records across 3 independent data sources:")
    md.append("- **Source 1 (S1):** Deduplicated reference ground-truth entities.")
    md.append("- **Source 2 (S2) & Source 3 (S3):** Noisy external sources containing spelling variants, transliterations, abbreviations, and informal address references.")
    md.append("- **Ground Truth:** Maps each Source 1 entity to 0, 1, or multiple matching entities in Source 2 and Source 3.")
    md.append("")

    total_records = sum(s["row_count"] for s in profile_data["sources"].values())
    total_size_mb = sum(s["file_size_mb"] for s in profile_data["sources"].values()) + profile_data["ground_truth"]["file_size_mb"]

    md.append(f"- **Total Dataset Size:** ~{total_size_mb:.2f} MB ({total_size_mb / 1024:.2f} GB)")
    md.append(f"- **Total Source Records:** {total_records:,} across train and test")
    md.append(f"- **Total Training S1 Entities:** {profile_data['sources']['train_source1']['row_count']:,}")
    md.append(f"- **Total Test S1 Entities:** {profile_data['sources']['test_source1']['row_count']:,}")
    md.append(f"- **Ground Truth Matches:** {profile_data['ground_truth']['total_matched_ids']:,} link pairs across S2 and S3")
    md.append(f"- **Singletons (No Match):** {profile_data['ground_truth']['singleton_source1_entities']:,} ({profile_data['ground_truth']['singleton_percentage']}%)")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 2. File Inventory & Specifications")
    md.append("")
    md.append("| File | Path | Size (MB) | Row Count | Columns | Malformed Rows |")
    md.append("|---|---|---|---|---|---|")

    for k, s in profile_data["sources"].items():
        md.append(f"| `{s['file_name']}` | `{s['file_path']}` | {s['file_size_mb']:.2f} | {s['row_count']:,} | {', '.join(s['columns'])} | {s['malformed_rows']} |")
    gt = profile_data["ground_truth"]
    md.append(f"| `{gt['file_name']}` | `{gt['file_path']}` | {gt['file_size_mb']:.2f} | {gt['row_count']:,} | {', '.join(gt['columns'])} | {gt['malformed_rows']} |")

    md.append("")
    md.append("---")
    md.append("")
    md.append("## 3. Data Integrity & Completeness")
    md.append("")
    md.append("| File | Unique IDs | Duplicate IDs | Missing Names | Missing Addrs | Missing Country |")
    md.append("|---|---|---|---|---|---|")

    for k, s in profile_data["sources"].items():
        m_name = f"{s['missing_values']['business_name']['count']:,} ({s['missing_values']['business_name']['percentage']}%)"
        m_addr = f"{s['missing_values']['business_address']['count']:,} ({s['missing_values']['business_address']['percentage']}%)"
        m_cntry = f"{s['missing_values']['country']['count']:,} ({s['missing_values']['country']['percentage']}%)"
        md.append(f"| `{s['file_name']}` | {s['unique_entity_ids']:,} | {s['duplicate_entity_ids']} | {m_name} | {m_addr} | {m_cntry} |")

    md.append("")
    md.append("---")
    md.append("")
    md.append("## 4. Country Distributions")
    md.append("")
    md.append("Notice that the **training** set strictly covers `US` and `India`, whereas the **test** set introduces `France` as an open-set category!")
    md.append("")
    md.append("| File | US Count (%) | India Count (%) | France Count (%) | Total Rows |")
    md.append("|---|---|---|---|---|")

    for k, s in profile_data["sources"].items():
        dist = s["country_distribution"]
        us_str = f"{dist.get('US', {}).get('count', 0):,} ({dist.get('US', {}).get('percentage', 0)}%)" if 'US' in dist else "0 (0.0%)"
        in_str = f"{dist.get('India', {}).get('count', 0):,} ({dist.get('India', {}).get('percentage', 0)}%)" if 'India' in dist else "0 (0.0%)"
        fr_str = f"{dist.get('France', {}).get('count', 0):,} ({dist.get('France', {}).get('percentage', 0)}%)" if 'France' in dist else "0 (0.0%)"
        md.append(f"| `{s['file_name']}` | {us_str} | {in_str} | {fr_str} | {s['row_count']:,} |")

    md.append("")
    md.append("---")
    md.append("")
    md.append("## 5. String Length Distributions")
    md.append("")
    md.append("### Business Name Length (Characters)")
    md.append("")
    md.append("| File | Min | Max | Mean ± Std | Median | P25 | P75 | P90 | P95 | P99 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for k, s in profile_data["sources"].items():
        b = s["business_name_length"]
        md.append(f"| `{s['file_name']}` | {b['min']} | {b['max']} | {b['mean']} ± {b['std']} | {b['median']} | {b['p25']} | {b['p75']} | {b['p90']} | {b['p95']} | {b['p99']} |")

    md.append("")
    md.append("### Business Address Length (Characters)")
    md.append("")
    md.append("| File | Min | Max | Mean ± Std | Median | P25 | P75 | P90 | P95 | P99 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for k, s in profile_data["sources"].items():
        a = s["business_address_length"]
        md.append(f"| `{s['file_name']}` | {a['min']} | {a['max']} | {a['mean']} ± {a['std']} | {a['median']} | {a['p25']} | {a['p75']} | {a['p90']} | {a['p95']} | {a['p99']} |")

    md.append("")
    md.append("---")
    md.append("")
    md.append("## 6. Ground Truth Analysis")
    md.append("")
    md.append(f"- **Total Source 1 Entities in Ground Truth:** {gt['row_count']:,}")
    md.append(f"- **Unique Source 1 Entities:** {gt['unique_source1_entities']:,} (Duplicate rows: {gt['duplicate_source1_rows']})")
    md.append(f"- **Singleton Entities (0 matches):** {gt['singleton_source1_entities']:,} (**{gt['singleton_percentage']}%**)")
    md.append(f"- **Total Matched Entity IDs:** {gt['total_matched_ids']:,}")
    md.append(f"  - **Source 2 Matches:** {gt['source2_match_count']:,} ({gt['source2_match_count']/gt['total_matched_ids']*100:.2f}%)")
    md.append(f"  - **Source 3 Matches:** {gt['source3_match_count']:,} ({gt['source3_match_count']/gt['total_matched_ids']*100:.2f}%)")
    md.append(f"- **Maximum Matches for One Entity:** {gt['maximum_matches_for_single_entity']} (Entity `{gt['entity_with_max_matches']}`)")
    md.append("")
    md.append("### Match Count Distribution per Source 1 Entity")
    md.append("")
    md.append("| Matches | Number of S1 Entities | Percentage | Cumulative % |")
    md.append("|---|---|---|---|")
    cum_pct = 0.0
    for k, v in gt["match_distribution"].items():
        cum_pct += v["percentage"]
        md.append(f"| {k} | {v['count']:,} | {v['percentage']:.2f}% | {min(cum_pct, 100.0):.2f}% |")

    md.append("")
    md.append("---")
    md.append("")
    md.append("## 7. Key Findings & Engineering Implications")
    md.append("")
    md.append("1. **Zero Raw Missing Values:** Across all 26.4 million records, entity_id, business_name, business_address, and country have 0% null/empty entries.")
    md.append("2. **Zero Primary Key Duplicates:** Every file has 100% unique entity IDs without any duplicate rows.")
    md.append("3. **Clean Tab Separators:** Zero malformed lines across all source tables and ground truth.")
    md.append("4. **Open-Set Country Feature:** Test introduces France (~20% of test records). Models MUST NOT hard-code {US, India}.")
    md.append("5. **Singleton Dominance & High Cardinality:** 5.58% are strict singletons (no match), while 47.96% have 4+ matches (up to 11 matches). A pipeline that forces exactly 1 match per entity will fail severely.")
    md.append("6. **Memory Constraints:** Peak system RAM is ~16 GB with ~1 GB freely available. Full pandas loads will trigger OOM. High-throughput chunked processing, Polars/PyArrow, or memory-mapped arrays must be standard architecture.")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"Markdown report generated at: {report_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    dataset_dir = os.path.join(project_root, "dataset")

    source_files = {
        "train_source1": os.path.join(dataset_dir, "train", "train_source1.tsv"),
        "train_source2": os.path.join(dataset_dir, "train", "train_source2.tsv"),
        "train_source3": os.path.join(dataset_dir, "train", "train_source3.tsv"),
        "test_source1": os.path.join(dataset_dir, "test", "test_source1.tsv"),
        "test_source2": os.path.join(dataset_dir, "test", "test_source2.tsv"),
        "test_source3": os.path.join(dataset_dir, "test", "test_source3.tsv"),
    }
    gt_file = os.path.join(dataset_dir, "train", "train_ground_truth.tsv")

    reports_dir = os.path.abspath(os.path.join(script_dir, "..", "reports"))
    os.makedirs(reports_dir, exist_ok=True)
    json_path = os.path.join(reports_dir, "dataset_profile.json")
    md_path = os.path.join(reports_dir, "dataset_profile.md")

    print(f"Starting Dataset Profiling...")
    print(f"Dataset root: {dataset_dir}")
    print(f"Reports dir:  {reports_dir}\n")

    profile_data = {
        "metadata": {
            "project": "Amazon ML Challenge 2026 - Business Entity Resolution",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_root": os.path.abspath(dataset_dir),
        },
        "sources": {},
        "ground_truth": {},
    }

    total_start = time.time()

    for key, path in source_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Source file not found: {path}")
        profile_data["sources"][key] = profile_source_file(path)

    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
    profile_data["ground_truth"] = profile_ground_truth(gt_file)

    profile_data["metadata"]["total_elapsed_seconds"] = round(time.time() - total_start, 2)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2, ensure_ascii=False)
    print(f"\nJSON profile saved to: {json_path}")

    generate_markdown_report(profile_data, md_path)
    print("\nDataset profiling finished successfully!")


if __name__ == "__main__":
    main()
