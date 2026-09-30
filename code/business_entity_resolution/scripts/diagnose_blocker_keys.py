import os
import sys
import time
import csv
import re
import unicodedata
from collections import defaultdict, Counter

print("=== DIAGNOSING BLOCKER MISSES ===", flush=True)

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.abspath(os.path.join(CODE_DIR, "..", ".."))
REPORTS_DIR = os.path.join(CODE_DIR, "reports")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "dataset", "train")
SRC_DIR = os.path.join(CODE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from blocking import extract_blocking_keys, clean_business_name
from build_pair_dataset import UNION_E_KEYS

fn_b_csv = os.path.join(REPORTS_DIR, "exp_n008_errors_fn_blocker.csv")
fn_b_rows = []
with open(fn_b_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        fn_b_rows.append(r)

print(f"Total Blocker Misses: {len(fn_b_rows):,}")

# Take first 50 blocker misses
sample_pairs = fn_b_rows[:50]
s1_needed = {r["s1_id"] for r in sample_pairs}
s2_needed = {r["cand_id"] for r in sample_pairs if r["cand_id"].startswith("S2")}
s3_needed = {r["cand_id"] for r in sample_pairs if r["cand_id"].startswith("S3")}

def load_recs(path, ids):
    res = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row.get("entity_id")
            if eid in ids:
                res[eid] = {
                    "name": row.get("business_name") or row.get("name") or "",
                    "address": row.get("business_address") or row.get("address") or "",
                    "country": row.get("country", "")
                }
                if len(res) == len(ids):
                    break
    return res

s1_data = load_recs(os.path.join(TRAIN_DIR, "train_source1.tsv"), s1_needed)
s2_data = load_recs(os.path.join(TRAIN_DIR, "train_source2.tsv"), s2_needed)
s3_data = load_recs(os.path.join(TRAIN_DIR, "train_source3.tsv"), s3_needed)
cand_data = {**s2_data, **s3_data}

print("\n--- Inspecting First 10 Missed Pairs Keys ---")
for i, pair in enumerate(sample_pairs[:10]):
    s1_id = pair["s1_id"]
    cid = pair["cand_id"]
    s1 = s1_data.get(s1_id, {})
    cand = cand_data.get(cid, {})

    keys_s1 = set(extract_blocking_keys(s1.get("name", ""), s1.get("address", ""), s1.get("country", ""), UNION_E_KEYS))
    keys_cand = set(extract_blocking_keys(cand.get("name", ""), cand.get("address", ""), cand.get("country", ""), UNION_E_KEYS))

    shared = keys_s1 & keys_cand
    print(f"\n[{i+1}] S1: {s1_id} ('{s1.get('name')}', '{s1.get('address')}')")
    print(f"    Cand: {cid} ('{cand.get('name')}', '{cand.get('address')}')")
    print(f"    Keys S1: {len(keys_s1)}, Keys Cand: {len(keys_cand)}, Shared in Union E: {len(shared)}")
    if shared:
        print(f"    SHARED KEYS FOUND (so why was it missed?): {list(shared)[:3]}")
    else:
        print(f"    Sample S1 keys: {list(keys_s1)[:2]}")
        print(f"    Sample Cand keys: {list(keys_cand)[:2]}")
