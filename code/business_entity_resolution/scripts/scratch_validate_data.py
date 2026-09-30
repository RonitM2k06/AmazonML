import sys, time, csv
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

EXP_DIR  = Path("experiments")
DATA_DIR = Path("../../dataset/train")

t0 = time.time()
_tr = np.load(str(EXP_DIR / "train_pairs_exp008.npz"), allow_pickle=True)
labels    = _tr["y"].astype(int)
s1_arr    = _tr["s1_ids"].astype(str)
cand_arr  = _tr["cand_ids"].astype(str)
needed_s1 = set(s1_arr.tolist())
pos_mask  = labels == 1
print(f"NPZ: {len(labels):,} pairs, {pos_mask.sum():,} positives  ({time.time()-t0:.1f}s)")

# Check GT
with open(DATA_DIR / "train_ground_truth.tsv", encoding="utf-8", errors="replace") as f:
    r = csv.DictReader(f, delimiter="\t")
    for row in r:
        s1 = row.get("source1_entity_id", "")
        ms = row.get("matched_entity_ids", "")
        if s1 in needed_s1:
            print(f"GT: {s1} -> {ms[:80]}")
            break

# Check first S1 record
with open(DATA_DIR / "train_source1.tsv", encoding="utf-8", errors="replace") as f:
    r = csv.DictReader(f, delimiter="\t")
    found = 0
    for row in r:
        eid = row.get("entity_id", "")
        if eid in needed_s1:
            nm = row.get("business_name", "")
            cn = row.get("country", "")
            print(f"S1: {eid} name={nm[:50]!r} country={cn!r}")
            found += 1
            if found >= 2:
                break

print(f"All OK  ({time.time()-t0:.1f}s total)")
