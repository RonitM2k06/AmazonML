import os, sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

for tsv in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]:
    with open(os.path.join("../../dataset/train", tsv), encoding="utf-8", errors="replace") as f:
        n = sum(1 for _ in f) - 1
    print(f"{tsv}: {n:,} records")

tr = np.load("experiments/train_pairs_exp008.npz", allow_pickle=True)
train_s1 = set(tr["s1_ids"].astype(str))
print(f"Unique train S1: {len(train_s1):,}")
print(f"Train positives: {int(tr['y'].sum()):,}")
print(f"Train negatives: {int((tr['y']==0).sum()):,}")

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)} GB")

# Count GT positive pairs total
pos_pairs = 0
s1_with_matches = 0
with open("../../dataset/train/train_ground_truth.tsv", encoding="utf-8", errors="replace") as f:
    f.readline()
    for line in f:
        parts = line.rstrip("\r\n").split("\t")
        if len(parts) > 1 and parts[1].strip():
            mids = [m for m in parts[1].split(",") if m.strip()]
            if mids:
                pos_pairs += len(mids)
                s1_with_matches += 1

print(f"GT positive pairs total: {pos_pairs:,}")
print(f"S1 entities with at least one match: {s1_with_matches:,}")
