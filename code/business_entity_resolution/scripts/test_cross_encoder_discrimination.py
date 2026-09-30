#!/usr/bin/env python3
"""
Test Cross-Encoder Discriminative Power on Borderline False Negatives and False Positives.
"""

import sys
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "scripts")
from authoritative_eval_fn import _load_val_s1_ids, _load_gt

print("Loading Cross-Encoder on GPU...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
tok = AutoTokenizer.from_pretrained(model_name)
ce_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
ce_model.eval()

# Load validation pairs and probabilities
va = np.load("experiments/val_pairs_n010.npz", allow_pickle=True)
pl = np.load("experiments/probs_n011_optimal_lgb.npy")
px = np.load("experiments/probs_n011_optimal_xgb.npy")
probs = 0.45 * pl + 0.55 * px

s1_ids = va["s1_ids"].astype(str)
c_ids = va["cand_ids"].astype(str)
y = va["y"]

# Borderline True Matches (0.85 <= p < 0.94, y == 1)
fn_mask = (probs >= 0.85) & (probs < 0.94) & (y == 1)
# False Positives from ANN pool (probs >= 0.90, y == 0)
fp_mask = (probs >= 0.90) & (y == 0)

fn_indices = np.where(fn_mask)[0][:30]
fp_indices = np.where(fp_mask)[0][:30]

print(f"Sampled {len(fn_indices)} Borderline True Positives and {len(fp_indices)} Hard False Positives.")

# Load raw text
needed_ids = set(s1_ids[fn_indices]) | set(c_ids[fn_indices]) | set(s1_ids[fp_indices]) | set(c_ids[fp_indices])
recs = {}

with open("../../dataset/train/train_source1.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        if p[0] in needed_ids:
            recs[p[0]] = (p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "")

for sf in ["train_source2.tsv", "train_source3.tsv"]:
    with open(f"../../dataset/train/{sf}", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in needed_ids:
                recs[p[0]] = (p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "")

def format_text(rec):
    return f"{rec[0]} {rec[1]}".strip()

# Run cross-encoder on FN sample
fn_pairs_a = [format_text(recs.get(s1_ids[i], ("", ""))) for i in fn_indices]
fn_pairs_b = [format_text(recs.get(c_ids[i], ("", ""))) for i in fn_indices]

fp_pairs_a = [format_text(recs.get(s1_ids[i], ("", ""))) for i in fp_indices]
fp_pairs_b = [format_text(recs.get(c_ids[i], ("", ""))) for i in fp_indices]

with torch.no_grad():
    inp_fn = tok(fn_pairs_a, fn_pairs_b, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    fn_scores = ce_model(**inp_fn).logits.squeeze().cpu().numpy()

    inp_fp = tok(fp_pairs_a, fp_pairs_b, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    fp_scores = ce_model(**inp_fp).logits.squeeze().cpu().numpy()

print("\n" + "=" * 60)
print("CROSS-ENCODER SCORE DISTRIBUTION")
print("=" * 60)
print(f"Borderline True Matches (y=1): Mean = {fn_scores.mean():.3f}, Median = {np.median(fn_scores):.3f}, Min = {fn_scores.min():.3f}, Max = {fn_scores.max():.3f}")
print(f"Hard False Positives   (y=0): Mean = {fp_scores.mean():.3f}, Median = {np.median(fp_scores):.3f}, Min = {fp_scores.min():.3f}, Max = {fp_scores.max():.3f}")
print(f"Score Separation Gap: {fn_scores.mean() - fp_scores.mean():.3f} logit points!")

# Convert to probabilities with sigmoid
p_fn = 1.0 / (1.0 + np.exp(-fn_scores))
p_fp = 1.0 / (1.0 + np.exp(-fp_scores))
print(f"\nSigmoid Probabilities:")
print(f"  True Positives  p >= 0.70: {(p_fn >= 0.70).sum()} / {len(p_fn)} ({(p_fn >= 0.70).sum()/len(p_fn)*100:.1f}%)")
print(f"  False Positives p <  0.30: {(p_fp < 0.30).sum()} / {len(p_fp)} ({(p_fp < 0.30).sum()/len(p_fp)*100:.1f}%)")
print("=" * 60)
