#!/usr/bin/env python3
"""
EXP_N017 — Step 3: Run Inference with Fine-Tuned XLM-RoBERTa Cross-Encoder.
Scores the 90,468 candidate pairs from val_scoring_manifest.npz and saves
probabilities to experiments/EXP_N017/probs_val_cross_encoder.npy.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONUNBUFFERED"] = "1"
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

CODE_DIR = Path(__file__).resolve().parents[1]
EXP_DIR = CODE_DIR / "experiments"
EXP_N017_DIR = EXP_DIR / "EXP_N017"

BATCH_SIZE = 64
MAX_LENGTH = 96
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EvalDataset(Dataset):
    def __init__(self, s1_ids, cand_ids, id_to_text):
        self.s1_ids = s1_ids
        self.cand_ids = cand_ids
        self.id_to_text = id_to_text

    def __len__(self):
        return len(self.s1_ids)

    def __getitem__(self, idx):
        s1 = self.id_to_text.get(self.s1_ids[idx], "")
        cand = self.id_to_text.get(self.cand_ids[idx], "")
        return s1, cand


def collate_fn(batch, tokenizer):
    texts_a = [item[0] for item in batch]
    texts_b = [item[1] for item in batch]
    enc = tokenizer(
        texts_a,
        texts_b,
        max_length=MAX_LENGTH,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return enc


def main():
    t_start = time.time()
    print("=" * 75)
    print("[EXP_N017] Scoring Validation Candidates with XLM-RoBERTa Cross-Encoder")
    print("=" * 75)

    # 1. Load formatted records
    store = np.load(EXP_N017_DIR / "formatted_record_store.npz")
    eids = store["eids"].astype(str)
    texts = store["texts"].astype(str)
    id_to_text = {eids[i]: texts[i] for i in range(len(eids))}
    print(f"Loaded {len(id_to_text):,} formatted records.")

    # 2. Load val scoring manifest
    val_man = np.load(EXP_N017_DIR / "val_scoring_manifest.npz")
    s1_ids = val_man["s1_ids"].astype(str)
    cand_ids = val_man["cand_ids"].astype(str)
    val_pair_indices = val_man["val_pair_indices"]
    N_pairs = len(s1_ids)
    print(f"Scoring {N_pairs:,} candidate pairs...")

    # 3. Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=1)
    checkpoint_path = EXP_N017_DIR / "xlm_roberta_cross_encoder.pt"
    print(f"Loading checkpoint from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.half().to(DEVICE)
    model.eval()

    dataset = EvalDataset(s1_ids, cand_ids, id_to_text)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    all_probs = []
    print("\nStarting batched inference on GPU...", flush=True)
    with torch.no_grad():
        for enc in tqdm(loader, desc="Inference", ncols=100):
            input_ids = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            all_probs.append(probs)

    scores = np.concatenate(all_probs)
    print(f"Inference finished in {time.time()-t_start:.1f}s. Score range: {scores.min():.4f} to {scores.max():.4f}")

    # Map back to full 2,000,000 validation array
    va = np.load(EXP_DIR / "val_pairs_n010.npz", allow_pickle=True)
    full_probs = np.zeros(len(va["y"]), dtype=np.float32)
    full_probs[val_pair_indices] = scores

    out_path = EXP_N017_DIR / "probs_val_cross_encoder.npy"
    np.save(out_path, full_probs)
    print(f"Saved complete 2M array to: {out_path}")


if __name__ == "__main__":
    main()
