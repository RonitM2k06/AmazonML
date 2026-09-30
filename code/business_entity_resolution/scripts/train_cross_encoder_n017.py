#!/usr/bin/env python3
"""
EXP_N017 — Step 2: Fine-Tune Multilingual XLM-RoBERTa Cross-Encoder.
Stage A:
- Train on 103,411 balanced positive + structured/ANN hard negatives
- Format: Text A: S1 [SEP] Text B: Candidate
- Loss: BCEWithLogitsLoss
- Precision/Recall tracking
- Saves model checkpoint to experiments/EXP_N017/xlm_roberta_cross_encoder.pt
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONUNBUFFERED"] = "1"
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

CODE_DIR = Path(__file__).resolve().parents[1]
EXP_DIR = CODE_DIR / "experiments"
EXP_N017_DIR = EXP_DIR / "EXP_N017"

BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 2
LR = 2e-5
EPOCHS = 2
MAX_LENGTH = 96
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class CrossEncoderDataset(Dataset):
    def __init__(self, s1_ids: np.ndarray, cand_ids: np.ndarray, y: np.ndarray, id_to_text: Dict[str, str]):
        self.s1_ids = s1_ids
        self.cand_ids = cand_ids
        self.y = y.astype(np.float32)
        self.id_to_text = id_to_text

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        s1 = self.id_to_text.get(self.s1_ids[idx], "")
        cand = self.id_to_text.get(self.cand_ids[idx], "")
        return s1, cand, self.y[idx]


def collate_fn(batch, tokenizer):
    texts_a = [item[0] for item in batch]
    texts_b = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.float32)

    enc = tokenizer(
        texts_a,
        texts_b,
        max_length=MAX_LENGTH,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return enc, labels


def main():
    t_start = time.time()
    print("=" * 75, flush=True)
    print("[EXP_N017] Fine-Tuning Multilingual XLM-RoBERTa Cross-Encoder", flush=True)
    print("=" * 75, flush=True)

    # 1. Load formatted records into fast dict
    print("Loading formatted record store...", flush=True)
    store = np.load(EXP_N017_DIR / "formatted_record_store.npz")
    eids = store["eids"].astype(str)
    texts = store["texts"].astype(str)
    id_to_text = {eids[i]: texts[i] for i in range(len(eids))}
    print(f"Loaded {len(id_to_text):,} formatted records into memory.", flush=True)

    # 2. Load training manifest
    print("Loading training pairs manifest...", flush=True)
    tr = np.load(EXP_N017_DIR / "train_pairs_manifest.npz")
    train_s1 = tr["s1_ids"].astype(str)
    train_cand = tr["cand_ids"].astype(str)
    train_y = tr["y"].astype(np.int8)
    N_train = len(train_y)
    print(f"Training pairs: {N_train:,} (Pos: {np.sum(train_y==1):,}, Neg: {np.sum(train_y==0):,})", flush=True)

    # 3. Initialize Model and Tokenizer
    model_name = "xlm-roberta-base"
    print(f"Loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.to(DEVICE)

    # DataLoader
    train_dataset = CrossEncoderDataset(train_s1, train_cand, train_y, id_to_text)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.10), num_training_steps=total_steps)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    print(f"\nTraining Config:")
    print(f"  Total Samples:       {N_train:,}")
    print(f"  Batch Size:          {BATCH_SIZE} (Effective: {BATCH_SIZE * GRAD_ACCUM_STEPS})")
    print(f"  Max Sequence Length: {MAX_LENGTH}")
    print(f"  Learning Rate:       {LR}")
    print(f"  Epochs:              {EPOCHS}")
    print(f"  Optimizer Steps:     {total_steps:,}")
    print("-" * 75, flush=True)

    model.train()
    step_count = 0
    running_loss = 0.0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        print(f"\n[Epoch {epoch}/{EPOCHS}] Starting training...", flush=True)
        epoch_loss = 0.0
        optimizer.zero_grad()

        from tqdm import tqdm
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=100)

        for batch_idx, (enc, labels) in enumerate(pbar):
            input_ids = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)
                loss = criterion(logits, labels) / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            running_loss += loss.item() * GRAD_ACCUM_STEPS
            epoch_loss += loss.item() * GRAD_ACCUM_STEPS

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                step_count += 1

                if step_count % 50 == 0:
                    avg_loss = running_loss / 50
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
                    running_loss = 0.0

        print(f"[Epoch {epoch}] Complete. Mean Loss: {epoch_loss / len(train_loader):.4f}", flush=True)

    # Save Model Checkpoint
    checkpoint_path = EXP_N017_DIR / "xlm_roberta_cross_encoder.pt"
    print(f"\nSaving fine-tuned Cross-Encoder weights to: {checkpoint_path}...", flush=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved successfully. Total training runtime: {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
