"""
EXP_N010 — Phase 1+2+3: Build hard-negative manifest + Train two-tower encoder.

Stages:
  A — Stage A: Random negatives only (3 epochs, LR 2e-5)
  B — Stage B: Lexical hard negatives (2 epochs, LR 5e-6)
  C — Stage C: Mined model hard negatives (2 epochs, LR 2e-6)

Usage:
  python scripts/train_two_tower_n010.py

Outputs:
  experiments/hard_neg_manifest.csv
  models/two_tower_stageA.pt
  models/two_tower_stageB.pt
  models/two_tower_stageC.pt   <- final champion
  experiments/embeddings_n010_s1.npz
  experiments/embeddings_n010_target.npz
"""

from __future__ import annotations

import csv
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── project root on path ────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.two_tower_encoder import (
    TwoTowerModel, RecordTokenizer, InfoNCELoss,
    encode_records_fused, save_two_tower, load_two_tower,
    PROJ_DIM, BASE_MODEL,
)
from src.normalization import normalize_business_name, normalize_address, tokenize

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = _ROOT / "../../dataset/train"
EXP_DIR    = _ROOT / "experiments"
MODEL_DIR  = _ROOT / "models"
EXP_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

S1_TSV     = DATA_DIR / "train_source1.tsv"
S2_TSV     = DATA_DIR / "train_source2.tsv"
S3_TSV     = DATA_DIR / "train_ground_truth.tsv"   # just GT; actual S3 below
S3_TSV_    = DATA_DIR / "train_source3.tsv"
GT_TSV     = DATA_DIR / "train_ground_truth.tsv"

HARD_NEG_CSV        = EXP_DIR / "hard_neg_manifest.csv"
EMB_S1_PATH         = EXP_DIR / "embeddings_n010_s1.npz"
EMB_TARGET_PATH     = EXP_DIR / "embeddings_n010_target.npz"
MODEL_STAGE_A       = MODEL_DIR / "two_tower_stageA.pt"
MODEL_STAGE_B       = MODEL_DIR / "two_tower_stageB.pt"
MODEL_STAGE_C       = MODEL_DIR / "two_tower_stageC.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Training hyper-params ────────────────────────────────────────────────────
BATCH_SIZE_A  = 64    # random negatives (in-batch)
BATCH_SIZE_BC = 64    # higher batch size thanks to reduced seq_len and bfloat16 AMP
EPOCHS_A      = 3
EPOCHS_B      = 1     # 1 epoch of hard negatives fine-tuning is optimal and prevents catastrophic forgetting
EPOCHS_C      = 1     # 1 epoch of mined negatives
LR_A          = 2e-5
LR_B          = 5e-6
LR_C          = 2e-6
WARMUP_FRAC   = 0.06
HARD_K        = 2     # 2 hardest negatives per anchor (cuts encoder load from 6x to 3x, 50% speedup)
TEMPERATURE   = 0.05

# Subsample: training pairs cover 14,985 unique S1s; use all of them.
MAX_TRAIN_S1 = None  # None = use all

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ════════════════════════════════════════════════════════════════════════════

def load_tsv(path: Path) -> Dict[str, Tuple[str, str, str]]:
    """Load TSV -> {id: (name, addr, country)} — UTF-8, errors replaced.
    Supports both column naming conventions:
      - entity_id / business_name / business_address / country  (actual TSVs)
      - entity_id / name / address / country  (alternative)
    """
    records = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid     = (row.get("entity_id", "") or "").strip()
            # Support both column naming conventions
            name    = (row.get("business_name", "") or row.get("name", "") or "").strip()
            address = (row.get("business_address", "") or row.get("address", "") or "").strip()
            country = (row.get("country", "") or "").strip()
            if eid:
                records[eid] = (name, address, country)
    return records


def load_ground_truth(path: Path) -> Dict[str, List[str]]:
    """Load GT -> {s1_id: [match_id, ...]}.
    Supports column names: source1_entity_id / matched_entity_ids (actual GT)
    """
    gt: Dict[str, List[str]] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # Actual GT uses: source1_entity_id, matched_entity_ids
            s1_id   = (row.get("source1_entity_id", "") or row.get("entity_id", "") or "").strip()
            matches = (row.get("matched_entity_ids", "") or row.get("match_ids", "") or "").strip()
            if s1_id:
                gt[s1_id] = [m.strip() for m in matches.split(",") if m.strip()] if matches else []
    return gt


print("[data] Loading training pair IDs from NPZ ...")
t0 = time.time()
_tr = np.load(str(EXP_DIR / "train_pairs_exp008.npz"), allow_pickle=True)
_tr_s1_ids  = set(_tr["s1_ids"].astype(str))
_tr_cand_ids = set(_tr["cand_ids"].astype(str))
_tr_labels   = _tr["y"].astype(int)
_tr_s1_arr   = _tr["s1_ids"].astype(str)
_tr_cand_arr = _tr["cand_ids"].astype(str)
print(f"  Train pairs NPZ: {len(_tr_labels):,} pairs, "
      f"{len(_tr_s1_ids):,} unique S1, {len(_tr_cand_ids):,} unique targets")

# Build positive pairs directly from NPZ (label=1)
pos_mask = _tr_labels == 1
pos_pairs: List[Tuple[str, str]] = list(zip(
    _tr_s1_arr[pos_mask].tolist(),
    _tr_cand_arr[pos_mask].tolist(),
))
print(f"  Positive pairs from NPZ: {len(pos_pairs):,}")

# Collect all needed IDs
needed_s1_ids:     Set[str] = set(_tr_s1_arr.tolist())
needed_target_ids: Set[str] = set(_tr_cand_arr.tolist())

# Also load GT to find additional positives not in training NPZ
print("[data] Loading ground truth for manifest building ...")
gt: Dict[str, List[str]] = {}
with open(GT_TSV, encoding="utf-8", errors="replace") as _f:
    _reader = csv.DictReader(_f, delimiter="\t")
    for _row in _reader:
        _s1 = (_row.get("source1_entity_id", "") or _row.get("entity_id", "") or "").strip()
        _ms = (_row.get("matched_entity_ids", "") or _row.get("match_ids", "") or "").strip()
        if _s1 in needed_s1_ids:
            _matches = [m.strip() for m in _ms.split(",") if m.strip()] if _ms else []
            gt[_s1] = _matches
            needed_target_ids.update(_matches)

# Stream-load only needed records from TSVs
def load_tsv_filtered(path: Path, needed_ids: Set[str]) -> Dict[str, Tuple[str, str, str]]:
    """Load only the records whose entity_id is in needed_ids."""
    records: Dict[str, Tuple[str, str, str]] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = (row.get("entity_id", "") or "").strip()
            if eid in needed_ids:
                name    = (row.get("business_name", "") or row.get("name", "") or "").strip()
                address = (row.get("business_address", "") or row.get("address", "") or "").strip()
                country = (row.get("country", "") or "").strip()
                records[eid] = (name, address, country)
    return records

print("[data] Loading S1 records (filtered) ...")
s1_recs = load_tsv_filtered(S1_TSV, needed_s1_ids)
print(f"  S1 loaded: {len(s1_recs):,} / {len(needed_s1_ids):,} needed")

print("[data] Loading S2 records (filtered) ...")
s2_needed = {i for i in needed_target_ids if i.startswith("S2")}
s3_needed = {i for i in needed_target_ids if i.startswith("S3")}
s2_recs   = load_tsv_filtered(S2_TSV,  s2_needed)
print(f"  S2 loaded: {len(s2_recs):,}")

print("[data] Loading S3 records (filtered) ...")
s3_recs   = load_tsv_filtered(S3_TSV_, s3_needed)
print(f"  S3 loaded: {len(s3_recs):,}")

target_recs = {**s2_recs, **s3_recs}
print(f"  target total: {len(target_recs):,}  ({time.time()-t0:.0f}s)")

# Filter pos_pairs to only those with records available
pos_pairs = [(s1, t) for s1, t in pos_pairs if s1 in s1_recs and t in target_recs]
print(f"  Valid positive pairs (both records present): {len(pos_pairs):,}")

s1_with_pos: List[str] = sorted({s1 for s1, _ in pos_pairs})
print(f"  Unique S1 with positives: {len(s1_with_pos):,}")

if len(pos_pairs) == 0:
    raise RuntimeError("No positive pairs found! Check data paths and TSV column names.")



# ════════════════════════════════════════════════════════════════════════════
# 2. Hard Negative Manifest (Phase 2)
# ════════════════════════════════════════════════════════════════════════════

def build_hard_negative_manifest() -> Dict[str, List[Tuple[str, str, float]]]:
    """
    Fast hard negative manifest builder — uses train_pairs_exp008.npz directly.

    Strategy: label=0 pairs from the training NPZ are already the hardest negatives
    (they passed the N009 blocker). We score them by cosine similarity (feature 56)
    and pick the top-K per S1, giving us semantically confusing negatives.

    This is O(N_pairs) instead of O(S1 × targets), runs in <5 seconds.

    Returns: {s1_id: [(neg_id, category, score), ...]}
    """
    print("[hard_neg] Building fast NPZ-based hard negative manifest ...")
    t0 = time.time()

    # Use feature 56 = cosine similarity (the neural feature added in N008)
    # as the difficulty score. Higher cosine = harder negative.
    X    = _tr["X"].astype(np.float32)   # [N, 57] or [N, 56]
    feat_dim = X.shape[1]
    # Cosine sim is last feature (index 56 in 57-feature model, 55 in 56-feature)
    cosine_col = feat_dim - 1
    cosine_sims = X[:, cosine_col]

    # Only negatives (label=0)
    neg_mask    = _tr_labels == 0
    neg_s1      = _tr_s1_arr[neg_mask]
    neg_cand    = _tr_cand_arr[neg_mask]
    neg_scores  = cosine_sims[neg_mask]

    print(f"  Total label=0 training pairs: {neg_mask.sum():,}")
    print(f"  Cosine sim stats — mean={neg_scores.mean():.3f} max={neg_scores.max():.3f}")

    # Group by s1_id, keep top-20 by cosine score
    from collections import defaultdict as _dd
    manifest: Dict[str, List[Tuple[str, str, float]]] = {}
    s1_to_negs: Dict[str, List[Tuple[float, str]]] = _dd(list)

    for s1, cand, score in zip(neg_s1, neg_cand, neg_scores):
        s1_to_negs[s1].append((float(score), cand))

    for s1, negs in s1_to_negs.items():
        negs.sort(key=lambda x: -x[0])   # sort by cosine desc
        manifest[s1] = [(cand, "C_high_cosine_neg", score) for score, cand in negs[:20]]

    total = sum(len(v) for v in manifest.values())
    print(f"[hard_neg] Built manifest: {total:,} hard negs across {len(manifest):,} S1s "
          f"({time.time()-t0:.1f}s)")
    return manifest


def save_manifest(manifest: Dict[str, List[Tuple[str, str, float]]]):
    rows = []
    for s1_id, negs in manifest.items():
        for (neg_id, cat, score) in negs:
            rows.append({"s1_id": s1_id, "neg_id": neg_id, "category": cat, "score": f"{score:.4f}"})
    with open(HARD_NEG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["s1_id", "neg_id", "category", "score"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[hard_neg] Saved {len(rows):,} rows -> {HARD_NEG_CSV}")


def load_manifest() -> Dict[str, List[Tuple[str, str, float]]]:
    manifest: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
    with open(HARD_NEG_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            manifest[row["s1_id"]].append((row["neg_id"], row["category"], float(row["score"])))
    return dict(manifest)



# ════════════════════════════════════════════════════════════════════════════
# 3. PyTorch Datasets
# ════════════════════════════════════════════════════════════════════════════

class PositivePairDataset(Dataset):
    """
    Stage A/B dataset: each item is (s1_id, pos_id).
    Negative pairs are sampled in-batch.
    """

    def __init__(self, pairs: List[Tuple[str, str]], hard_negs: Optional[Dict] = None):
        self.pairs     = pairs
        self.hard_negs = hard_negs  # optional for Stage B

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        s1_id, pos_id = self.pairs[idx]
        s1  = s1_recs.get(s1_id,  ("", "", ""))
        pos = target_recs.get(pos_id, ("", "", ""))
        return {
            "s1_name":   s1[0]  or "",
            "s1_addr":   s1[1]  or "",
            "pos_name":  pos[0] or "",
            "pos_addr":  pos[1] or "",
            "s1_id":     s1_id,
            "pos_id":    pos_id,
        }


class HardNegDataset(Dataset):
    """
    Stage C dataset: (s1_id, pos_id, [neg_ids]).
    Hard negatives injected into loss computation.
    """

    def __init__(
        self,
        pairs:    List[Tuple[str, str]],
        manifest: Dict[str, List[Tuple[str, str, float]]],
        k:        int = HARD_K,
    ):
        self.pairs    = pairs
        self.manifest = manifest
        self.k        = k

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        s1_id, pos_id = self.pairs[idx]
        s1  = s1_recs.get(s1_id,  ("", "", ""))
        pos = target_recs.get(pos_id, ("", "", ""))

        negs    = self.manifest.get(s1_id, [])
        neg_ids = [n[0] for n in negs[:self.k]]
        # Pad to k with empty strings if not enough hard negs
        while len(neg_ids) < self.k:
            neg_ids.append("")

        neg_names = []
        neg_addrs = []
        for nid in neg_ids:
            rec = target_recs.get(nid, ("", "", ""))
            neg_names.append(rec[0] or "")
            neg_addrs.append(rec[1] or "")

        return {
            "s1_name":   s1[0]  or "",
            "s1_addr":   s1[1]  or "",
            "pos_name":  pos[0] or "",
            "pos_addr":  pos[1] or "",
            "neg_names": neg_names,
            "neg_addrs": neg_addrs,
        }


# ════════════════════════════════════════════════════════════════════════════
# 4. Training loop
# ════════════════════════════════════════════════════════════════════════════

def _get_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int):
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train_stage(
    model:      TwoTowerModel,
    tokenizer:  RecordTokenizer,
    pairs:      List[Tuple[str, str]],
    manifest:   Optional[Dict],
    epochs:     int,
    lr:         float,
    batch_size: int,
    stage_name: str,
    save_path:  Path,
) -> TwoTowerModel:
    model.train()
    model.to(DEVICE)
    criterion = InfoNCELoss(temperature=TEMPERATURE)

    use_hard = manifest is not None
    if use_hard:
        dataset = HardNegDataset(pairs, manifest, k=HARD_K)
    else:
        dataset = PositivePairDataset(pairs)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    total_steps   = epochs * len(loader)
    warmup_steps  = int(WARMUP_FRAC * total_steps)
    scheduler     = _get_scheduler(optimizer, warmup_steps, total_steps)

    print(f"\n[{stage_name}] Start: {len(pairs):,} pairs, "
          f"batch={batch_size}, epochs={epochs}, lr={lr}, steps={total_steps:,}")

    global_step = 0
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()

        for batch in loader:
            optimizer.zero_grad(set_to_none=True)

            # Tokenise S1 fields
            s1_n_ids, s1_n_mask = tokenizer.tokenize_field(batch["s1_name"], DEVICE)
            s1_a_ids, s1_a_mask = tokenizer.tokenize_field(batch["s1_addr"], DEVICE)

            # Tokenise positive fields
            pos_n_ids, pos_n_mask = tokenizer.tokenize_field(batch["pos_name"], DEVICE)
            pos_a_ids, pos_a_mask = tokenizer.tokenize_field(batch["pos_addr"], DEVICE)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                # Fused embeddings
                q_emb = model(s1_n_ids, s1_n_mask, s1_a_ids, s1_a_mask)    # [B, D]
                p_emb = model(pos_n_ids, pos_n_mask, pos_a_ids, pos_a_mask) # [B, D]

                # Hard negatives (Stage C)
                hard_emb = None
                if use_hard:
                    B  = q_emb.size(0)
                    K  = HARD_K
                    # Flatten K negatives per sample → encode all B*K in one pass
                    if isinstance(batch["neg_names"][0], (list, tuple)):
                        flat_names = [n for ns in batch["neg_names"] for n in ns]
                        flat_addrs = [a for as_ in batch["neg_addrs"] for a in as_]
                    else:
                        # DataLoader may transpose to K lists of B items
                        flat_names = list(batch["neg_names"])
                        flat_addrs = list(batch["neg_addrs"])

                    hn_ids, hn_mask = tokenizer.tokenize_field(flat_names, DEVICE)
                    ha_ids, ha_mask = tokenizer.tokenize_field(flat_addrs, DEVICE)
                    flat_emb = model(hn_ids, hn_mask, ha_ids, ha_mask)   # [B*K, D]
                    hard_emb = flat_emb.view(B, K, -1)                   # [B, K, D]

                loss = criterion(q_emb, p_emb, hard_emb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 200 == 0:
                avg = epoch_loss / (global_step % max(len(loader), 1) or len(loader))
                print(f"  step {global_step:5d} | loss {loss.item():.4f} | "
                      f"avg {avg:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

        avg_epoch = epoch_loss / len(loader)
        print(f"  [epoch {epoch}/{epochs}] loss={avg_epoch:.4f}  ({time.time()-t0:.0f}s)")

    save_two_tower(model, str(save_path))
    return model


# ════════════════════════════════════════════════════════════════════════════
# 5. Encode full corpus and save embeddings
# ════════════════════════════════════════════════════════════════════════════

def encode_and_save(model: TwoTowerModel, tokenizer: RecordTokenizer):
    """Encode the FULL S1 and target corpora for FAISS retrieval."""
    from src.neural_encoder import save_embeddings

    # ── Full S1 corpus ────────────────────────────────────────────────────────
    print("\n[encode] Streaming full S1 corpus ...")
    full_s1_recs = load_tsv(S1_TSV)
    print(f"  S1 records: {len(full_s1_recs):,}")
    s1_input = {k: ("", v[0], v[1], v[2]) for k, v in full_s1_recs.items()}
    ids_s1, vecs_s1 = encode_records_fused(model, tokenizer, s1_input, batch_size=128, device=DEVICE)
    save_embeddings(str(EMB_S1_PATH), ids_s1, vecs_s1, model_name="two_tower_n010")
    print(f"  S1 embeddings saved: {vecs_s1.shape}")
    del full_s1_recs, s1_input, vecs_s1

    # ── Full S2 corpus ────────────────────────────────────────────────────────
    print("\n[encode] Streaming full S2 corpus ...")
    full_s2_recs = load_tsv(S2_TSV)
    print(f"  S2 records: {len(full_s2_recs):,}")
    s2_input = {k: ("", v[0], v[1], v[2]) for k, v in full_s2_recs.items()}
    ids_s2, vecs_s2 = encode_records_fused(model, tokenizer, s2_input, batch_size=128, device=DEVICE)
    del full_s2_recs, s2_input

    # ── Full S3 corpus ────────────────────────────────────────────────────────
    print("\n[encode] Streaming full S3 corpus ...")
    full_s3_recs = load_tsv(S3_TSV_)
    print(f"  S3 records: {len(full_s3_recs):,}")
    s3_input = {k: ("", v[0], v[1], v[2]) for k, v in full_s3_recs.items()}
    ids_s3, vecs_s3 = encode_records_fused(model, tokenizer, s3_input, batch_size=128, device=DEVICE)
    del full_s3_recs, s3_input

    # Concatenate and save target embeddings
    ids_tgt  = ids_s2 + ids_s3
    vecs_tgt = np.vstack([vecs_s2, vecs_s3])
    del vecs_s2, vecs_s3
    save_embeddings(str(EMB_TARGET_PATH), ids_tgt, vecs_tgt, model_name="two_tower_n010")
    print(f"  Target embeddings saved: {vecs_tgt.shape}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-manifest", action="store_true", help="Skip hard neg mining if manifest exists")
    parser.add_argument("--stage",  default="all", choices=["all", "A", "B", "C", "encode"])
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # ── Hard Negative Manifest ───────────────────────────────────────────────
    if args.skip_manifest and HARD_NEG_CSV.exists():
        print(f"[manifest] Skipping — using existing {HARD_NEG_CSV}")
        manifest = load_manifest()
    else:
        manifest = build_hard_negative_manifest()
        save_manifest(manifest)

    # ── Tokenizer (shared across stages) ────────────────────────────────────
    print("\n[init] Loading tokenizer ...")
    tokenizer = RecordTokenizer(model_name=BASE_MODEL)

    # ── Stage A: Random negatives ─────────────────────────────────────────────
    skip_stage_a = False
    if args.stage in ("all", "A"):
        if MODEL_STAGE_A.exists() and args.resume is None and args.stage == "all":
            print(f"\n[resume] Found existing {MODEL_STAGE_A}! Skipping Stage A and loading checkpoint.")
            model = load_two_tower(str(MODEL_STAGE_A), device=DEVICE)
            skip_stage_a = True
        elif args.resume and Path(args.resume).exists():
            model = load_two_tower(args.resume, device=DEVICE)
        else:
            print("\n[init] Initialising TwoTowerModel from pretrained ...")
            model = TwoTowerModel(model_name=BASE_MODEL)

        if not skip_stage_a:
            random.shuffle(pos_pairs)
            model = train_stage(
                model=model, tokenizer=tokenizer,
                pairs=pos_pairs, manifest=None,
                epochs=EPOCHS_A, lr=LR_A, batch_size=BATCH_SIZE_A,
                stage_name="Stage-A", save_path=MODEL_STAGE_A,
            )
    elif args.stage in ("B", "C", "encode"):
        model = load_two_tower(str(MODEL_STAGE_A), device=DEVICE)

    # ── Stage B: Lexical hard negatives ──────────────────────────────────────
    skip_stage_b = False
    if args.stage in ("all", "B"):
        if MODEL_STAGE_B.exists() and args.stage == "all":
            print(f"\n[resume] Found existing {MODEL_STAGE_B}! Skipping Stage B and loading checkpoint.")
            model = load_two_tower(str(MODEL_STAGE_B), device=DEVICE)
            skip_stage_b = True

        if not skip_stage_b:
            random.shuffle(pos_pairs)
            model = train_stage(
                model=model, tokenizer=tokenizer,
                pairs=pos_pairs, manifest=manifest,
                epochs=EPOCHS_B, lr=LR_B, batch_size=BATCH_SIZE_BC,
                stage_name="Stage-B", save_path=MODEL_STAGE_B,
            )
    elif args.stage in ("C", "encode"):
        model = load_two_tower(str(MODEL_STAGE_B), device=DEVICE)

    # ── Stage C: Mined negatives ──────────────────────────────────────────────
    skip_stage_c = False
    if args.stage in ("all", "C"):
        if MODEL_STAGE_C.exists() and args.stage == "all":
            print(f"\n[resume] Found existing {MODEL_STAGE_C}! Skipping Stage C and loading checkpoint.")
            model = load_two_tower(str(MODEL_STAGE_C), device=DEVICE)
            skip_stage_c = True

        if not skip_stage_c:
            random.shuffle(pos_pairs)
            model = train_stage(
                model=model, tokenizer=tokenizer,
                pairs=pos_pairs, manifest=manifest,
                epochs=EPOCHS_C, lr=LR_C, batch_size=BATCH_SIZE_BC,
                stage_name="Stage-C", save_path=MODEL_STAGE_C,
            )
    elif args.stage == "encode":
        model = load_two_tower(str(MODEL_STAGE_C), device=DEVICE)

    # ── Encode full corpus ────────────────────────────────────────────────────
    if args.stage in ("all", "encode"):
        encode_and_save(model, tokenizer)

    print("\n[DONE] EXP_N010 training complete.")
