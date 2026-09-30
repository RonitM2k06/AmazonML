"""
Two-Tower Encoder for Business Entity Resolution — EXP_N010.

Architecture:
  - Shared transformer backbone: paraphrase-multilingual-MiniLM-L12-v2 (384-dim)
  - NameTower:    backbone + linear projection → 256-dim, L2-normalised
  - AddressTower: backbone + linear projection → 256-dim, L2-normalised
  - FusionHead:   concat (512-dim) + LayerNorm + Linear → 256-dim, L2-normalised

Training:
  - Stage A: Random negatives  (InfoNCE, temperature=0.05)
  - Stage B: Lexical negatives (hard negatives by BM25/token overlap)
  - Stage C: Mined negatives   (high-scoring false positives from N009)

Inference:
  - Encode full record as fused 256-dim vector for FAISS ANN retrieval.
  - Country acts as a HARD partition — never mix countries in the index.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BACKBONE_DIM = 384      # output dim of the MiniLM backbone
PROJ_DIM     = 256      # projected / fused embedding dim
MAX_SEQ_LEN  = 64       # Business names & addresses rarely exceed 30 tokens; 64 cuts attention complexity by ~55%
TEMPERATURE  = 0.05     # InfoNCE temperature


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings, ignoring padding positions."""
    mask = attention_mask.unsqueeze(-1).float()
    return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


# ---------------------------------------------------------------------------
# Tower: single-field encoder (name or address)
# ---------------------------------------------------------------------------

class Tower(nn.Module):
    """
    Encode one text field using the shared backbone + a learned projection.
    Output is L2-normalised.
    """

    def __init__(self, backbone: nn.Module, in_dim: int = BACKBONE_DIM, out_dim: int = PROJ_DIM):
        super().__init__()
        self.backbone  = backbone
        self.proj      = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.LayerNorm(out_dim),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Handle both BaseModelOutput and plain tensor outputs
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        pooled = mean_pool(hidden, attention_mask)   # [B, backbone_dim]
        proj   = self.proj(pooled)                    # [B, out_dim]
        return F.normalize(proj, dim=-1)


# ---------------------------------------------------------------------------
# Two-Tower Model with learned fusion
# ---------------------------------------------------------------------------

class TwoTowerModel(nn.Module):
    """
    TwoTowerModel:
      name_emb    = NameTower(name_text)        # [B, 256] L2-normed
      addr_emb    = AddressTower(addr_text)     # [B, 256] L2-normed
      fused       = FusionHead(cat(name, addr)) # [B, 256] L2-normed

    Optionally exposes name-only or address-only towers for ablation.
    """

    def __init__(self, model_name: str = BASE_MODEL):
        super().__init__()
        from transformers import AutoModel
        backbone = AutoModel.from_pretrained(model_name)

        self.name_tower = Tower(backbone, BACKBONE_DIM, PROJ_DIM)
        self.addr_tower = Tower(backbone, BACKBONE_DIM, PROJ_DIM)   # shared backbone

        # Fusion: concat → LayerNorm → Linear → L2-norm (no ReLU — keeps gradient flow)
        self.fusion = nn.Sequential(
            nn.LayerNorm(PROJ_DIM * 2),
            nn.Linear(PROJ_DIM * 2, PROJ_DIM, bias=False),
        )

    def encode_name(self, name_ids: torch.Tensor, name_mask: torch.Tensor) -> torch.Tensor:
        return self.name_tower(name_ids, name_mask)

    def encode_addr(self, addr_ids: torch.Tensor, addr_mask: torch.Tensor) -> torch.Tensor:
        return self.addr_tower(addr_ids, addr_mask)

    def fuse(self, name_emb: torch.Tensor, addr_emb: torch.Tensor) -> torch.Tensor:
        cat    = torch.cat([name_emb, addr_emb], dim=-1)   # [B, 512]
        fused  = self.fusion(cat)                           # [B, 256]
        return F.normalize(fused, dim=-1)

    def forward(
        self,
        name_ids: torch.Tensor, name_mask: torch.Tensor,
        addr_ids: torch.Tensor, addr_mask: torch.Tensor,
    ) -> torch.Tensor:
        name_emb = self.encode_name(name_ids, name_mask)
        addr_emb = self.encode_addr(addr_ids, addr_mask)
        return self.fuse(name_emb, addr_emb)


# ---------------------------------------------------------------------------
# InfoNCE Loss (in-batch negatives + optional hard negatives)
# ---------------------------------------------------------------------------

class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE (NTXentLoss) with optional hard-negative injection.

    Inputs:
        query_emb:   [B, D]  — S1 fused embeddings
        pos_emb:     [B, D]  — matched S2/S3 fused embeddings
        hard_emb:    Optional [B, K, D] — K hard negatives per query

    Returns:
        Scalar loss averaged over the batch.
    """

    def __init__(self, temperature: float = TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query_emb: torch.Tensor,
        pos_emb:   torch.Tensor,
        hard_emb:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = query_emb.size(0)

        # In-batch similarity matrix [B, B]
        sim_matrix = torch.matmul(query_emb, pos_emb.T) / self.temperature

        # If hard negatives supplied, append them as extra columns
        if hard_emb is not None:
            # hard_emb: [B, K, D] → expand query: [B, 1, D] → [B, K]
            q_exp    = query_emb.unsqueeze(1)          # [B, 1, D]
            hard_sim = (q_exp * hard_emb).sum(-1) / self.temperature   # [B, K]
            # Build augmented logits: [B, B+K] — diagonal = positive
            logits   = torch.cat([sim_matrix, hard_sim], dim=-1)       # [B, B+K]
        else:
            logits   = sim_matrix

        labels = torch.arange(B, device=query_emb.device)
        loss_q = F.cross_entropy(logits,           labels)             # query→positive
        loss_k = F.cross_entropy(sim_matrix.T,     labels)             # positive→query (symmetric)
        return (loss_q + loss_k) / 2.0


# ---------------------------------------------------------------------------
# Tokenization helper (stateless, call once per batch)
# ---------------------------------------------------------------------------

class RecordTokenizer:
    """
    Lightweight wrapper: tokenise name/address separately for two-tower input.
    """

    def __init__(self, model_name: str = BASE_MODEL, max_len: int = MAX_SEQ_LEN):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len   = max_len

    def tokenize_field(self, texts: List[str], device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)


# ---------------------------------------------------------------------------
# Inference: encode a full record corpus to FAISS-ready vectors
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_records_fused(
    model:      TwoTowerModel,
    tokenizer:  RecordTokenizer,
    records:    Dict[str, Tuple],   # {id: (eid, name, addr, country)}
    batch_size: int = 256,
    device:     str = "cuda",
    progress_every: int = 25000,
) -> Tuple[List[str], np.ndarray]:
    """
    Encode a dict of records using the fused two-tower embedding.

    Returns:
        ids  : list of record IDs (same order as rows)
        vecs : float32 ndarray [N, PROJ_DIM], L2-normalised
    """
    model.eval()
    model.to(device)

    ids   = list(records.keys())
    names = [(records[i][1] if len(records[i]) > 1 else "") or "" for i in ids]
    addrs = [(records[i][2] if len(records[i]) > 2 else "") or "" for i in ids]

    N      = len(ids)
    all_v  = []
    t0     = time.time()

    for start in range(0, N, batch_size):
        end      = min(start + batch_size, N)
        b_names  = names[start:end]
        b_addrs  = addrs[start:end]

        n_ids, n_mask = tokenizer.tokenize_field(b_names, device)
        a_ids, a_mask = tokenizer.tokenize_field(b_addrs, device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            fused = model(n_ids, n_mask, a_ids, a_mask)  # [b, PROJ_DIM]
        all_v.append(fused.cpu().float().numpy())

        if (start + batch_size) % progress_every < batch_size:
            pct = min(100, end / N * 100)
            print(f"  [encode_fused] {end:,}/{N:,} ({pct:.1f}%) — {time.time()-t0:.0f}s")

    vecs = np.vstack(all_v) if all_v else np.zeros((0, PROJ_DIM), dtype=np.float32)
    print(f"[encode_fused] Done: {N:,} records -> shape {vecs.shape} in {time.time()-t0:.1f}s")
    return ids, vecs


# ---------------------------------------------------------------------------
# Save / load fine-tuned model
# ---------------------------------------------------------------------------

def save_two_tower(model: TwoTowerModel, path: str):
    """Save model state dict."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"[TwoTower] Saved to {path} ({size_mb:.1f} MB)")


def load_two_tower(path: str, model_name: str = BASE_MODEL, device: str = "cuda") -> TwoTowerModel:
    """Load model from state dict."""
    model = TwoTowerModel(model_name=model_name)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"[TwoTower] Loaded from {path}")
    return model
