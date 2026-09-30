"""
EXP_N027 — Character-Level Entity Foundation Model Architecture.

Implements multi-scale character n-gram convolutional filters with BiGRU and
multi-head attention pooling to encode business names and addresses into compact,
dense representations robust to severe orthographic corruptions, acronyms, and
transliterations.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class MultiScaleCharCNNBiGRU(nn.Module):
    """
    Multi-scale Character CNN + BiGRU with Attention Pooling.
    
    Filters across n-gram scales (2, 3, 4, 5, 6) capture character n-gram motifs,
    followed by a bidirectional GRU to capture global sequence dependencies,
    and multi-head attention pooling for adaptive token aggregation.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        out_dim: int = 256,
        filter_sizes: tuple[int, ...] = (2, 3, 4, 5, 6),
        num_filters: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Parallel multi-scale 1D CNNs
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k // 2,
            )
            for k in filter_sizes
        ])
        
        conv_total_dim = num_filters * len(filter_sizes)
        self.gru = nn.GRU(
            input_size=conv_total_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        
        # Attention Pooling
        self.attn_dense = nn.Linear(hidden_dim * 2, 64)
        self.attn_score = nn.Linear(64, 1, bias=False)
        
        # Projection Head
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [batch_size, seq_len] int64 character IDs
        mask: [batch_size, seq_len] bool (True for valid chars, False for pad)
        returns: [batch_size, out_dim] L2-normalized embedding
        """
        # [B, L, E]
        emb = self.embedding(x)
        # Permute for Conv1d: [B, E, L]
        emb_conv = emb.transpose(1, 2)
        
        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(emb_conv))
            # Slice or pad if even kernel size caused length mismatch
            if c.size(2) > emb.size(1):
                c = c[:, :, :emb.size(1)]
            elif c.size(2) < emb.size(1):
                c = F.pad(c, (0, emb.size(1) - c.size(2)))
            conv_outs.append(c)
        
        # Concat along channels: [B, total_filters, L]
        multi_conv = torch.cat(conv_outs, dim=1)
        multi_conv = multi_conv.transpose(1, 2)  # [B, L, total_filters]
        multi_conv = self.dropout(multi_conv)
        
        gru_out, _ = self.gru(multi_conv)  # [B, L, hidden_dim * 2]
        
        # Multi-head / self-attention pooling
        attn_weights = self.attn_score(torch.tanh(self.attn_dense(gru_out)))  # [B, L, 1]
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            attn_weights = attn_weights.masked_fill(~mask_expanded, -1e9)
        
        attn_norm = F.softmax(attn_weights, dim=1)
        pooled = torch.sum(gru_out * attn_norm, dim=1)  # [B, hidden_dim * 2]
        
        out = self.proj(pooled)
        return F.normalize(out, p=2, dim=-1)


class CharEntityEncoder(nn.Module):
    """
    Dual-Pathway Character Entity Encoder with Learned Fusion Gate.
    Encodes business name and address separately, then combines them via an adaptive
    gated mechanism.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        out_dim: int = 256,
    ):
        super().__init__()
        self.name_encoder = MultiScaleCharCNNBiGRU(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
        )
        self.addr_encoder = MultiScaleCharCNNBiGRU(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
        )
        # Fusion gate learns when address is more reliable than name
        self.gate = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.Sigmoid(),
        )
        self.joint_proj = nn.Linear(out_dim, out_dim)

    def forward(
        self,
        name_ids: torch.Tensor,
        name_mask: torch.Tensor,
        addr_ids: torch.Tensor,
        addr_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_name = self.name_encoder(name_ids, name_mask)
        z_addr = self.addr_encoder(addr_ids, addr_mask)
        
        # Gated fusion
        gate_val = self.gate(torch.cat([z_name, z_addr], dim=-1))
        z_fused = gate_val * z_name + (1.0 - gate_val) * z_addr
        z_joint = F.normalize(self.joint_proj(z_fused), p=2, dim=-1)
        
        return z_name, z_addr, z_joint
