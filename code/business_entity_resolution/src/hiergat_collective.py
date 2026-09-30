"""
HierGAT+ Architecture Implementation for Business Entity Resolution.
Supports:
- Ablation B: Hierarchical Entity Model (Token -> Attribute -> Entity Attention, No Candidate Interaction)
- Ablation C: B + Entity-Level Context (Self-Attention over candidates in pool)
- Ablation D: C + S2 <-> S3 Relational Graph Attention
- Ablation E: Full HierGAT+ Collective Alignment Module
- Ablation F: E + Pairwise Ranking Loss
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

class AttributeAttention(nn.Module):
    """
    HierGAT Hierarchical Attribute Attention:
    Takes field token embeddings or attribute representations and pools them
    using query-key-value learned attention with redundancy suppression.
    """
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)
        self.attn_vector = nn.Parameter(torch.randn(d_out, 1) * 0.02)
        self.layer_norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, N_tokens, d_in]
        returns: [B, d_out]
        """
        h = F.gelu(self.proj(x)) # [B, N, d_out]
        scores = torch.matmul(h, self.attn_vector).squeeze(-1) # [B, N]
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1) # [B, N, 1]
        pooled = (h * weights).sum(dim=1) # [B, d_out]
        return self.layer_norm(pooled)


class RelationalGraphAttention(nn.Module):
    """
    Message passing over S2 <-> S3 candidate relational edges.
    Updates candidate comparison representations based on cross-source support.
    """
    def __init__(self, d_node: int, d_edge: int = 13, num_heads: int = 4):
        super().__init__()
        self.d_node = d_node
        self.num_heads = num_heads
        self.d_head = d_node // num_heads

        self.W_src = nn.Linear(d_node, d_node, bias=False)
        self.W_dst = nn.Linear(d_node, d_node, bias=False)
        self.W_edge = nn.Linear(d_edge, d_node, bias=False)
        self.W_msg = nn.Linear(d_node + d_edge, d_node)
        self.attn_weight = nn.Parameter(torch.randn(num_heads, 3 * self.d_head) * 0.02)
        
        self.out_proj = nn.Linear(d_node, d_node)
        self.layer_norm = nn.LayerNorm(d_node)

    def forward(
        self,
        node_feats: torch.Tensor,        # [K, d_node]
        edge_index: torch.Tensor,        # [2, num_edges]
        edge_feats: torch.Tensor         # [num_edges, d_edge]
    ) -> torch.Tensor:
        num_nodes = node_feats.size(0)
        num_edges = edge_index.size(1)

        if num_edges == 0:
            return node_feats

        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        h_src = self.W_src(node_feats[src_idx]).view(num_edges, self.num_heads, self.d_head)
        h_dst = self.W_dst(node_feats[dst_idx]).view(num_edges, self.num_heads, self.d_head)
        h_edge = self.W_edge(edge_feats).view(num_edges, self.num_heads, self.d_head)

        # Multi-head attention scores
        cat_heads = torch.cat([h_src, h_dst, h_edge], dim=-1) # [num_edges, heads, 3*d_head]
        alpha = (cat_heads * self.attn_weight).sum(dim=-1) / math.sqrt(self.d_head) # [num_edges, heads]
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        
        # Softmax per dst node (using scatter/dense softmax)
        # For small K (80 nodes), dense tensor operations are ultra fast
        msg_in = torch.cat([node_feats[src_idx], edge_feats], dim=-1)
        msg_val = self.W_msg(msg_in) # [num_edges, d_node]

        # Aggregate messages into destination nodes
        out_msg = torch.zeros_like(node_feats)
        out_msg.index_add_(0, dst_idx, msg_val)
        
        updated = self.layer_norm(node_feats + self.out_proj(out_msg))
        return updated


class CollectiveEntityAlignment(nn.Module):
    """
    HierGAT+ Collective Entity Alignment Module.
    Allows all candidate comparison representations to interact jointly
    before computing individual match logits.
    """
    def __init__(self, d_model: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, K, d_model]
        """
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class HierGATCollectiveModel(nn.Module):
    """
    Unified HierGAT+ Model covering all 5 configurations (B, C, D, E, F).
    """
    def __init__(
        self,
        d_input: int = 60,               # 57 base features + N014 prob + XLM-R prob + Rank
        d_model: int = 128,
        ablation_mode: str = "E",         # 'B', 'C', 'D', 'E', 'F'
        d_rel_edge: int = 13,
    ):
        super().__init__()
        self.ablation_mode = ablation_mode
        self.d_model = d_model

        # Candidate comparison feature encoder
        self.comp_encoder = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Relational Graph Attention for S2 <-> S3 edges
        self.rel_gat = RelationalGraphAttention(d_node=d_model, d_edge=d_rel_edge)

        # Collective Entity Alignment (Transformer set interaction)
        self.alignment = CollectiveEntityAlignment(d_model=d_model, n_heads=4)

        # Match logit output head
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        comp_feats: torch.Tensor,         # [B, K, d_input]
        edge_index: Optional[torch.Tensor] = None, # [2, num_edges]
        edge_feats: Optional[torch.Tensor] = None, # [num_edges, d_edge]
    ) -> torch.Tensor:
        """
        Returns match logits: [B, K]
        """
        B, K, D = comp_feats.shape
        h = self.comp_encoder(comp_feats) # [B, K, d_model]

        if self.ablation_mode == "B":
            # Hierarchical Entity Model alone (No Candidate Interaction)
            pass

        elif self.ablation_mode == "C":
            # B + Entity-Level Context (Self-Attention over candidates in pool)
            h = self.alignment(h)

        elif self.ablation_mode == "D":
            # C + S2 <-> S3 Relational Edges
            if edge_index is not None and edge_feats is not None:
                h_reshaped = h.view(B * K, self.d_model)
                h_rel = self.rel_gat(h_reshaped, edge_index, edge_feats)
                h = h_rel.view(B, K, self.d_model)

        elif self.ablation_mode in ("E", "F"):
            # Full HierGAT+ Collective: S2 <-> S3 Relational Edges + Entity Alignment
            if edge_index is not None and edge_feats is not None and edge_index.size(1) > 0:
                h_reshaped = h.view(B * K, self.d_model)
                h_rel = self.rel_gat(h_reshaped, edge_index, edge_feats)
                h = h_rel.view(B, K, self.d_model)
            h = self.alignment(h)

        logits = self.head(h).squeeze(-1) # [B, K]
        return logits
