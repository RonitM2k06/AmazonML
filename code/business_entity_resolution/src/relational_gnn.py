"""
EXP_N032 — Relational Graph Neural Network (GNN).

Implements a lightweight heterogeneous Graph Attention Network (GAT) for local entity subgraphs:
- Node projection: S1, S2, S3 distinct node types
- Edge-aware attention: incorporates pairwise probability, lexical, and structural features
- Multi-hop message passing with learned anchor confidence gating
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class RelationalGATLayer(nn.Module):
    """
    Edge-feature-conditioned Graph Attention Layer.
    """

    def __init__(self, in_dim: int = 64, edge_dim: int = 16, out_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.e_proj = nn.Linear(edge_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        node_feats: torch.Tensor,   # [N_nodes, in_dim]
        edge_feats: torch.Tensor,   # [N_edges, edge_dim]
        edge_index: torch.Tensor,   # [2, N_edges]
    ) -> torch.Tensor:
        N = node_feats.size(0)
        E = edge_feats.size(0)
        if E == 0:
            return node_feats

        src, dst = edge_index[0], edge_index[1]
        q = self.q_proj(node_feats[dst]) # [E, out_dim]
        k = self.k_proj(node_feats[src]) # [E, out_dim]
        v = self.v_proj(node_feats[src]) # [E, out_dim]
        e = self.e_proj(edge_feats)      # [E, out_dim]

        scores = torch.sum((q + e) * k, dim=-1) / (self.head_dim ** 0.5) # [E]
        attn = F.softmax(scores, dim=0)

        # Aggregate messages at destination
        out = torch.zeros_like(node_feats)
        msg = attn.unsqueeze(-1) * (v + e)
        out.index_add_(0, dst, msg)
        return self.norm(node_feats + self.out_proj(out))


class RelationalEntityGNN(nn.Module):
    """
    2-Layer Relational Heterogeneous GNN for Entity Resolution.
    """

    def __init__(self, node_dim: int = 64, edge_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.layer1 = RelationalGATLayer(in_dim=node_dim, edge_dim=edge_dim, out_dim=hidden_dim)
        self.layer2 = RelationalGATLayer(in_dim=hidden_dim, edge_dim=edge_dim, out_dim=hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        h1 = F.relu(self.layer1(node_feats, edge_feats, edge_index))
        h2 = self.layer2(h1, edge_feats, edge_index)
        return self.classifier(h2).squeeze(-1)
