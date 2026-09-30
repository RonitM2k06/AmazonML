"""
EXP_N027 — Address Character-Level Encoder.

Dedicated character encoder for address strings: learns house numbers, postal codes,
localities, streets, cities, abbreviations, and reordered address components.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from char_entity_encoder import MultiScaleCharCNNBiGRU

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class AddressCharEncoder(nn.Module):
    """
    Dedicated address sequence encoder.
    Optimized for multi-lingual and variable-format geographic addresses.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        out_dim: int = 256,
        filter_sizes: tuple[int, ...] = (2, 3, 4, 5, 6),
        num_filters: int = 64,
    ):
        super().__init__()
        self.encoder = MultiScaleCharCNNBiGRU(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            filter_sizes=filter_sizes,
            num_filters=num_filters,
        )

    def forward(self, addr_ids: torch.Tensor, addr_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(addr_ids, addr_mask)
