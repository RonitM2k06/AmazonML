"""
EXP_N025: Cross-Source Entity Translation & Latent Transformation Dataset.

Constructs training pairs and triplets across sources (S1 <-> S2, S1 <-> S3, S2 <-> S3)
with explicit source encoding, transformation tracking, and hard negatives.

Strictly entity-disjoint from the 25k validation set.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class CrossSourceEntityDataset(Dataset):
    """
    Dataset of matched pairs across (S1, S2, S3) with source IDs for translation learning.
    """
    def __init__(
        self,
        pairs: List[Tuple[str, str, str, str]], # (id_src, id_tgt, type_src, type_tgt)
        records: Dict[str, Tuple[str, str, str]], # {id: (name, addr, country)}
        hard_negatives: Optional[Dict[str, List[str]]] = None,
    ):
        self.pairs = pairs
        self.records = records
        self.hard_negatives = hard_negatives or {}

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        src_id, tgt_id, src_type, tgt_type = self.pairs[idx]
        
        src_name, src_addr, src_country = self.records.get(src_id, ("", "", ""))
        tgt_name, tgt_addr, tgt_country = self.records.get(tgt_id, ("", "", ""))

        item = {
            "src_id": src_id,
            "tgt_id": tgt_id,
            "src_type": src_type,
            "tgt_type": tgt_type,
            "src_name": src_name,
            "src_addr": src_addr,
            "src_country": src_country,
            "tgt_name": tgt_name,
            "tgt_addr": tgt_addr,
            "tgt_country": tgt_country,
        }

        if src_id in self.hard_negatives and len(self.hard_negatives[src_id]) > 0:
            neg_id = random.choice(self.hard_negatives[src_id])
            neg_name, neg_addr, neg_country = self.records.get(neg_id, ("", "", ""))
            item["neg_id"] = neg_id
            item["neg_name"] = neg_name
            item["neg_addr"] = neg_addr
            item["neg_country"] = neg_country
        else:
            item["neg_id"] = ""
            item["neg_name"] = ""
            item["neg_addr"] = ""
            item["neg_country"] = ""

        return item


def load_cross_source_training_pairs(
    gt_path: Path,
    val_s1_set: Set[str],
    max_entities: int = 50000,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str, str, str]], Set[str]]:
    """
    Extracts cross-source pairs (S1 <-> S2, S1 <-> S3, S2 <-> S3) strictly excluding val_s1_set.
    """
    random.seed(seed)
    pairs = []
    needed_ids = set()

    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        entity_count = 0
        for row in reader:
            s1_id = (row.get("source1_entity_id", "") or row.get("entity_id", "") or "").strip()
            if not s1_id or s1_id in val_s1_set:
                continue

            matches_raw = (row.get("matched_entity_ids", "") or row.get("match_ids", "") or "").strip()
            if not matches_raw:
                continue

            matches = [m.strip() for m in matches_raw.split(",") if m.strip()]
            s2_matches = [m for m in matches if m.startswith("S2")]
            s3_matches = [m for m in matches if m.startswith("S3")]

            # S1 <-> S2
            for s2 in s2_matches:
                pairs.append((s1_id, s2, "S1", "S2"))
                pairs.append((s2, s1_id, "S2", "S1"))
                needed_ids.add(s1_id)
                needed_ids.add(s2)

            # S1 <-> S3
            for s3 in s3_matches:
                pairs.append((s1_id, s3, "S1", "S3"))
                pairs.append((s3, s1_id, "S3", "S1"))
                needed_ids.add(s1_id)
                needed_ids.add(s3)

            # S2 <-> S3 (Crucial cross-source sibling translation pairs)
            for s2 in s2_matches:
                for s3 in s3_matches:
                    pairs.append((s2, s3, "S2", "S3"))
                    pairs.append((s3, s2, "S3", "S2"))
                    needed_ids.add(s2)
                    needed_ids.add(s3)

            entity_count += 1
            if entity_count >= max_entities:
                break

    print(f"[Dataset] Extracted {len(pairs):,} cross-source translation pairs from {entity_count:,} entities.")
    return pairs, needed_ids
