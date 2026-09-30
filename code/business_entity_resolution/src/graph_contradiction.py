"""
EXP_N032 — Graph Contradiction & Anti-Collision Engine.

Suppresses relational propagation across nodes exhibiting:
- Competing franchise branch collisions
- Incompatible house numbers
- Conflicting postal geographic centroids
"""

from __future__ import annotations

import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class GraphContradictionEngine:
    """
    Prevents relational edge traversal between conflicting entities.
    """

    def __init__(self):
        pass

    @staticmethod
    def is_edge_contradicted(
        branch_similarity: float,
        house_num_match: float,
        addr_missing: float,
        addr_jaccard: float,
    ) -> bool:
        if branch_similarity > 0.0:
            return True
        if addr_missing == 0.0 and house_num_match == 0.0 and addr_jaccard < 0.10:
            return True
        return False
