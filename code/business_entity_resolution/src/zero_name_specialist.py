"""
EXP_N030 — Zero-Name Specialist Module.

Specialist classifier dedicated strictly to candidate pairs with name_token_jaccard == 0:
- Uses address token and character overlap
- House number exact match
- Postal code prefix agreement
- Subword character n-gram similarities
- Dense neural embedding cosine
- Contradiction guard to prevent false positive branch admissions
"""

from __future__ import annotations

import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ZeroNameSpecialist:
    """
    Decides when a zero-token-overlap candidate has sufficient geographic and subword evidence
    to be safely admitted without triggering a branch false positive.
    """

    def __init__(self, min_address_jaccard: float = 0.50, min_neural_cosine: float = 0.75):
        self.min_address_jaccard = min_address_jaccard
        self.min_neural_cosine = min_neural_cosine

    def evaluate_rescue(
        self,
        name_jaccard: float,
        addr_jaccard: float,
        house_num_match: float,
        branch_similarity: float,
        neural_cosine: float,
        addr_missing: float,
        triangulation_score: float,
    ) -> bool:
        """
        Precision-preserving rescue rule:
        Only rescues if name_jaccard == 0, branch flag == 0, address is complete and matching,
        and neural cosine is high, or triangulation confirms it.
        """
        if name_jaccard > 0.0:
            return False
        if branch_similarity > 0.0:
            return False
        if addr_missing == 1.0:
            return False
            
        # Case A: Exact house number match with strong address overlap
        if house_num_match == 1.0 and addr_jaccard >= self.min_address_jaccard:
            return True
            
        # Case B: Cross-source triangulation confirmation
        if triangulation_score >= 1.50 and neural_cosine >= self.min_neural_cosine:
            return True
            
        return False
