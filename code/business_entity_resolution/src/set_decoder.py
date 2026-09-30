"""
EXP_N028 — Global Set Decoder.

Decodes the optimal candidate subset conditioned on:
- Global cardinality distribution P(K matches)
- Candidate match probabilities
- Precision-safety guards against branch/same-chain conflict
"""

from __future__ import annotations

import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class GlobalSetDecoder:
    """
    Decodes predicted candidate sets conditioned on cardinality and coherence.
    """

    def __init__(self, precision_gate_threshold: float = 0.90):
        self.precision_gate_threshold = precision_gate_threshold

    @staticmethod
    def decode_cardinality_conditioned(
        match_probs: np.ndarray,      # [N, K]
        card_probs: np.ndarray,       # [N, max_k + 1]
        base_accepted: np.ndarray,    # [N, K] bool
        max_k_admit: int = 8,
    ) -> np.ndarray:
        """
        Conditions set selection on the most likely non-zero cardinality:
        If model strongly predicts k matches, admits top-k candidates provided they pass
        the local confidence floor.
        """
        N, K = match_probs.shape
        admitted = np.copy(base_accepted)
        
        # Predicted expected cardinality
        k_weights = np.arange(card_probs.shape[1])
        expected_k = np.sum(card_probs * k_weights[None, :], axis=1)
        
        for i in range(N):
            exp_k = int(np.round(expected_k[i]))
            if exp_k > 0:
                current_admit_count = np.sum(admitted[i])
                if current_admit_count < exp_k:
                    needed = min(exp_k - current_admit_count, max_k_admit)
                    unadmitted_ranks = np.where(~admitted[i])[0]
                    # Sort unadmitted by match probability
                    sorted_unadmitted = unadmitted_ranks[np.argsort(-match_probs[i, unadmitted_ranks])]
                    for r in sorted_unadmitted[:needed]:
                        if match_probs[i, r] >= 0.75:
                            admitted[i, r] = True
        return admitted
