"""
Cross-Encoder Module — Pairwise reranking of ANN retrieval candidates.

A cross-encoder jointly encodes (record_A, record_B) as a single sequence and
predicts a match score in one forward pass, leveraging full attention across both records.

Design decisions:
- Uses the same multilingual MiniLM backbone as the bi-encoder.
- Input format: "[S1] {name_s1} | {addr_s1} [SEP] {name_cand} | {addr_cand}"
- Classification head (1 sigmoid output) for binary match probability.
- Batch inference only (no pair-by-pair forward), capped by VRAM budget.
- Cross-encoder is ONLY used for reranking the top-K ANN candidates, NOT for
  full-scan candidate generation (which would be O(N²) = infeasible).
- Designed to be trained with hard negatives from the ANN retrieval step.
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"     # confirmed working, 22M params
MULTILINGUAL_CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # fall back: multilingual model unavailable
RERANK_BATCH_SIZE = 128    # ~512MB VRAM at fp32; safe for RTX 4050 6GB
MAX_SEQ_LEN_CE = 256       # cross-encoder can use longer sequences than bi-encoder


def _format_cross_encoder_input(
    name_s1: str,
    addr_s1: str,
    name_cand: str,
    addr_cand: str,
) -> Tuple[str, str]:
    """
    Format (S1, candidate) record pair into (query, passage) for cross-encoder.

    Uses the cross-encoder convention: (query, document) pair as two separate strings,
    which cross-encoder libraries concatenate with [SEP] automatically.
    """
    s1_text = f"{(name_s1 or '').strip()} | {(addr_s1 or '').strip()}"
    cand_text = f"{(name_cand or '').strip()} | {(addr_cand or '').strip()}"
    return s1_text, cand_text


class CrossEncoderReranker:
    """
    Wrapper around sentence-transformers CrossEncoder for pairwise reranking.

    Usage:
        reranker = CrossEncoderReranker()
        scores = reranker.score_pairs(pair_texts)  # [(s1_text, cand_text), ...]
    """

    def __init__(
        self,
        model_name: str = MULTILINGUAL_CE_MODEL,
        device: Optional[str] = None,
        batch_size: int = RERANK_BATCH_SIZE,
        max_length: int = MAX_SEQ_LEN_CE,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
        self._device = device

    def _load_model(self):
        """Lazy-load the CrossEncoder on first call."""
        if self._model is not None:
            return

        try:
            from sentence_transformers.cross_encoder import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            ) from exc

        import torch
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[CrossEncoder] Loading '{self.model_name}' on {self._device}...")
        t0 = time.time()
        from sentence_transformers.cross_encoder import CrossEncoder
        self._model = CrossEncoder(
            self.model_name,
            device=self._device,
            max_length=self.max_length,
        )
        print(f"[CrossEncoder] Loaded in {time.time()-t0:.1f}s")

    def score_pairs(
        self,
        pairs: List[Tuple[str, str]],
        apply_sigmoid: bool = True,
    ) -> np.ndarray:
        """
        Score a list of (text_a, text_b) pairs.

        Args:
            pairs         : list of (s1_text, cand_text) tuples
            apply_sigmoid : apply sigmoid to get [0,1] match probability

        Returns:
            float32 array [N] of scores
        """
        self._load_model()
        if not pairs:
            return np.array([], dtype=np.float32)

        t0 = time.time()
        raw_scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            apply_softmax=False,
        )
        scores = raw_scores.astype(np.float32)

        if apply_sigmoid:
            scores = 1.0 / (1.0 + np.exp(-scores))

        print(f"[CrossEncoder] Scored {len(pairs):,} pairs in {time.time()-t0:.1f}s. "
              f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        return scores

    def rerank_candidates(
        self,
        s1_id: str,
        s1_name: str,
        s1_addr: str,
        candidates: List[Dict],  # [{cand_id, source, name, addr, bi_score}, ...]
    ) -> List[Dict]:
        """
        Rerank a list of candidates for a single S1 entity.

        Returns:
            Same list with 'ce_score' field added, sorted descending by ce_score.
        """
        if not candidates:
            return []

        pairs = [
            _format_cross_encoder_input(s1_name, s1_addr, c["name"], c["addr"])
            for c in candidates
        ]
        scores = self.score_pairs(pairs, apply_sigmoid=True)

        for i, cand in enumerate(candidates):
            cand["ce_score"] = float(scores[i])

        return sorted(candidates, key=lambda c: c["ce_score"], reverse=True)


def rerank_all_s1_candidates(
    reranker: CrossEncoderReranker,
    s1_records: Dict[str, Tuple],
    target_records: Dict[str, Tuple],
    ann_candidates: Dict[str, List[Tuple[str, float]]],
    top_k_before_rerank: int = 50,
    progress_every: int = 1000,
) -> Dict[str, List[Dict]]:
    """
    Rerank ANN retrieval candidates for all S1 entities using the cross-encoder.

    Args:
        s1_records           : {s1_id: (eid, name, addr, country)}
        target_records       : {cand_id: (eid, name, addr, country)}
        ann_candidates       : {s1_id: [(cand_id, bi_score), ...]}
        top_k_before_rerank  : number of top ANN candidates to rerank per S1
        progress_every       : print progress every N S1 entities

    Returns:
        {s1_id: [{"cand_id", "source", "name", "addr", "bi_score", "ce_score"}, ...]}
        Sorted descending by ce_score.
    """
    t0 = time.time()
    results = {}
    total = len(s1_records)

    # Build flat list of all pairs for batched cross-encoder inference
    pair_texts = []
    pair_meta = []  # (s1_id, cand_id, bi_score)

    for s1_id, (_, s1_name, s1_addr, _) in s1_records.items():
        cand_list = ann_candidates.get(s1_id, [])[:top_k_before_rerank]
        for cand_id, bi_score in cand_list:
            cand_rec = target_records.get(cand_id)
            if cand_rec is None:
                continue
            _, cand_name, cand_addr, _ = cand_rec
            pair_texts.append(_format_cross_encoder_input(s1_name, s1_addr, cand_name, cand_addr))
            pair_meta.append((s1_id, cand_id, bi_score))

    print(f"[CrossEncoder] Reranking {len(pair_texts):,} pairs for {total:,} S1 entities...")
    scores = reranker.score_pairs(pair_texts, apply_sigmoid=True)

    # Aggregate back to per-S1 structure
    s1_cand_map: Dict[str, List[Dict]] = {}
    for (s1_id, cand_id, bi_score), ce_score in zip(pair_meta, scores):
        cand_rec = target_records.get(cand_id, (None, "", "", ""))
        entry = {
            "cand_id": cand_id,
            "source": "S2" if cand_id.startswith("S2-") else "S3",
            "name": cand_rec[1] if len(cand_rec) > 1 else "",
            "addr": cand_rec[2] if len(cand_rec) > 2 else "",
            "bi_score": bi_score,
            "ce_score": float(ce_score),
        }
        s1_cand_map.setdefault(s1_id, []).append(entry)

    for s1_id in s1_cand_map:
        s1_cand_map[s1_id].sort(key=lambda c: c["ce_score"], reverse=True)

    # Ensure all S1 IDs present
    for s1_id in s1_records:
        results[s1_id] = s1_cand_map.get(s1_id, [])

    print(f"[CrossEncoder] Reranking complete in {time.time()-t0:.1f}s")
    return results
