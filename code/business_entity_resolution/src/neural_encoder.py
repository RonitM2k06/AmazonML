"""
Neural Encoder Module — Multilingual Sentence-Transformer based entity embedding.

Encodes business records as dense vectors using a compact multilingual transformer
(paraphrase-multilingual-MiniLM-L12-v2, 384-dim, ~117MB).

Design decisions:
- GPU batched inference with automatic CPU fallback.
- Record text format: "{name} | {address} | {country}" (pipe-separated, fits 128 tokens).
- Streaming encode: processes records in mini-batches to stay within 6GB VRAM budget.
- FAISS-compatible float32 L2-normalised output (cosine similarity via inner product).
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─────────────────────────────────────────────────────────────────────────────
# Model configuration
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384
MAX_SEQ_LEN = 128          # keeps context within tokenizer limit
ENCODE_BATCH_SIZE = 256    # ~640MB VRAM at fp32; safe for RTX 4050 6GB
SHOW_PROGRESS_BAR = False  # suppress sentence-transformers tqdm in scripts


def _format_record(name: str, address: str, country: str) -> str:
    """Convert a (name, address, country) triple to a fixed encoder input string."""
    name = (name or "").strip()
    address = (address or "").strip()
    country = (country or "").strip()
    parts = [p for p in [name, address, country] if p]
    return " | ".join(parts)[:512]  # hard-cap to avoid tokenizer OOM


class NeuralEncoder:
    """
    Wrapper around a SentenceTransformer for batch-encoding business records.

    Usage:
        encoder = NeuralEncoder()
        vecs = encoder.encode_records(records_dict)  # {id: (eid, name, addr, country)}
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        batch_size: int = ENCODE_BATCH_SIZE,
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self._model = None
        self._device = device  # None = auto-detect

    def _load_model(self):
        """Lazy-load the SentenceTransformer model on first call."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            ) from exc

        import torch

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[NeuralEncoder] Loading model '{self.model_name}' on {self._device}...")
        t0 = time.time()
        self._model = SentenceTransformer(self.model_name, device=self._device)
        print(f"[NeuralEncoder] Model loaded in {time.time()-t0:.1f}s. "
              f"Embedding dim: {self._model.get_sentence_embedding_dimension()}")

    @property
    def device(self) -> str:
        self._load_model()
        return self._device

    @property
    def embedding_dim(self) -> int:
        self._load_model()
        return self._model.get_sentence_embedding_dimension()

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of text strings.

        Returns:
            float32 array of shape [N, D] — L2-normalised if normalize_embeddings=True.
        """
        self._load_model()
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=SHOW_PROGRESS_BAR,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    def encode_records(
        self,
        records: Dict[str, Tuple],
        progress_every: int = 50000,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Encode a dict of records {id: (eid, name, addr, country)}.

        Returns:
            ids   : list of record IDs, same order as rows
            vecs  : float32 ndarray [N, D]
        """
        self._load_model()
        t0 = time.time()

        ids = list(records.keys())
        texts = []
        for rec_id in ids:
            rec = records[rec_id]
            # rec = (eid, name, addr, country) — index 1,2,3
            name = rec[1] if len(rec) > 1 else ""
            addr = rec[2] if len(rec) > 2 else ""
            country = rec[3] if len(rec) > 3 else ""
            texts.append(_format_record(name, addr, country))

        N = len(texts)
        print(f"[NeuralEncoder] Encoding {N:,} records in batches of {self.batch_size}...")

        all_vecs = []
        for start in range(0, N, self.batch_size):
            batch = texts[start : start + self.batch_size]
            vecs = self._model.encode(
                batch,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
            )
            all_vecs.append(vecs.astype(np.float32))
            if (start + self.batch_size) % progress_every < self.batch_size:
                pct = min(100, (start + self.batch_size) / N * 100)
                print(f"  {start + self.batch_size:,}/{N:,} ({pct:.1f}%) — "
                      f"{time.time()-t0:.0f}s elapsed")

        out = np.vstack(all_vecs) if all_vecs else np.zeros((0, self.embedding_dim), dtype=np.float32)
        print(f"[NeuralEncoder] Encoded {N:,} records → shape {out.shape} in {time.time()-t0:.1f}s")
        return ids, out

    def encode_pair_texts(
        self,
        s1_texts: List[str],
        cand_texts: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode parallel lists of S1 and candidate texts.
        Returns two float32 arrays [N, D] each.
        """
        combined = s1_texts + cand_texts
        vecs = self.encode_texts(combined)
        n = len(s1_texts)
        return vecs[:n], vecs[n:]


def save_embeddings(
    path: str,
    ids: List[str],
    vecs: np.ndarray,
    model_name: str = DEFAULT_MODEL_NAME,
):
    """Save embeddings and ID list to a compressed npz file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path,
        ids=np.array(ids, dtype=object),
        vecs=vecs,
        model_name=np.array([model_name]),
    )
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"[NeuralEncoder] Saved embeddings to {path} ({size_mb:.1f} MB)")


def load_embeddings(path: str) -> Tuple[List[str], np.ndarray]:
    """Load embeddings saved by save_embeddings()."""
    data = np.load(path, allow_pickle=True)
    ids = list(data["ids"])
    vecs = data["vecs"]
    print(f"[NeuralEncoder] Loaded embeddings from {path}: {len(ids):,} records, shape {vecs.shape}")
    return ids, vecs
