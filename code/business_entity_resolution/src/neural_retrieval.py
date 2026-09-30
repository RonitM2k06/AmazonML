"""
Neural Retrieval Module — FAISS-based Approximate Nearest-Neighbor (ANN) retrieval.

Builds a FAISS flat-IP index over S2+S3 target embeddings, partitioned by country,
and retrieves top-K nearest neighbors for each S1 query.

Design decisions:
- Country-partitioned index: zero cross-country candidates (safe per ground truth).
- IndexFlatIP for exact cosine search (L2-normalised vecs → inner product = cosine).
- Fallback to IndexFlatL2 if cosine preferred; both produce identical ranking when
  embeddings are L2-normalised.
- Memory budget: 10M targets × 384 float32 = ~14.4 GB → load per-country chunk.
- Streaming output: yields (s1_id, [(cand_id, score), ...]) to avoid materialising
  the full candidate matrix.
"""

import os
import time
from collections import defaultdict
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


class NeuralRetriever:
    """
    Country-partitioned FAISS retriever.

    Usage:
        retriever = NeuralRetriever(top_k=100)
        retriever.build_index(target_ids, target_vecs, target_countries)
        for s1_id, candidates in retriever.retrieve(s1_ids, s1_vecs, s1_countries):
            # candidates: List[Tuple[str, float]]  (cand_id, cosine_score)
    """

    def __init__(self, top_k: int = 100, use_gpu: bool = False):
        if not _FAISS_AVAILABLE:
            raise RuntimeError("faiss not installed. Run: pip install faiss-cpu")
        self.top_k = top_k
        self.use_gpu = use_gpu

        # Per-country index state
        self._country_index: Dict[str, faiss.IndexFlatIP] = {}
        self._country_ids: Dict[str, List[str]] = {}

    def build_index(
        self,
        target_ids: List[str],
        target_vecs: np.ndarray,
        target_countries: List[str],
        progress: bool = True,
    ):
        """
        Build FAISS IndexFlatIP per country.

        Args:
            target_ids      : list of N target record IDs
            target_vecs     : float32 [N, D] L2-normalised embeddings
            target_countries: list of N country strings (same order)
        """
        t0 = time.time()
        dim = target_vecs.shape[1]

        # Group by country
        country_groups: Dict[str, List[int]] = defaultdict(list)
        for i, c in enumerate(target_countries):
            country_groups[c].append(i)

        if progress:
            print(f"[NeuralRetriever] Building FAISS index for {len(target_ids):,} targets "
                  f"across {len(country_groups)} countries, dim={dim}...")

        for country, indices in sorted(country_groups.items()):
            vecs_c = target_vecs[indices]
            ids_c = [target_ids[i] for i in indices]

            index = faiss.IndexFlatIP(dim)
            if self.use_gpu:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)

            # Ensure F-contiguous float32
            vecs_c = np.ascontiguousarray(vecs_c, dtype=np.float32)
            index.add(vecs_c)

            self._country_index[country] = index
            self._country_ids[country] = ids_c

            if progress:
                print(f"  [{country}] {len(ids_c):,} targets indexed")

        if progress:
            print(f"[NeuralRetriever] Index built in {time.time()-t0:.1f}s")

    def retrieve(
        self,
        s1_ids: List[str],
        s1_vecs: np.ndarray,
        s1_countries: List[str],
        batch_size: int = 512,
        progress_every: int = 10000,
    ) -> Generator[Tuple[str, List[Tuple[str, float]]], None, None]:
        """
        Retrieve top-K candidates for each S1 query.

        Yields:
            (s1_id, [(cand_id, cosine_score), ...])
            — sorted descending by cosine score.
        """
        t0 = time.time()
        N = len(s1_ids)

        # Group S1 by country for batched search
        country_query_groups: Dict[str, List[int]] = defaultdict(list)
        for i, c in enumerate(s1_countries):
            country_query_groups[c].append(i)

        # Collect all results then yield in original order for determinism
        results: Dict[str, List[Tuple[str, float]]] = {}

        processed = 0
        for country, query_indices in sorted(country_query_groups.items()):
            if country not in self._country_index:
                # No targets for this country — return empty
                for i in query_indices:
                    results[s1_ids[i]] = []
                continue

            index = self._country_index[country]
            cand_ids = self._country_ids[country]
            k = min(self.top_k, index.ntotal)

            for batch_start in range(0, len(query_indices), batch_size):
                batch_idx = query_indices[batch_start : batch_start + batch_size]
                q_vecs = np.ascontiguousarray(
                    s1_vecs[batch_idx], dtype=np.float32
                )
                scores, faiss_ids = index.search(q_vecs, k)

                for j, orig_i in enumerate(batch_idx):
                    cands = []
                    for rank in range(k):
                        fid = faiss_ids[j, rank]
                        if fid < 0 or fid >= len(cand_ids):
                            continue
                        cands.append((cand_ids[fid], float(scores[j, rank])))
                    results[s1_ids[orig_i]] = cands

                processed += len(batch_idx)
                if processed % progress_every < batch_size:
                    print(f"[NeuralRetriever] Retrieved {processed:,}/{N:,} — "
                          f"{time.time()-t0:.0f}s elapsed")

        # Yield in original S1 order
        for s1_id in s1_ids:
            yield s1_id, results.get(s1_id, [])

    def save_index(self, directory: str):
        """Persist all per-country FAISS indices to disk."""
        os.makedirs(directory, exist_ok=True)
        for country, index in self._country_index.items():
            safe = country.replace(" ", "_").replace("/", "_")
            idx_path = os.path.join(directory, f"index_{safe}.faiss")
            ids_path = os.path.join(directory, f"ids_{safe}.npy")

            cpu_index = faiss.index_gpu_to_cpu(index) if self.use_gpu else index
            faiss.write_index(cpu_index, idx_path)
            np.save(ids_path, np.array(self._country_ids[country], dtype=object))

        print(f"[NeuralRetriever] Saved {len(self._country_index)} country indices to {directory}")

    def load_index(self, directory: str):
        """Load previously saved per-country FAISS indices from disk."""
        import glob
        self._country_index = {}
        self._country_ids = {}
        for idx_path in sorted(glob.glob(os.path.join(directory, "index_*.faiss"))):
            safe = os.path.basename(idx_path)[len("index_"):-len(".faiss")]
            ids_path = os.path.join(directory, f"ids_{safe}.npy")
            index = faiss.read_index(idx_path)
            ids = list(np.load(ids_path, allow_pickle=True))
            self._country_index[safe] = index
            self._country_ids[safe] = ids

        print(f"[NeuralRetriever] Loaded {len(self._country_index)} country indices from {directory}")


def evaluate_retrieval_recall(
    retriever_results: Dict[str, List[Tuple[str, float]]],
    ground_truth: Dict[str, List[str]],
    ks: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute Recall@K for the ANN retrieval step.

    Args:
        retriever_results : {s1_id: [(cand_id, score), ...]} top-K ranked list
        ground_truth      : {s1_id: [matched_cand_id, ...]}
        ks                : list of K values to evaluate (default: [10, 25, 50, 100, 200])

    Returns:
        Dict with recall@K for each K, plus miss count and total true pairs.
    """
    if ks is None:
        ks = [10, 25, 50, 100, 200]

    total_true = 0
    found_at = {k: 0 for k in ks}
    total_s1 = 0

    for s1_id, true_list in ground_truth.items():
        true_set = set(true_list)
        if not true_set:
            continue  # skip singletons
        total_s1 += 1
        total_true += len(true_set)

        retrieved = [cid for cid, _ in retriever_results.get(s1_id, [])]

        for k in ks:
            top_k_set = set(retrieved[:k])
            found_at[k] += len(true_set.intersection(top_k_set))

    metrics = {
        "total_s1_with_matches": total_s1,
        "total_true_pairs": total_true,
    }
    for k in ks:
        recall = found_at[k] / total_true if total_true > 0 else 0.0
        metrics[f"recall@{k}"] = round(recall, 6)

    return metrics
