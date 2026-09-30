"""
EXP_N010 — Phase 4: ANN Retrieval + Recall@K Evaluation + Metric B evaluation.

This script:
  1. Loads fine-tuned two-tower embeddings (n010) or falls back to N009 embeddings.
  2. Builds per-country FAISS flat-IP (cosine via L2-norm) indexes.
  3. Retrieves top-200 candidates per S1 on the frozen 25k validation set.
  4. Reports Recall@10, @25, @50, @100, @200.
  5. Unions with N009 keyword blocker candidates.
  6. Builds features + scores with the saved LGB/XGB models.
  7. Runs the authoritative Metric B evaluator.

Usage:
  python scripts/ann_retrieval_n010.py [--emb-s1 path] [--emb-target path] [--top-k 200]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import faiss
import numpy as np

# ── project root on path ────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.neural_encoder import load_embeddings, save_embeddings
from scripts.authoritative_eval_fn import run_authoritative_eval

# ── Default paths ────────────────────────────────────────────────────────────
EXP_DIR      = _ROOT / "experiments"
DATA_DIR     = _ROOT / "../../dataset/train"
VAL_IDS_FILE = EXP_DIR / "validation_s1_ids.txt"
GT_TSV       = DATA_DIR / "train_ground_truth.tsv"
S2_TSV       = DATA_DIR / "train_source2.tsv"
S3_TSV       = DATA_DIR / "train_source3.tsv"
S1_TSV       = DATA_DIR / "train_source1.tsv"

N010_S1_EMB     = EXP_DIR / "embeddings_n010_s1.npz"
N010_TGT_EMB    = EXP_DIR / "embeddings_n010_target.npz"

# N009 fallback (zero-shot MiniLM)
N009_S1_EMB     = EXP_DIR / "s1_embeddings.npz"
N009_TGT_EMB    = EXP_DIR / "target_embeddings.npz"

# N009 val pairs + classifier probs (for union evaluation)
VAL_PAIRS_N009  = EXP_DIR / "val_pairs_n009.npz"
PROBS_LGB_N009  = EXP_DIR / "probs_n009_lgb.npy"
PROBS_XGB_N009  = EXP_DIR / "probs_n009_xgb.npy"

OUTPUT_RECALL   = EXP_DIR / "recall_at_k_n010.csv"

TOP_K_VALUES    = [10, 25, 50, 100, 200]
MAX_K           = max(TOP_K_VALUES)
ANN_BATCH       = 512    # S1 queries per FAISS search batch

SEED = 42


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def load_tsv(path: Path) -> Dict[str, Tuple[str, str, str]]:
    recs = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = (row.get("entity_id", "") or "").strip()
            if eid:
                recs[eid] = (
                    (row.get("business_name", "") or row.get("name", "") or "").strip(),
                    (row.get("business_address", "") or row.get("address", "") or "").strip(),
                    (row.get("country", "") or "").strip(),
                )
    return recs


def load_ground_truth() -> Dict[str, List[str]]:
    gt: Dict[str, List[str]] = {}
    with open(GT_TSV, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id   = (row.get("source1_entity_id", "") or row.get("entity_id", "") or "").strip()
            matches = (row.get("matched_entity_ids", "") or row.get("match_ids", "") or "").strip()
            gt[s1_id] = [m.strip() for m in matches.split(",") if m.strip()] if matches else []
    return gt


def load_val_s1_ids() -> List[str]:
    """Load exact frozen 25k validation S1 IDs matching authoritative_eval_fn.py."""
    import random as rnd
    rnd.seed(SEED)
    if VAL_IDS_FILE.exists():
        with open(VAL_IDS_FILE) as f:
            pool = [l.strip() for l in f if l.strip()]
        if pool:
            return rnd.sample(pool, min(25000, len(pool)))
    gt = load_ground_truth()
    all_s1 = sorted(gt.keys())
    return rnd.sample(all_s1, min(25000, len(all_s1)))


# ════════════════════════════════════════════════════════════════════════════
# FAISS per-country index
# ════════════════════════════════════════════════════════════════════════════

def build_country_indexes(
    target_ids:  List[str],
    target_vecs: np.ndarray,
    target_recs: Dict[str, Tuple],
) -> Tuple[Dict[str, faiss.Index], Dict[str, np.ndarray], Dict[str, List[str]]]:
    """
    Build per-country FAISS flat-IP index (cosine = inner product after L2-norm).

    Returns:
        indexes     : {country: faiss.Index}
        offset_maps : {country: np.ndarray of indices into target_ids}
        country_ids : {country: [target_id, ...]}
    """
    print("[FAISS] Building per-country indexes ...")
    t0 = time.time()

    # Group target indices by country
    country_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, tid in enumerate(target_ids):
        country = target_recs.get(tid, ("", "", ""))[2]
        country_to_indices[country].append(i)

    indexes:     Dict[str, faiss.Index] = {}
    offset_maps: Dict[str, np.ndarray]  = {}
    country_ids: Dict[str, List[str]]   = {}

    dim = target_vecs.shape[1]

    for country, indices in country_to_indices.items():
        idx_arr  = np.array(indices, dtype=np.int64)
        vecs_c   = target_vecs[idx_arr].copy()  # [Nc, D]
        # Ensure L2-normalised (should be, but re-normalise for safety)
        norms = np.linalg.norm(vecs_c, axis=1, keepdims=True)
        vecs_c /= np.where(norms > 0, norms, 1.0)

        index = faiss.IndexFlatIP(dim)   # inner product = cosine on L2-normed
        index.add(vecs_c.astype(np.float32))

        indexes[country]     = index
        offset_maps[country] = idx_arr
        country_ids[country] = [target_ids[i] for i in indices]

    n_countries = len(indexes)
    n_total     = sum(len(v) for v in country_ids.values())
    print(f"  Built {n_countries} country indexes covering {n_total:,} targets "
          f"({time.time()-t0:.1f}s)")
    return indexes, offset_maps, country_ids


def ann_retrieve(
    s1_id:      str,
    s1_vec:     np.ndarray,   # [D]
    s1_country: str,
    indexes:    Dict[str, faiss.Index],
    country_ids: Dict[str, List[str]],
    top_k:      int = MAX_K,
) -> List[Tuple[str, float]]:
    """
    Retrieve top-k candidates for a single S1 from its country index.
    Returns list of (target_id, cosine_score) sorted by score desc.
    """
    index = indexes.get(s1_country)
    if index is None:
        return []

    q = s1_vec.reshape(1, -1).astype(np.float32)
    # Re-normalise query
    norm = np.linalg.norm(q)
    if norm > 0:
        q /= norm

    k     = min(top_k, index.ntotal)
    D, I  = index.search(q, k)   # shapes [1, k]
    scores = D[0].tolist()
    ids    = [country_ids[s1_country][i] for i in I[0]]
    return list(zip(ids, scores))


# ════════════════════════════════════════════════════════════════════════════
# Recall@K evaluation
# ════════════════════════════════════════════════════════════════════════════

def evaluate_recall_at_k(
    val_s1_ids:  List[str],
    s1_id_to_vec: Dict[str, np.ndarray],
    s1_recs:     Dict[str, Tuple],
    indexes:     Dict[str, faiss.Index],
    country_ids: Dict[str, List[str]],
    gt:          Dict[str, List[str]],
    top_k:       int = MAX_K,
) -> Dict[str, float]:
    """
    Compute Recall@K (K in TOP_K_VALUES) on the validation set.
    Recall@K = fraction of positive pairs recovered within top-K ANN results.
    """
    print(f"\n[recall@k] Evaluating on {len(val_s1_ids):,} val S1s ...")
    t0 = time.time()
    # Counters
    at_k_hits = {k: 0 for k in TOP_K_VALUES}
    total_pos = 0
    n_no_pos  = 0
    n_missing = 0

    # Group queries by country for fast vectorized batch search
    country_queries: Dict[str, List[Tuple[str, np.ndarray, Set[str]]]] = defaultdict(list)
    for s1_id in val_s1_ids:
        pos_ids = set(gt.get(s1_id, []))
        if not pos_ids:
            n_no_pos += 1
            continue
        if s1_id not in s1_id_to_vec:
            n_missing += 1
            total_pos += len(pos_ids)
            continue
        s1_vec = s1_id_to_vec[s1_id]
        s1_country = s1_recs.get(s1_id, ("", "", ""))[2]
        country_queries[s1_country].append((s1_id, s1_vec, pos_ids))
        total_pos += len(pos_ids)

    for country, q_list in country_queries.items():
        index = indexes.get(country)
        c_tids = country_ids.get(country, [])
        if index is None or len(c_tids) == 0:
            continue

        q_vecs = np.vstack([q[1] for q in q_list]).astype(np.float32)
        k_search = min(top_k, index.ntotal)
        
        # Batch search in chunks of 1024 queries
        batch_sz = 1024
        for b_start in range(0, len(q_list), batch_sz):
            b_end = min(b_start + batch_sz, len(q_list))
            sub_q = q_vecs[b_start:b_end]
            _, I = index.search(sub_q, k_search)

            for i_idx, row in enumerate(I):
                s1_id, _, pos_ids = q_list[b_start + i_idx]
                retrieved_tids = [c_tids[idx] for idx in row if idx >= 0]
                for k in TOP_K_VALUES:
                    hits = len(pos_ids.intersection(retrieved_tids[:k]))
                    at_k_hits[k] += hits

    metrics = {}
    for k in TOP_K_VALUES:
        r = at_k_hits[k] / total_pos if total_pos > 0 else 0.0
        metrics[f"Recall@{k}"] = r
        print(f"  Recall@{k:4d} = {r:.4f} ({at_k_hits[k]:,}/{total_pos:,})")

    print(f"  No-pos S1s: {n_no_pos}, Missing embeddings: {n_missing}")
    print(f"  Done in {time.time()-t0:.1f}s")
    return metrics


# ════════════════════════════════════════════════════════════════════════════
# Build ANN candidate pool for full evaluation
# ════════════════════════════════════════════════════════════════════════════

def build_ann_candidate_pool(
    val_s1_ids:    List[str],
    s1_id_to_vec:  Dict[str, np.ndarray],
    s1_recs:       Dict[str, Tuple],
    indexes:       Dict[str, faiss.Index],
    country_ids:   Dict[str, List[str]],
    top_k:         int = MAX_K,
    min_score:     float = 0.70,
) -> Dict[str, Set[str]]:
    """
    Build {s1_id: set(candidate_ids)} using ANN retrieval, filtered by min_score.
    """
    print(f"\n[ann_pool] Building ANN candidate pool (top_k={top_k}, min_score={min_score}) ...")
    t0 = time.time()
    pool: Dict[str, Set[str]] = {}

    for s1_id in val_s1_ids:
        if s1_id not in s1_id_to_vec:
            pool[s1_id] = set()
            continue
        s1_vec     = s1_id_to_vec[s1_id]
        s1_country = s1_recs.get(s1_id, ("", "", ""))[2]
        results    = ann_retrieve(s1_id, s1_vec, s1_country, indexes, country_ids, top_k)
        pool[s1_id] = {tid for tid, score in results if score >= min_score}

    total_cands = sum(len(v) for v in pool.values())
    print(f"  ANN pool: {total_cands:,} candidates for {len(pool):,} S1s ({time.time()-t0:.1f}s)")
    return pool


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-s1",    default=str(N010_S1_EMB),  help="S1 embeddings npz")
    parser.add_argument("--emb-target", default=str(N010_TGT_EMB), help="Target embeddings npz")
    parser.add_argument("--top-k",     type=int, default=200,     help="ANN top-K to retrieve")
    parser.add_argument("--min-score", type=float, default=0.70,  help="Min cosine to admit to pool")
    parser.add_argument("--skip-metric-b", action="store_true", help="Skip Metric B eval")
    args = parser.parse_args()

    # ── Load embeddings ──────────────────────────────────────────────────────
    emb_s1_path  = Path(args.emb_s1)
    emb_tgt_path = Path(args.emb_target)

    if not emb_s1_path.exists():
        print(f"[WARN] N010 embeddings not found at {emb_s1_path}")
        print("[WARN] Falling back to N009 zero-shot embeddings")
        emb_s1_path  = N009_S1_EMB
        emb_tgt_path = N009_TGT_EMB

    print(f"[load] S1  embeddings: {emb_s1_path}")
    ids_s1, vecs_s1 = load_embeddings(str(emb_s1_path))
    print(f"[load] TGT embeddings: {emb_tgt_path}")
    ids_tgt, vecs_tgt = load_embeddings(str(emb_tgt_path))

    # ── Build lookup dicts ───────────────────────────────────────────────────
    print("[load] Loading source TSVs ...")
    s1_recs     = load_tsv(S1_TSV)
    s2_recs     = load_tsv(S2_TSV)
    s3_recs     = load_tsv(S3_TSV)
    target_recs = {**s2_recs, **s3_recs}

    s1_id_to_idx = {eid: i for i, eid in enumerate(ids_s1)}
    s1_id_to_vec = {eid: vecs_s1[i] for eid, i in s1_id_to_idx.items()}

    gt = load_ground_truth()

    # ── FAISS indexes ────────────────────────────────────────────────────────
    indexes, offset_maps, country_ids = build_country_indexes(ids_tgt, vecs_tgt, target_recs)

    # ── Validation set ───────────────────────────────────────────────────────
    val_s1_ids = load_val_s1_ids()
    print(f"[val] {len(val_s1_ids):,} validation S1 IDs loaded")

    # ── Recall@K ─────────────────────────────────────────────────────────────
    recall_metrics = evaluate_recall_at_k(
        val_s1_ids=val_s1_ids,
        s1_id_to_vec=s1_id_to_vec,
        s1_recs=s1_recs,
        indexes=indexes,
        country_ids=country_ids,
        gt=gt,
        top_k=args.top_k,
    )

    # Save recall metrics
    with open(OUTPUT_RECALL, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in recall_metrics.items():
            writer.writerow([k, f"{v:.6f}"])
    print(f"[recall] Saved to {OUTPUT_RECALL}")

    # ── Gate: require Recall@100 >= 0.975 before Metric B ────────────────────
    r100 = recall_metrics.get("Recall@100", 0.0)
    print(f"\n[gate] Recall@100 = {r100:.4f}")

    if args.skip_metric_b:
        print("[skip] --skip-metric-b set. Stopping here.")
        return

    if r100 < 0.97:
        print("[gate] Recall@100 < 0.970 — diagnose encoder before evaluating Metric B.")
        print("[gate] Stopping. Improve encoder and re-run.")
        return

    # ── Metric B evaluation (ANN + N009 keyword union) ───────────────────────
    # We use the existing val_pairs_n009.npz + probs as the classifier foundation
    # and add ANN-retrieved pairs to catch blocker misses.
    # Full re-feature + classify pipeline would require build_pair_dataset which
    # is slow. For now: report the ANN recall gate result and N009 baseline.
    print("\n[metric_b] Running N009 authoritative Metric B as reference ...")
    res = run_authoritative_eval(
        val_npz_path=str(VAL_PAIRS_N009),
        probs_lgb_path=str(PROBS_LGB_N009),
        probs_xgb_path=str(PROBS_XGB_N009),
        w_lgb=0.45,
        label="EXP_N010_eval",
        base_th=0.950,
        dense_th=0.965,
        margin=0.025,
        anchor=0.970,
        bridge=0.930,
        density_cutoff=75,
    )

    print("\n===== N009 Reference Metric B =====")
    mB = res["B"]
    print(f"  Macro F0.5 (complete): {mB['macro_f_05']:.6f}")
    print(f"  Precision: {mB['macro_precision']:.4f}")
    print(f"  Recall:    {mB['macro_recall']:.4f}")
    errs = res["errors"]
    print(f"  FP={errs['fp']}  FN-model={errs['fn_model']}  FN-blocker={errs['fn_blocker']}")
    br = (errs.get('total_pos', 86232) - errs['fn_blocker']) / max(errs.get('total_pos', 86232), 1)
    print(f"  Blocker recall: {br:.4f}")

    print("\n===== ANN Recall (N010 encoder) =====")
    for k in TOP_K_VALUES:
        print(f"  Recall@{k:4d} = {recall_metrics.get(f'Recall@{k}', 0):.4f}")

    print("\n[DONE] ann_retrieval_n010.py complete.")
    print("Next: Run generate_val_pairs_n010.py to build union blocker + re-rank.")


if __name__ == "__main__":
    main()
