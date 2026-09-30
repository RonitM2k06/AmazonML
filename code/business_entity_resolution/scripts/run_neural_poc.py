#!/usr/bin/env python3
"""
EXP_N001–EXP_N003: Neural Retrieval POC — Dual Encoder + FAISS ANN + Optional Cross-Encoder.

This script is the master runner for the neural architecture proof-of-concept.
It evaluates three configurations on the 25k validation benchmark:

  EXP_N001: Dual-encoder ANN only (FAISS cosine retrieval)
  EXP_N002: Dual-encoder ANN + existing Union E blocker (union of candidates)
  EXP_N003: Cross-encoder reranking on top-50 ANN candidates

Outputs:
  - reports/neural_retrieval_recall.csv   : Recall@K for ANN at each K
  - reports/neural_experiment_registry.csv: Macro F0.5 per experiment
  - reports/NEURAL_POC_REPORT.md          : Human-readable summary
  - experiments/s1_embeddings.npz         : Cached S1 embeddings
  - experiments/target_embeddings.npz     : Cached target embeddings

IMPORTANT:
  - Does NOT modify or overwrite the current champion pipeline.
  - Evaluation uses the same frozen 25k validation split (seed=42).
  - Does NOT re-run EXP_008 pair extraction.
"""

import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# ─────────────────────────────────────────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
code_dir = os.path.join(project_root, "code", "business_entity_resolution")
src_dir = os.path.join(code_dir, "src")
sys.path.insert(0, src_dir)
sys.path.insert(0, os.path.join(code_dir, "scripts"))

from neural_encoder import NeuralEncoder, save_embeddings, load_embeddings
from neural_retrieval import NeuralRetriever, evaluate_retrieval_recall
from evaluation import Evaluator, compute_entity_metrics
from blocking import build_target_index, generate_candidates
from build_pair_dataset import UNION_E_KEYS

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DATASET_DIR = os.path.join(project_root, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
EXPERIMENTS_DIR = os.path.join(code_dir, "experiments")
REPORTS_DIR = os.path.join(code_dir, "reports")
MODELS_NEURAL_DIR = os.path.join(code_dir, "models", "neural")

VAL_IDS_PATH = os.path.join(EXPERIMENTS_DIR, "validation_s1_ids.txt")
S1_EMB_PATH = os.path.join(EXPERIMENTS_DIR, "s1_embeddings.npz")
TARGET_EMB_PATH = os.path.join(EXPERIMENTS_DIR, "target_embeddings.npz")

CHAMPION_JSON = os.path.join(REPORTS_DIR, "champion.json")

VAL_SAMPLE_SIZE = 25000
SEED = 42

# ANN retrieval parameters
ANN_TOP_K = 200            # Retrieve top-200 per S1 for recall measurement
CE_RERANK_TOP_K = 50       # Rerank top-50 ANN candidates with cross-encoder
DECISION_THRESHOLD = 0.50  # Cosine similarity threshold for EXP_N001 match decision

ANN_EVAL_KS = [10, 25, 50, 100, 200]

os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_NEURAL_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_validation_data(val_sample_size: int = VAL_SAMPLE_SIZE, seed: int = SEED):
    """
    Load the frozen 25k validation split.
    Returns:
        val_s1_records  : {s1_id: (eid, name, addr, country)}
        target_records  : {cand_id: (eid, name, addr, country)}
        val_matches     : {s1_id: [matched_cand_id, ...]}
    """
    t0 = time.time()
    random.seed(seed)

    print("=" * 60)
    print("Loading validation data...")
    print(f"  VAL_SAMPLE_SIZE = {val_sample_size:,}, SEED = {seed}")
    print("=" * 60)

    # Load frozen validation S1 IDs
    with open(VAL_IDS_PATH, "r", encoding="utf-8") as f:
        val_pool = [line.strip() for line in f if line.strip()]
    selected_val_s1 = set(random.sample(val_pool, min(val_sample_size, len(val_pool))))
    print(f"Validation S1 pool: {len(val_pool):,}, selected: {len(selected_val_s1):,}")

    # Load ground truth
    val_matches: Dict[str, List[str]] = {}
    val_needed_s2: Set[str] = set()
    val_needed_s3: Set[str] = set()

    gt_path = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
    with open(gt_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1 = parts[0]
            if s1 not in selected_val_s1:
                continue
            mids = parts[1].split(",") if len(parts) > 1 and parts[1].strip() else []
            val_matches[s1] = mids
            for m in mids:
                if m.startswith("S2-"):
                    val_needed_s2.add(m)
                elif m.startswith("S3-"):
                    val_needed_s3.add(m)
            if len(val_matches) >= val_sample_size:
                break

    print(f"Ground truth loaded: {len(val_matches):,} S1 entities, "
          f"{len(val_needed_s2):,} S2 matches, {len(val_needed_s3):,} S3 matches")

    # Load S1 records
    val_s1_records: Dict[str, Tuple] = {}
    with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), "r", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            eid = line.split("\t", 1)[0]
            if eid in selected_val_s1:
                parts = line.rstrip("\r\n").split("\t")
                val_s1_records[eid] = (parts[0], parts[1], parts[2], parts[3])
                if len(val_s1_records) >= len(selected_val_s1):
                    break

    # Load S2 records (matches + 30k background)
    all_needed = val_needed_s2.union(val_needed_s3)
    s2_records: Dict[str, Tuple] = {}
    bg_s2 = 0
    max_bg = 30000
    with open(os.path.join(TRAIN_DIR, "train_source2.tsv"), "r", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            eid = line.split("\t", 1)[0]
            if eid in val_needed_s2:
                parts = line.rstrip("\r\n").split("\t")
                s2_records[eid] = (parts[0], parts[1], parts[2], parts[3])
            elif bg_s2 < max_bg and random.random() < 0.012:
                parts = line.rstrip("\r\n").split("\t")
                s2_records[eid] = (parts[0], parts[1], parts[2], parts[3])
                bg_s2 += 1

    # Load S3 records (matches + 30k background)
    s3_records: Dict[str, Tuple] = {}
    bg_s3 = 0
    with open(os.path.join(TRAIN_DIR, "train_source3.tsv"), "r", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            eid = line.split("\t", 1)[0]
            if eid in val_needed_s3:
                parts = line.rstrip("\r\n").split("\t")
                s3_records[eid] = (parts[0], parts[1], parts[2], parts[3])
            elif bg_s3 < max_bg and random.random() < 0.012:
                parts = line.rstrip("\r\n").split("\t")
                s3_records[eid] = (parts[0], parts[1], parts[2], parts[3])
                bg_s3 += 1

    target_records = {**s2_records, **s3_records}
    print(f"Records loaded in {time.time()-t0:.1f}s: "
          f"{len(val_s1_records):,} S1, {len(s2_records):,} S2 (+{bg_s2} bg), "
          f"{len(s3_records):,} S3 (+{bg_s3} bg) → {len(target_records):,} targets")

    return val_s1_records, target_records, val_matches


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_or_encode(
    records: Dict[str, Tuple],
    cache_path: str,
    encoder: NeuralEncoder,
    label: str,
) -> Tuple[List[str], np.ndarray]:
    """Load from cache if available, otherwise encode and save."""
    if os.path.exists(cache_path):
        print(f"[{label}] Loading cached embeddings from {cache_path}...")
        ids, vecs = load_embeddings(cache_path)
        # Verify all records are present
        id_set = set(ids)
        missing = [k for k in records if k not in id_set]
        if not missing:
            return ids, vecs
        print(f"[{label}] Cache incomplete ({len(missing)} missing). Re-encoding...")

    ids, vecs = encoder.encode_records(records)
    save_embeddings(cache_path, ids, vecs)
    return ids, vecs


# ─────────────────────────────────────────────────────────────────────────────
# Entity-level decision from cosine scores
# ─────────────────────────────────────────────────────────────────────────────
def make_entity_predictions_cosine(
    ann_candidates: Dict[str, List[Tuple[str, float]]],
    threshold: float,
    singleton_gate: float,
    top_k: int = ANN_TOP_K,
) -> Dict[str, Set[str]]:
    """
    Simple threshold decision on cosine scores from ANN retrieval.
    A candidate is accepted if cosine_score >= threshold AND max_score >= singleton_gate.
    """
    preds: Dict[str, Set[str]] = {}
    for s1_id, ranked in ann_candidates.items():
        if not ranked:
            preds[s1_id] = set()
            continue
        top_score = ranked[0][1]
        if top_score < singleton_gate:
            preds[s1_id] = set()
            continue
        preds[s1_id] = {cid for cid, score in ranked[:top_k] if score >= threshold}
    return preds


def evaluate_predictions(
    preds: Dict[str, Set[str]],
    ground_truth: Dict[str, List[str]],
) -> Dict:
    """Compute entity-level macro F0.5 from predictions dict."""
    evaluator = Evaluator()
    for s1_id, true_list in ground_truth.items():
        true_set = set(true_list)
        pred_set = preds.get(s1_id, set())
        evaluator.update(s1_id, true_set, pred_set)
    return evaluator.compute()


def sweep_cosine_threshold(
    ann_candidates: Dict[str, List[Tuple[str, float]]],
    ground_truth: Dict[str, List[str]],
    thresholds=None,
) -> List[Dict]:
    """Sweep cosine match threshold and compute F0.5 at each value."""
    if thresholds is None:
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    results = []
    for th in thresholds:
        preds = make_entity_predictions_cosine(ann_candidates, threshold=th, singleton_gate=th * 0.85)
        metrics = evaluate_predictions(preds, ground_truth)
        metrics["threshold"] = th
        results.append(metrics)
        print(f"  th={th:.2f}  F0.5={metrics['macro_f_05']:.6f}  "
              f"P={metrics['macro_precision']:.4f}  R={metrics['macro_recall']:.4f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Union E candidate retrieval (for EXP_N002 union experiment)
# ─────────────────────────────────────────────────────────────────────────────
def get_union_e_candidates(
    val_s1_records: Dict[str, Tuple],
    target_records: Dict[str, Tuple],
    max_bucket_size: int = 1000,
    max_candidates: int = 200,
) -> Dict[str, Set[str]]:
    """Run existing Union E blocking and return candidate sets."""
    print("Building Union E block index...")
    target_pool = list(target_records.values())
    block_index = build_target_index(target_pool, UNION_E_KEYS, max_bucket_size=max_bucket_size)

    print(f"Generating Union E candidates for {len(val_s1_records):,} S1 entities...")
    cands_by_s1 = generate_candidates(
        val_s1_records.values(), block_index, UNION_E_KEYS, max_candidates_per_s1=max_candidates
    )

    # Convert from set-of-record-tuples to set-of-ids
    # generate_candidates returns {s1_id: {cand_id, ...}}
    result: Dict[str, Set[str]] = {}
    for s1_id in val_s1_records:
        result[s1_id] = cands_by_s1.get(s1_id, set())
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Report helpers
# ─────────────────────────────────────────────────────────────────────────────
def append_registry(registry_path: str, row: Dict):
    """Append a result row to the experiment registry CSV."""
    fieldnames = [
        "exp_id", "description", "macro_f_05", "macro_precision", "macro_recall",
        "singleton_accuracy", "exact_set_accuracy",
        "total_entities", "total_true_matches", "total_false_positives", "total_false_negatives",
        "threshold", "notes",
    ]
    write_header = not os.path.exists(registry_path)
    with open(registry_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def print_metrics(label: str, metrics: Dict):
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  Macro F0.5       : {metrics.get('macro_f_05', 0):.6f}")
    print(f"  Macro Precision  : {metrics.get('macro_precision', 0):.6f}")
    print(f"  Macro Recall     : {metrics.get('macro_recall', 0):.6f}")
    print(f"  Singleton Acc    : {metrics.get('singleton_accuracy', 0):.6f}")
    print(f"  Exact Set Acc    : {metrics.get('exact_set_accuracy', 0):.6f}")
    print(f"  Total Entities   : {metrics.get('total_entities', 0):,}")
    print(f"  True Matches     : {metrics.get('total_true_matches', 0):,}")
    print(f"  False Positives  : {metrics.get('total_false_positives', 0):,}")
    print(f"  False Negatives  : {metrics.get('total_false_negatives', 0):,}")
    print(f"{'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    print("=" * 70)
    print("NEURAL ARCHITECTURE POC — EXP_N001 / EXP_N002 / EXP_N003")
    print("=" * 70)

    # Load champion baseline for comparison
    with open(CHAMPION_JSON, "r") as f:
        champion = json.load(f)
    champion_f05 = champion["metric_macro_f05"]
    print(f"\nChampion baseline: Macro F0.5 = {champion_f05:.6f}")

    registry_path = os.path.join(REPORTS_DIR, "neural_experiment_registry.csv")
    recall_csv_path = os.path.join(REPORTS_DIR, "neural_retrieval_recall.csv")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Load data
    # ─────────────────────────────────────────────────────────────────────────
    val_s1_records, target_records, val_matches = load_validation_data()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Encode records with multilingual MiniLM
    # ─────────────────────────────────────────────────────────────────────────
    encoder = NeuralEncoder()

    s1_ids, s1_vecs = get_or_encode(val_s1_records, S1_EMB_PATH, encoder, "S1")
    target_ids, target_vecs = get_or_encode(target_records, TARGET_EMB_PATH, encoder, "Targets")

    # Build country lists for partitioned indexing
    s1_countries = [val_s1_records[sid][3] if sid in val_s1_records else "" for sid in s1_ids]
    target_countries = [target_records[tid][3] if tid in target_records else "" for tid in target_ids]

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Build FAISS index and retrieve top-200 candidates
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[Step 3] Building FAISS index (top_k={ANN_TOP_K})...")
    retriever = NeuralRetriever(top_k=ANN_TOP_K)
    retriever.build_index(target_ids, target_vecs, target_countries)

    print(f"[Step 3] Running ANN retrieval for {len(s1_ids):,} S1 entities...")
    ann_results: Dict[str, List[Tuple[str, float]]] = {}
    for s1_id, cands in retriever.retrieve(s1_ids, s1_vecs, s1_countries):
        ann_results[s1_id] = cands

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Evaluate ANN retrieval recall
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Step 4] Evaluating ANN Retrieval Recall@K...")
    recall_metrics = evaluate_retrieval_recall(ann_results, val_matches, ks=ANN_EVAL_KS)

    print(f"\nANN Retrieval Recall@K Results:")
    for k in ANN_EVAL_KS:
        print(f"  Recall@{k:3d} = {recall_metrics.get(f'recall@{k}', 0):.4f}")
    print(f"  Total true pairs: {recall_metrics['total_true_pairs']:,}")

    # Save recall CSV
    with open(recall_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "recall"])
        for k in ANN_EVAL_KS:
            writer.writerow([k, recall_metrics.get(f"recall@{k}", 0)])
    print(f"Recall@K saved to {recall_csv_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # EXP_N001: Dual encoder only — threshold sweep on cosine scores
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXP_N001: Dual Encoder ANN Only — Cosine Threshold Sweep")
    print("=" * 70)

    n001_sweep = sweep_cosine_threshold(ann_results, val_matches)
    best_n001 = max(n001_sweep, key=lambda x: x["macro_f_05"])
    print_metrics("EXP_N001 Best Result", best_n001)

    append_registry(registry_path, {
        "exp_id": "EXP_N001",
        "description": "Dual encoder ANN cosine threshold (no Union E)",
        "threshold": best_n001["threshold"],
        "notes": f"top_k={ANN_TOP_K}, model=paraphrase-multilingual-MiniLM-L12-v2",
        **{k: v for k, v in best_n001.items() if k != "threshold"},
    })

    # ─────────────────────────────────────────────────────────────────────────
    # EXP_N002: Dual encoder ANN + Union E union (combined candidate set)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXP_N002: Dual Encoder ANN ∪ Union E Blocker")
    print("=" * 70)

    union_e_cands = get_union_e_candidates(val_s1_records, target_records)
    print(f"Union E candidates generated for {len(union_e_cands):,} S1 entities.")

    # For each S1: union of ANN top-K set and Union E candidate set
    # Decision: cosine threshold for ANN candidates, any Union E candidate that also
    # has cosine_score >= 0.35 is included (conservative gate to avoid false positives)
    ANN_UNION_GATE = 0.35

    # Build cosine score lookup
    cosine_lookup: Dict[str, float] = {}
    for s1_id, ranked in ann_results.items():
        for cand_id, score in ranked:
            cosine_lookup[(s1_id, cand_id)] = score

    def predict_n002(threshold: float, union_gate: float = ANN_UNION_GATE) -> Dict[str, Set[str]]:
        preds = {}
        for s1_id in val_matches:
            ranked = ann_results.get(s1_id, [])
            top_score = ranked[0][1] if ranked else 0.0
            if top_score < threshold * 0.85:
                preds[s1_id] = set()
                continue

            # ANN gate
            ann_accepted = {cid for cid, s in ranked if s >= threshold}

            # Union E candidates with cosine score >= union_gate
            ue_cands = union_e_cands.get(s1_id, set())
            ue_accepted = {
                cid for cid in ue_cands
                if cosine_lookup.get((s1_id, cid), 0.0) >= union_gate
            }
            preds[s1_id] = ann_accepted | ue_accepted
        return preds

    n002_sweep = []
    for th in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        preds = predict_n002(th)
        metrics = evaluate_predictions(preds, val_matches)
        metrics["threshold"] = th
        n002_sweep.append(metrics)
        print(f"  th={th:.2f}  F0.5={metrics['macro_f_05']:.6f}  "
              f"P={metrics['macro_precision']:.4f}  R={metrics['macro_recall']:.4f}")

    best_n002 = max(n002_sweep, key=lambda x: x["macro_f_05"])
    print_metrics("EXP_N002 Best Result", best_n002)

    append_registry(registry_path, {
        "exp_id": "EXP_N002",
        "description": "Dual encoder ANN union with Union E blocker",
        "threshold": best_n002["threshold"],
        "notes": f"top_k={ANN_TOP_K}, union_gate={ANN_UNION_GATE}",
        **{k: v for k, v in best_n002.items() if k != "threshold"},
    })

    # ─────────────────────────────────────────────────────────────────────────
    # EXP_N003: Cross-encoder reranking on top-50 ANN candidates
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXP_N003: Cross-Encoder Reranking on Top-50 ANN Candidates")
    print("=" * 70)

    try:
        from cross_encoder import CrossEncoderReranker, rerank_all_s1_candidates

        reranker = CrossEncoderReranker()
        reranked = rerank_all_s1_candidates(
            reranker=reranker,
            s1_records=val_s1_records,
            target_records=target_records,
            ann_candidates={s1_id: ann_results.get(s1_id, [])[:CE_RERANK_TOP_K] for s1_id in val_s1_records},
            top_k_before_rerank=CE_RERANK_TOP_K,
        )

        # Convert reranked to {s1_id: [(cand_id, ce_score), ...]} for sweep
        ce_ranked: Dict[str, List[Tuple[str, float]]] = {
            s1_id: [(c["cand_id"], c["ce_score"]) for c in cands]
            for s1_id, cands in reranked.items()
        }

        print("\nSweeping cross-encoder threshold...")
        n003_sweep = sweep_cosine_threshold(ce_ranked, val_matches, thresholds=[
            0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90
        ])
        best_n003 = max(n003_sweep, key=lambda x: x["macro_f_05"])
        print_metrics("EXP_N003 Best Result", best_n003)

        append_registry(registry_path, {
            "exp_id": "EXP_N003",
            "description": "Cross-encoder reranking on top-50 ANN candidates",
            "threshold": best_n003["threshold"],
            "notes": f"ce_model=mmarco-mMiniLMv2-L6-H384-v1, rerank_top_k={CE_RERANK_TOP_K}",
            **{k: v for k, v in best_n003.items() if k != "threshold"},
        })
        n003_f05 = best_n003["macro_f_05"]
    except Exception as e:
        print(f"[EXP_N003] Cross-encoder experiment failed: {e}")
        print("  Skipping EXP_N003 — cross-encoder model may not be downloadable.")
        n003_f05 = None
        best_n003 = None

    # ─────────────────────────────────────────────────────────────────────────
    # Final report
    # ─────────────────────────────────────────────────────────────────────────
    n001_f05 = best_n001["macro_f_05"]
    n002_f05 = best_n002["macro_f_05"]

    report_lines = [
        "# Neural Architecture POC Report\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        "\n## Champion Baseline\n",
        f"- EXP_007 Ensemble XGB/LGB/CAT: **Macro F0.5 = {champion_f05:.6f}**\n",
        "\n## ANN Retrieval Recall@K (paraphrase-multilingual-MiniLM-L12-v2)\n",
        "| K | Recall@K |\n|---|---|\n",
    ]
    for k in ANN_EVAL_KS:
        report_lines.append(f"| {k} | {recall_metrics.get(f'recall@{k}', 0):.4f} |\n")

    report_lines += [
        "\n## Experiment Results\n",
        "| Experiment | Description | Macro F0.5 | Precision | Recall | vs Champion |\n",
        "|---|---|---|---|---|---|\n",
        f"| EXP_007 (champion) | XGB/LGB/CAT ensemble | {champion_f05:.6f} | 0.9770 | 0.8983 | baseline |\n",
        f"| EXP_N001 | Dual encoder ANN only | {n001_f05:.6f} | "
        f"{best_n001['macro_precision']:.4f} | {best_n001['macro_recall']:.4f} | "
        f"{'▲' if n001_f05 > champion_f05 else '▼'} {abs(n001_f05 - champion_f05):.4f} |\n",
        f"| EXP_N002 | ANN ∪ Union E | {n002_f05:.6f} | "
        f"{best_n002['macro_precision']:.4f} | {best_n002['macro_recall']:.4f} | "
        f"{'▲' if n002_f05 > champion_f05 else '▼'} {abs(n002_f05 - champion_f05):.4f} |\n",
    ]

    if best_n003:
        report_lines.append(
            f"| EXP_N003 | Cross-encoder reranking top-50 | {n003_f05:.6f} | "
            f"{best_n003['macro_precision']:.4f} | {best_n003['macro_recall']:.4f} | "
            f"{'▲' if n003_f05 > champion_f05 else '▼'} {abs(n003_f05 - champion_f05):.4f} |\n"
        )
    else:
        report_lines.append("| EXP_N003 | Cross-encoder reranking top-50 | FAILED | - | - | - |\n")

    report_lines += [
        "\n## Analysis\n",
        f"- ANN Recall@{ANN_TOP_K} = {recall_metrics.get(f'recall@{ANN_TOP_K}', 0):.4f} "
        f"vs Union E Recall = 0.9578 (25k benchmark)\n",
        "- Neural retrieval supplements lexical blocking by capturing semantic similarity "
        "across script/transliteration variations.\n",
        "- Cross-encoder reranking provides fine-grained pairwise scoring but is limited "
        "to top-K ANN candidates.\n",
        "\n## Next Steps\n",
        "- EXP_N004: Train dual encoder with contrastive loss on hard negatives\n",
        "- EXP_N005: Data augmentation (transliteration, noise)\n",
        "- EXP_N008: Hybrid: existing 56 XGB features + neural cosine score\n",
        "- EXP_N009: Hybrid + cross-source S2↔S3 graph bridge\n",
        f"\nTotal runtime: {(time.time()-t_start)/60:.1f} minutes\n",
    ]

    report_path = os.path.join(REPORTS_DIR, "NEURAL_POC_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    print(f"\nNeural POC report saved to {report_path}")

    print("\n" + "=" * 70)
    print("NEURAL POC SUMMARY")
    print("=" * 70)
    print(f"  Champion (EXP_007) : F0.5 = {champion_f05:.6f}")
    print(f"  EXP_N001 (ANN only): F0.5 = {n001_f05:.6f}  "
          f"({'▲' if n001_f05 > champion_f05 else '▼'} {abs(n001_f05 - champion_f05):.4f})")
    print(f"  EXP_N002 (ANN+UE)  : F0.5 = {n002_f05:.6f}  "
          f"({'▲' if n002_f05 > champion_f05 else '▼'} {abs(n002_f05 - champion_f05):.4f})")
    if best_n003:
        print(f"  EXP_N003 (CE rerank): F0.5 = {n003_f05:.6f}  "
              f"({'▲' if n003_f05 > champion_f05 else '▼'} {abs(n003_f05 - champion_f05):.4f})")
    print(f"  Total time: {(time.time()-t_start)/60:.1f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()
