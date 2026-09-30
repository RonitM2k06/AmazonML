#!/usr/bin/env python3
"""
Fast vectorized bridge threshold sweep for EXP_N008.

Loads the already-generated val_pairs_n008.npz and pre-computed model probabilities,
then runs the full transitive bridge parameter sweep in a vectorized fashion.

This is 100-1000x faster than the nested Python loop in run_exp_n008_hybrid.py
because it pre-groups candidates by S1 and uses NumPy operations.

Saves final results to:
  - reports/neural_experiment_registry.csv (appended)
  - reports/champion.json (updated if improved)
  - experiments/probs_n008_lgb.npy
  - experiments/probs_n008_xgb.npy
"""

import csv
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["PYTHONUNBUFFERED"] = "1"

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
code_dir = os.path.join(project_root, "code", "business_entity_resolution")
src_dir = os.path.join(code_dir, "src")
sys.path.insert(0, src_dir)
sys.path.insert(0, os.path.join(code_dir, "scripts"))

from evaluation import Evaluator

EXPERIMENTS_DIR = os.path.join(code_dir, "experiments")
REPORTS_DIR = os.path.join(code_dir, "reports")
MODELS_DIR = os.path.join(code_dir, "models")

VAL_NPZ_N008 = os.path.join(EXPERIMENTS_DIR, "val_pairs_n008.npz")
TRAIN_NPZ_EXP008 = os.path.join(EXPERIMENTS_DIR, "train_pairs_exp008.npz")
CHAMPION_JSON = os.path.join(REPORTS_DIR, "champion.json")
REGISTRY_PATH = os.path.join(REPORTS_DIR, "neural_experiment_registry.csv")
PROBS_LGB_PATH = os.path.join(EXPERIMENTS_DIR, "probs_n008_lgb.npy")
PROBS_XGB_PATH = os.path.join(EXPERIMENTS_DIR, "probs_n008_xgb.npy")

SEED = 42
VAL_SAMPLE_SIZE = 25000
NEURAL_FEATURE_NAME = "neural_cosine_sim"


def load_val_data():
    """Load val pairs, metadata and ground truth."""
    import random
    random.seed(SEED)
    TRAIN_DIR = os.path.join(project_root, "dataset", "train")
    VAL_IDS_PATH = os.path.join(EXPERIMENTS_DIR, "validation_s1_ids.txt")

    with open(VAL_IDS_PATH, "r", encoding="utf-8") as f:
        val_pool = [l.strip() for l in f if l.strip()]
    selected_val_s1 = set(random.sample(val_pool, min(VAL_SAMPLE_SIZE, len(val_pool))))

    val_matches = {}
    with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1 = parts[0]
            if s1 not in selected_val_s1:
                continue
            mids = parts[1].split(",") if len(parts) > 1 and parts[1].strip() else []
            val_matches[s1] = mids
            if len(val_matches) >= VAL_SAMPLE_SIZE:
                break

    return val_matches


def build_or_load_models_and_probs(X_train, y_train, X_val_57):
    """Train LGB+XGB on 57-feature data, or load saved probabilities."""
    if os.path.exists(PROBS_LGB_PATH) and os.path.exists(PROBS_XGB_PATH):
        print("Loading pre-computed probabilities from disk...")
        probs_lgb = np.load(PROBS_LGB_PATH)
        probs_xgb = np.load(PROBS_XGB_PATH)
        print(f"  LGB probs: {probs_lgb.shape}, XGB probs: {probs_xgb.shape}")
        return probs_lgb, probs_xgb

    print(f"Training LGB on {X_train.shape[0]:,} pairs ({X_train.shape[1]} features)...")
    import lightgbm as lgb
    pos = int(np.sum(y_train == 1))
    neg = int(np.sum(y_train == 0))

    lgb_model = lgb.LGBMClassifier(
        objective="binary", metric="binary_logloss",
        n_estimators=600, max_depth=7, num_leaves=63,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, scale_pos_weight=neg / max(1, pos),
        n_jobs=4, random_state=42, verbose=-1,
    )
    lgb_model.fit(X_train, y_train)
    probs_lgb = lgb_model.predict_proba(X_val_57)[:, 1].astype(np.float32)
    np.save(PROBS_LGB_PATH, probs_lgb)
    print(f"  LGB done. Saved to {PROBS_LGB_PATH}")

    print("Training XGB...")
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        n_estimators=600, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=neg / max(1, pos),
        n_jobs=4, random_state=42, tree_method="hist", device="cuda", verbosity=0,
    )
    xgb_model.fit(X_train, y_train)
    probs_xgb = xgb_model.predict_proba(X_val_57)[:, 1].astype(np.float32)
    np.save(PROBS_XGB_PATH, probs_xgb)
    print(f"  XGB done. Saved to {PROBS_XGB_PATH}")

    return probs_lgb, probs_xgb


def build_s1_index(val_meta_s1_ids, val_meta_cand_ids, val_meta_sources, val_y):
    """
    Build per-S1 index: {s1_id -> (array of cand_ids, array of sources, array of pair_indices)}
    This avoids per-pair Python loops during the sweep.
    """
    print("Building per-S1 candidate index...")
    t0 = time.time()

    # Group pair indices by s1_id
    s1_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, s1_id in enumerate(val_meta_s1_ids):
        s1_to_indices[str(s1_id)].append(i)

    s1_index = {}
    for s1_id, indices in s1_to_indices.items():
        idx_arr = np.array(indices, dtype=np.int32)
        cand_ids = [str(val_meta_cand_ids[i]) for i in indices]
        sources = [str(val_meta_sources[i]) for i in indices]
        labels = val_y[idx_arr]
        s1_index[s1_id] = {
            "indices": idx_arr,
            "cand_ids": cand_ids,
            "sources": sources,
            "labels": labels,
        }

    print(f"  Indexed {len(s1_index):,} S1 entities in {time.time()-t0:.1f}s")
    return s1_index


def fast_bridge_sweep(
    s1_index: Dict,
    val_matches: Dict[str, List[str]],
    probs: np.ndarray,
    base_thresholds=None,
    dense_thresholds=None,
    margin_deltas=None,
    anchor_gates=None,
    bridge_thresholds=None,
    density_cutoffs=None,
):
    """
    Vectorized bridge parameter sweep.
    For each configuration, iterates over S1 entities (not pairs), using pre-indexed arrays.
    """
    if base_thresholds is None:
        base_thresholds  = [0.950, 0.955, 0.960, 0.962, 0.965, 0.968, 0.970, 0.972, 0.975]
    if dense_thresholds is None:
        dense_thresholds = [0.960, 0.963, 0.966, 0.968, 0.970, 0.972, 0.975]
    if margin_deltas is None:
        margin_deltas    = [0.020, 0.025, 0.030, 0.035, 0.040]
    if anchor_gates is None:
        anchor_gates     = [0.975, 0.980, 0.985, 0.990]
    if bridge_thresholds is None:
        bridge_thresholds = [0.925, 0.930, 0.935, 0.938, 0.942, 0.945]
    if density_cutoffs is None:
        density_cutoffs  = [80, 100, 120]

    configs = [
        (bt, dt, mg, ag, br, dc)
        for bt in base_thresholds
        for dt in dense_thresholds if dt > bt
        for mg in margin_deltas
        for ag in anchor_gates
        for br in bridge_thresholds
        for dc in density_cutoffs
    ]
    total_configs = len(configs)
    print(f"Sweeping {total_configs:,} bridge configurations over {len(val_matches):,} S1 entities...")

    best_f05 = 0.0
    best_metrics = {}
    best_params = {}
    t0 = time.time()
    log_every = max(1, total_configs // 20)

    for cfg_i, (base_th, dense_th, margin, anchor, bridge, density_cutoff) in enumerate(configs):
        singleton_gate = base_th * 0.85

        evaluator = Evaluator()
        for s1_id, true_mids in val_matches.items():
            info = s1_index.get(s1_id)
            if info is None or len(info["indices"]) == 0:
                evaluator.update(s1_id, set(true_mids), set())
                continue

            idx_arr = info["indices"]
            cand_ids = info["cand_ids"]
            sources = info["sources"]
            pair_probs = probs[idx_arr]

            sort_order = np.argsort(-pair_probs)
            top_prob = pair_probs[sort_order[0]]

            if top_prob < singleton_gate:
                evaluator.update(s1_id, set(true_mids), set())
                continue

            num_cands = len(cand_ids)
            effective_th = dense_th if num_cands > density_cutoff else base_th

            accepted = set()
            anchor_ids = set()
            anchor_sources = set()

            for j in sort_order:
                prob = pair_probs[j]
                cid = cand_ids[j]
                src = sources[j]
                if prob >= effective_th and (top_prob - prob) <= margin:
                    accepted.add(cid)
                    if prob >= anchor:
                        anchor_ids.add(cid)
                        anchor_sources.add(src)

            # Transitive cross-source bridge
            if anchor_ids:
                for j in sort_order:
                    cid = cand_ids[j]
                    if cid in accepted:
                        continue
                    prob = pair_probs[j]
                    src = sources[j]
                    if prob >= bridge and src not in anchor_sources:
                        accepted.add(cid)

            evaluator.update(s1_id, set(true_mids), accepted)

        metrics = evaluator.compute()
        f05 = metrics["macro_f_05"]

        if f05 > best_f05:
            best_f05 = f05
            best_metrics = metrics
            best_params = {
                "base_th": base_th, "dense_th": dense_th, "margin": margin,
                "anchor": anchor, "bridge": bridge, "density_cutoff": density_cutoff,
            }

        if (cfg_i + 1) % log_every == 0 or cfg_i == total_configs - 1:
            elapsed = time.time() - t0
            rate = (cfg_i + 1) / elapsed
            eta = (total_configs - cfg_i - 1) / rate
            print(f"  [{cfg_i+1:,}/{total_configs:,}] best F0.5={best_f05:.6f}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    print(f"\nSweep complete in {time.time()-t0:.1f}s")
    print(f"Best F0.5: {best_f05:.6f}  Params: {best_params}")
    return best_metrics, best_params


def main():
    t_start = time.time()
    print("=" * 70)
    print("EXP_N008: Fast Vectorized Bridge Sweep")
    print("=" * 70)

    with open(CHAMPION_JSON, "r") as f:
        champion = json.load(f)
    champion_f05 = champion["metric_macro_f05"]
    print(f"Champion baseline: {champion_f05:.6f}")

    # ── Load val data ──
    print("\nLoading val pairs (val_pairs_n008.npz)...")
    va = np.load(VAL_NPZ_N008, allow_pickle=True)
    X_val_56 = va["X"].astype(np.float32)
    y_val = va["y"].astype(np.int8)
    val_s1_ids = va["s1_ids"]
    val_cand_ids = va["cand_ids"]
    val_sources = va["sources"]
    val_neg_cats = va["neg_cats"]
    print(f"  {X_val_56.shape[0]:,} pairs, {X_val_56.shape[1]} features, "
          f"{int(np.sum(y_val==1)):,} positives")

    # ── Load cosine scores ──
    from neural_encoder import load_embeddings
    print("Loading embeddings for cosine scores...")
    s1_ids_emb, s1_vecs = load_embeddings(
        os.path.join(EXPERIMENTS_DIR, "s1_embeddings.npz"))
    t_ids_emb, t_vecs = load_embeddings(
        os.path.join(EXPERIMENTS_DIR, "target_embeddings.npz"))

    s1_lookup = {sid: s1_vecs[i] for i, sid in enumerate(s1_ids_emb)}
    target_lookup = {tid: t_vecs[i] for i, tid in enumerate(t_ids_emb)}

    print("Computing val cosine scores...")
    N = len(val_s1_ids)
    val_cosine = np.zeros(N, dtype=np.float32)
    miss_s1 = miss_cand = 0
    for i in range(N):
        sv = s1_lookup.get(str(val_s1_ids[i]))
        tv = target_lookup.get(str(val_cand_ids[i]))
        if sv is None:
            miss_s1 += 1
        elif tv is None:
            miss_cand += 1
        else:
            val_cosine[i] = float(np.dot(sv, tv))

    print(f"  Cosine: mean={val_cosine.mean():.4f}, missing: {miss_s1} S1 / {miss_cand} cand")

    # ── Build 57-feature matrix ──
    X_val_57 = np.hstack([X_val_56, val_cosine.reshape(-1, 1)])
    print(f"  Val feature matrix: {X_val_57.shape}")

    # ── Load train data and get model probs ──
    print("\nLoading train pairs for model training...")
    tr = np.load(TRAIN_NPZ_EXP008, allow_pickle=True)
    X_train_56 = tr["X"].astype(np.float32)
    y_train = tr["y"].astype(np.int8)
    train_s1_ids = tr["s1_ids"]
    train_cand_ids = tr["cand_ids"]

    # Compute train cosine
    print("Computing train cosine scores...")
    N_tr = len(train_s1_ids)
    train_cosine = np.zeros(N_tr, dtype=np.float32)
    for i in range(N_tr):
        sv = s1_lookup.get(str(train_s1_ids[i]))
        tv = target_lookup.get(str(train_cand_ids[i]))
        if sv is not None and tv is not None:
            train_cosine[i] = float(np.dot(sv, tv))
    X_train_57 = np.hstack([X_train_56, train_cosine.reshape(-1, 1)])
    print(f"  Train feature matrix: {X_train_57.shape}, cosine mean={train_cosine.mean():.4f}")

    probs_lgb, probs_xgb = build_or_load_models_and_probs(X_train_57, y_train, X_val_57)

    # ── Find best blend weight ──
    print("\nSweeping blend weights...")
    val_matches = load_val_data()
    s1_index = build_s1_index(val_s1_ids, val_cand_ids, val_sources, y_val)

    best_blend_f05 = 0.0
    best_w_lgb = 0.45
    for w_lgb in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        probs_blend = w_lgb * probs_lgb + (1 - w_lgb) * probs_xgb
        # Quick evaluation at single threshold
        evaluator = Evaluator()
        for s1_id, true_mids in val_matches.items():
            info = s1_index.get(s1_id)
            if info is None:
                evaluator.update(s1_id, set(true_mids), set())
                continue
            pair_probs = probs_blend[info["indices"]]
            top = pair_probs.max() if len(pair_probs) else 0.0
            if top < 0.962 * 0.85:
                evaluator.update(s1_id, set(true_mids), set())
                continue
            accepted = {info["cand_ids"][j] for j, p in enumerate(pair_probs) if p >= 0.962}
            evaluator.update(s1_id, set(true_mids), accepted)
        m = evaluator.compute()
        f05 = m["macro_f_05"]
        print(f"  w_lgb={w_lgb:.2f}  F0.5={f05:.6f}  P={m['macro_precision']:.4f}  R={m['macro_recall']:.4f}")
        if f05 > best_blend_f05:
            best_blend_f05 = f05
            best_w_lgb = w_lgb

    probs_final = best_w_lgb * probs_lgb + (1 - best_w_lgb) * probs_xgb
    print(f"\nBest blend: w_lgb={best_w_lgb:.2f} / w_xgb={1-best_w_lgb:.2f}  "
          f"Quick F0.5={best_blend_f05:.6f}")

    # ── Full bridge sweep ──
    print("\nRunning full bridge sweep...")
    best_metrics, best_params = fast_bridge_sweep(
        s1_index, val_matches, probs_final,
        base_thresholds  = [0.945, 0.950, 0.955, 0.960, 0.962, 0.965, 0.968, 0.970, 0.972, 0.975],
        dense_thresholds = [0.958, 0.960, 0.963, 0.965, 0.968, 0.970, 0.972, 0.975],
        margin_deltas    = [0.015, 0.020, 0.025, 0.030, 0.035, 0.040],
        anchor_gates     = [0.975, 0.980, 0.982, 0.985, 0.988, 0.990],
        bridge_thresholds= [0.920, 0.925, 0.930, 0.933, 0.935, 0.938, 0.940, 0.942],
        density_cutoffs  = [60, 80, 100, 120, 150],
    )

    best_f05 = best_metrics["macro_f_05"]
    delta = best_f05 - champion_f05
    verdict = "▲ NEW CHAMPION" if delta > 0 else "▼ REGRESSION"

    print("\n" + "=" * 70)
    print("EXP_N008 FINAL RESULTS")
    print("=" * 70)
    print(f"  Champion (EXP_007)       : F0.5 = {champion_f05:.6f}")
    print(f"  EXP_N008 (57-feat hybrid): F0.5 = {best_f05:.6f}  {verdict} {abs(delta):.6f}")
    print(f"  Precision                : {best_metrics.get('macro_precision', 0):.6f}")
    print(f"  Recall                   : {best_metrics.get('macro_recall', 0):.6f}")
    print(f"  Singleton Accuracy       : {best_metrics.get('singleton_accuracy', 0):.6f}")
    print(f"  Exact Set Accuracy       : {best_metrics.get('exact_set_accuracy', 0):.6f}")
    print(f"  False Positives          : {best_metrics.get('total_false_positives', 0):,}")
    print(f"  False Negatives          : {best_metrics.get('total_false_negatives', 0):,}")
    print(f"  Best bridge params       : {best_params}")
    print(f"  Runtime                  : {(time.time()-t_start)/60:.1f} min")
    print("=" * 70)

    # Write registry
    fieldnames = [
        "exp_id", "description", "macro_f_05", "macro_precision", "macro_recall",
        "singleton_accuracy", "exact_set_accuracy", "total_entities",
        "total_true_matches", "total_false_positives", "total_false_negatives",
        "threshold", "notes",
    ]
    write_header = not os.path.exists(REGISTRY_PATH)
    with open(REGISTRY_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({
            "exp_id": "EXP_N008_fast",
            "description": "Hybrid 57-feat (56+cosine) vectorized bridge sweep",
            "macro_f_05": best_f05,
            "macro_precision": best_metrics.get("macro_precision", 0),
            "macro_recall": best_metrics.get("macro_recall", 0),
            "singleton_accuracy": best_metrics.get("singleton_accuracy", 0),
            "exact_set_accuracy": best_metrics.get("exact_set_accuracy", 0),
            "total_entities": best_metrics.get("total_entities", 0),
            "total_true_matches": best_metrics.get("total_true_matches", 0),
            "total_false_positives": best_metrics.get("total_false_positives", 0),
            "total_false_negatives": best_metrics.get("total_false_negatives", 0),
            "threshold": json.dumps(best_params),
            "notes": (f"blend=LGB{best_w_lgb:.2f}+XGB{1-best_w_lgb:.2f}, "
                      f"cosine=paraphrase-multilingual-MiniLM-L12-v2, "
                      f"val_blocker_recall=0.9644"),
        })

    if best_f05 > champion_f05:
        champion_data = {
            "iteration": champion.get("iteration", 4) + 1,
            "champion_model": f"hybrid_57feat_LGB{best_w_lgb:.2f}_XGB{1-best_w_lgb:.2f}",
            "decision_strategy": "transitive_cross_source_consistency_bridge",
            "metric_macro_f05": best_f05,
            "metric_macro_precision": best_metrics.get("macro_precision", 0),
            "metric_macro_recall": best_metrics.get("macro_recall", 0),
            "singleton_accuracy": best_metrics.get("singleton_accuracy", 0),
            "exact_set_accuracy": best_metrics.get("exact_set_accuracy", 0),
            "num_features": 57,
            "eval_population_s1": VAL_SAMPLE_SIZE,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "experiment": "EXP_N008",
            "neural_feature": NEURAL_FEATURE_NAME,
            "neural_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "base_threshold": best_params.get("base_th"),
            "dense_threshold": best_params.get("dense_th"),
            "margin_delta": best_params.get("margin"),
            "anchor_gate": best_params.get("anchor"),
            "bridge_threshold": best_params.get("bridge"),
            "density_cutoff": best_params.get("density_cutoff"),
            "blend_w_lgb": best_w_lgb,
            "blend_w_xgb": round(1 - best_w_lgb, 2),
        }
        with open(CHAMPION_JSON, "w") as f:
            json.dump(champion_data, f, indent=2)
        print(f"\n🏆 Champion updated → {CHAMPION_JSON}")
    else:
        print(f"\n  Champion retained at {champion_f05:.6f}")


if __name__ == "__main__":
    main()
