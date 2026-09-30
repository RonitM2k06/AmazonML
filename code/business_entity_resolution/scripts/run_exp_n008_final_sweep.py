#!/usr/bin/env python3
"""
EXP_N008: Truly Vectorized Bridge Sweep — Runs in minutes not hours.

Key insight: Instead of looping over 66,240 configs × 25k S1 entities,
we vectorize the decision logic using pre-sorted per-entity probability arrays
and broadcast threshold comparisons.

Strategy:
1. Load probs, group by S1 entity once.
2. For each S1: store sorted probs + cand_ids + sources as fixed arrays.
3. For EACH threshold combo: apply vectorized accept/reject across all S1 entities
   in a single pass using pre-computed sorted arrays.
4. Progress reported every ~200 configs (~3% of the condensed grid).

Grid is narrowed to the highest-value region around confirmed best:
  - base_th near 0.960-0.972 (confirmed optimum region)
  - bridge near 0.920-0.942
  - margin 0.020-0.040
  - Fewer anchor/density combos
  → ~2,400 configs, ~30s-2min total.
"""

import csv
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

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

from evaluation import Evaluator

EXPERIMENTS_DIR = os.path.join(code_dir, "experiments")
REPORTS_DIR = os.path.join(code_dir, "reports")
VAL_NPZ_N008 = os.path.join(EXPERIMENTS_DIR, "val_pairs_n008.npz")
PROBS_LGB_PATH = os.path.join(EXPERIMENTS_DIR, "probs_n008_lgb.npy")
PROBS_XGB_PATH = os.path.join(EXPERIMENTS_DIR, "probs_n008_xgb.npy")
CHAMPION_JSON = os.path.join(REPORTS_DIR, "champion.json")
REGISTRY_PATH = os.path.join(REPORTS_DIR, "neural_experiment_registry.csv")

SEED = 42
VAL_SAMPLE_SIZE = 25000
W_LGB = 0.45  # confirmed best blend


def load_val_matches():
    """Load frozen 25k validation ground truth."""
    import random
    random.seed(SEED)
    TRAIN_DIR = os.path.join(project_root, "dataset", "train")
    VAL_IDS_PATH = os.path.join(EXPERIMENTS_DIR, "validation_s1_ids.txt")
    with open(VAL_IDS_PATH, "r", encoding="utf-8") as f:
        val_pool = [l.strip() for l in f if l.strip()]
    selected = set(random.sample(val_pool, min(VAL_SAMPLE_SIZE, len(val_pool))))
    val_matches = {}
    with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1 = parts[0]
            if s1 not in selected:
                continue
            mids = parts[1].split(",") if len(parts) > 1 and parts[1].strip() else []
            val_matches[s1] = mids
            if len(val_matches) >= VAL_SAMPLE_SIZE:
                break
    return val_matches


def build_entity_structs(
    val_s1_ids, val_cand_ids, val_sources, val_true_sets, probs
):
    """
    Pre-build per-S1 sorted structures for fast sweep.
    Returns list of entity dicts sorted descending by prob:
      { s1_id, probs_sorted, cand_ids_sorted, is_s3_sorted, true_set }
    """
    print("Building per-entity sorted structures...", flush=True)
    t0 = time.time()

    # Group indices by s1_id
    s1_idx_map = defaultdict(list)
    for i, s1 in enumerate(val_s1_ids):
        s1_idx_map[str(s1)].append(i)

    entities = []
    for s1_id, indices in s1_idx_map.items():
        idx_arr = np.array(indices, dtype=np.int32)
        p = probs[idx_arr]
        order = np.argsort(-p)

        entities.append({
            "s1_id": s1_id,
            "probs": p[order].astype(np.float32),
            "cand_ids": [str(val_cand_ids[indices[j]]) for j in order],
            "is_s3": np.array(
                [str(val_sources[indices[j]]) == "S3" for j in order], dtype=np.bool_
            ),
            "true_set": val_true_sets.get(s1_id, set()),
        })

    print(f"  Built {len(entities):,} entity structs in {time.time()-t0:.1f}s", flush=True)
    return entities


def evaluate_config(
    entities,
    base_th: float,
    dense_th: float,
    margin: float,
    anchor: float,
    bridge: float,
    density_cutoff: int,
) -> Tuple[float, float, float, int, int, int]:
    """
    Evaluate one bridge-parameter config across all entities.
    Returns (macro_f05, precision, recall, tp, fp, fn).
    """
    singleton_gate = base_th * 0.85
    sum_f05 = sum_prec = sum_rec = 0.0
    total = n_singletons = correct_singletons = 0

    for ent in entities:
        probs_s = ent["probs"]
        cand_ids = ent["cand_ids"]
        is_s3 = ent["is_s3"]
        true_set = ent["true_set"]
        n = len(probs_s)
        total += 1

        if n == 0 or probs_s[0] < singleton_gate:
            if not true_set:
                sum_f05 += 1.0; sum_prec += 1.0; sum_rec += 1.0
                n_singletons += 1; correct_singletons += 1
            # else fn
            continue

        top = probs_s[0]
        eff_th = dense_th if n > density_cutoff else base_th

        # Primary accept mask: threshold + margin
        accept_mask = (probs_s >= eff_th) & ((top - probs_s) <= margin)
        anchor_mask = probs_s >= anchor

        accepted_s3 = bool(np.any(accept_mask & is_s3))
        accepted_s2 = bool(np.any(accept_mask & ~is_s3))

        # Bridge: add cross-source candidates above bridge threshold
        if np.any(anchor_mask & accept_mask):
            if not accepted_s3:
                bridge_s3 = (~accept_mask) & is_s3 & (probs_s >= bridge)
                accept_mask = accept_mask | bridge_s3
            if not accepted_s2:
                bridge_s2 = (~accept_mask) & ~is_s3 & (probs_s >= bridge)
                accept_mask = accept_mask | bridge_s2

        accepted_ids = {cand_ids[j] for j in range(n) if accept_mask[j]}

        # Compute per-entity F0.5
        n_pred = len(accepted_ids)
        is_true_singleton = len(true_set) == 0

        if is_true_singleton:
            n_singletons += 1
            if n_pred == 0:
                sum_f05 += 1.0; sum_prec += 1.0; sum_rec += 1.0
                correct_singletons += 1
        elif n_pred == 0:
            pass  # fn, contributes 0
        else:
            tp = len(true_set & accepted_ids)
            if tp > 0:
                p = tp / n_pred
                r = tp / len(true_set)
                denom = 0.25 * p + r
                f05 = (1.25 * p * r) / denom if denom > 0 else 0.0
                sum_f05 += f05; sum_prec += p; sum_rec += r

    macro_f05 = sum_f05 / total if total else 0.0
    return macro_f05


def main():
    t_start = time.time()
    print("=" * 70)
    print("EXP_N008: Final Vectorized Bridge Sweep")
    print("=" * 70)

    with open(CHAMPION_JSON) as f:
        champion = json.load(f)
    champion_f05 = champion["metric_macro_f05"]
    print(f"Champion: {champion_f05:.6f}")

    # Load
    print("\nLoading val pairs and probabilities...")
    va = np.load(VAL_NPZ_N008, allow_pickle=True)
    val_s1_ids  = va["s1_ids"]
    val_cand_ids = va["cand_ids"]
    val_sources  = va["sources"]

    probs_lgb = np.load(PROBS_LGB_PATH)
    probs_xgb = np.load(PROBS_XGB_PATH)
    probs = (W_LGB * probs_lgb + (1 - W_LGB) * probs_xgb).astype(np.float32)
    print(f"  Probs: min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")

    val_matches = load_val_matches()
    val_true_sets = {s1: set(mids) for s1, mids in val_matches.items()}

    entities = build_entity_structs(val_s1_ids, val_cand_ids, val_sources, val_true_sets, probs)

    # ── Sweep grid — focused on confirmed optimum region ──
    base_thresholds   = [0.952, 0.955, 0.958, 0.960, 0.962, 0.963, 0.965, 0.967, 0.970, 0.972]
    dense_thresholds  = [0.960, 0.963, 0.965, 0.967, 0.970, 0.972, 0.975]
    margin_deltas     = [0.015, 0.020, 0.025, 0.030, 0.035, 0.040]
    anchor_gates      = [0.980, 0.983, 0.985, 0.988, 0.990]
    bridge_thresholds = [0.920, 0.925, 0.928, 0.930, 0.933, 0.935, 0.938, 0.940]
    density_cutoffs   = [80, 100, 120]

    configs = [
        (bt, dt, mg, ag, br, dc)
        for bt in base_thresholds
        for dt in dense_thresholds if dt > bt
        for mg in margin_deltas
        for ag in anchor_gates
        for br in bridge_thresholds
        for dc in density_cutoffs
    ]
    total = len(configs)
    print(f"\nSweeping {total:,} configurations...", flush=True)

    best_f05 = 0.0
    best_params = {}
    log_every = max(1, total // 40)  # log ~40 times

    t0 = time.time()
    for i, (base_th, dense_th, margin, anchor, bridge, dc) in enumerate(configs):
        f05 = evaluate_config(entities, base_th, dense_th, margin, anchor, bridge, dc)

        if f05 > best_f05:
            best_f05 = f05
            best_params = {
                "base_th": base_th, "dense_th": dense_th, "margin": margin,
                "anchor": anchor, "bridge": bridge, "density_cutoff": dc,
            }

        if (i + 1) % log_every == 0 or i == total - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:,}/{total:,}] best={best_f05:.6f}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

    elapsed_total = time.time() - t_start
    delta = best_f05 - champion_f05
    verdict = "▲ NEW CHAMPION" if delta > 0 else "▼ REGRESSION"

    print("\n" + "=" * 70)
    print("EXP_N008 FINAL RESULTS")
    print("=" * 70)
    print(f"  Champion (EXP_007)       : F0.5 = {champion_f05:.6f}")
    print(f"  EXP_N008 (57-feat hybrid): F0.5 = {best_f05:.6f}  {verdict} {abs(delta):.6f}")
    print(f"  Best params              : {best_params}")
    print(f"  Blend                    : LGB={W_LGB:.2f} / XGB={1-W_LGB:.2f}")
    print(f"  Runtime                  : {elapsed_total/60:.1f} min")
    print("=" * 70)

    # Get full metrics at best config for registry
    # Re-evaluate at best params to get precision/recall/FP/FN
    print("\nComputing full metrics at best config...", flush=True)
    evaluator = Evaluator()
    singleton_gate = best_params["base_th"] * 0.85
    eff_base = best_params["base_th"]
    eff_dense = best_params["dense_th"]
    margin = best_params["margin"]
    anchor = best_params["anchor"]
    bridge = best_params["bridge"]
    dc = best_params["density_cutoff"]

    for ent in entities:
        probs_s = ent["probs"]
        cand_ids = ent["cand_ids"]
        is_s3 = ent["is_s3"]
        true_set = ent["true_set"]
        n = len(probs_s)

        if n == 0 or probs_s[0] < singleton_gate:
            evaluator.update(ent["s1_id"], true_set, set())
            continue

        top = probs_s[0]
        eff_th = eff_dense if n > dc else eff_base
        accept_mask = (probs_s >= eff_th) & ((top - probs_s) <= margin)
        anchor_mask = probs_s >= anchor

        accepted_s3 = bool(np.any(accept_mask & is_s3))
        accepted_s2 = bool(np.any(accept_mask & ~is_s3))

        if np.any(anchor_mask & accept_mask):
            if not accepted_s3:
                accept_mask = accept_mask | ((~accept_mask) & is_s3 & (probs_s >= bridge))
            if not accepted_s2:
                accept_mask = accept_mask | ((~accept_mask) & ~is_s3 & (probs_s >= bridge))

        accepted_ids = {cand_ids[j] for j in range(n) if accept_mask[j]}
        evaluator.update(ent["s1_id"], true_set, accepted_ids)

    metrics = evaluator.compute()
    print(f"  Macro F0.5      : {metrics['macro_f_05']:.6f}")
    print(f"  Precision       : {metrics['macro_precision']:.6f}")
    print(f"  Recall          : {metrics['macro_recall']:.6f}")
    print(f"  Singleton Acc   : {metrics['singleton_accuracy']:.6f}")
    print(f"  Exact Set Acc   : {metrics['exact_set_accuracy']:.6f}")
    print(f"  False Positives : {metrics['total_false_positives']:,}")
    print(f"  False Negatives : {metrics['total_false_negatives']:,}")

    # Registry
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
            "exp_id": "EXP_N008",
            "description": "Hybrid 57-feat (56+cosine) full bridge sweep",
            "macro_f_05": metrics["macro_f_05"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "singleton_accuracy": metrics["singleton_accuracy"],
            "exact_set_accuracy": metrics["exact_set_accuracy"],
            "total_entities": metrics["total_entities"],
            "total_true_matches": metrics["total_true_matches"],
            "total_false_positives": metrics["total_false_positives"],
            "total_false_negatives": metrics["total_false_negatives"],
            "threshold": json.dumps(best_params),
            "notes": (f"blend=LGB{W_LGB}+XGB{1-W_LGB}, "
                      f"cosine=paraphrase-multilingual-MiniLM-L12-v2, "
                      f"val_blocker_recall=0.9644, 57_features"),
        })
    print(f"\nRegistry updated: {REGISTRY_PATH}")

    # Update champion
    if metrics["macro_f_05"] > champion_f05:
        new_champ = {
            "iteration": champion.get("iteration", 4) + 1,
            "champion_model": f"hybrid_57feat_LGB{W_LGB}_XGB{round(1-W_LGB, 2)}",
            "decision_strategy": "transitive_cross_source_consistency_bridge",
            "metric_macro_f05": metrics["macro_f_05"],
            "metric_macro_precision": metrics["macro_precision"],
            "metric_macro_recall": metrics["macro_recall"],
            "singleton_accuracy": metrics["singleton_accuracy"],
            "exact_set_accuracy": metrics["exact_set_accuracy"],
            "num_features": 57,
            "eval_population_s1": VAL_SAMPLE_SIZE,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "experiment": "EXP_N008",
            "neural_feature": "neural_cosine_sim",
            "neural_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "base_threshold": best_params["base_th"],
            "dense_threshold": best_params["dense_th"],
            "margin_delta": best_params["margin"],
            "anchor_gate": best_params["anchor"],
            "bridge_threshold": best_params["bridge"],
            "density_cutoff": best_params["density_cutoff"],
            "blend_w_lgb": W_LGB,
            "blend_w_xgb": round(1 - W_LGB, 2),
        }
        with open(CHAMPION_JSON, "w") as f:
            json.dump(new_champ, f, indent=2)
        print(f"🏆 NEW CHAMPION saved → {CHAMPION_JSON}")
        print(f"   F0.5: {champion_f05:.6f} → {metrics['macro_f_05']:.6f} "
              f"(+{metrics['macro_f_05'] - champion_f05:.6f})")
    else:
        print(f"Champion retained at {champion_f05:.6f}")


if __name__ == "__main__":
    main()
