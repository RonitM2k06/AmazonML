#!/usr/bin/env python3
"""
EXP_N008: Final Focused Bridge Sweep — ~500 configs, ~2-3 minutes.

Per-config timing: 0.317s → 500 configs = ~2.6min.
Grid is focused around the confirmed best region:
  base_th ~ 0.960-0.972, blend LGB0.45/XGB0.55.
"""

import csv
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONUNBUFFERED"] = "1"

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
code_dir = os.path.join(project_root, "code", "business_entity_resolution")
sys.path.insert(0, os.path.join(code_dir, "src"))

from evaluation import Evaluator

EXPERIMENTS_DIR = os.path.join(code_dir, "experiments")
REPORTS_DIR = os.path.join(code_dir, "reports")
VAL_NPZ_N008 = os.path.join(EXPERIMENTS_DIR, "val_pairs_n008.npz")
PROBS_LGB_PATH = os.path.join(EXPERIMENTS_DIR, "probs_n008_lgb.npy")
PROBS_XGB_PATH = os.path.join(EXPERIMENTS_DIR, "probs_n008_xgb.npy")
CHAMPION_JSON = os.path.join(REPORTS_DIR, "champion.json")
REGISTRY_PATH = os.path.join(REPORTS_DIR, "neural_experiment_registry.csv")
TRAIN_GT = os.path.join(project_root, "dataset", "train", "train_ground_truth.tsv")

SEED = 42
VAL_SAMPLE_SIZE = 25000
W_LGB = 0.45


def load_val_matches():
    random.seed(SEED)
    with open(os.path.join(EXPERIMENTS_DIR, "validation_s1_ids.txt")) as f:
        val_pool = [l.strip() for l in f if l.strip()]
    selected = set(random.sample(val_pool, min(VAL_SAMPLE_SIZE, len(val_pool))))
    val_matches = {}
    with open(TRAIN_GT) as f:
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


def build_entity_structs(val_s1_ids, val_cand_ids, val_sources, val_true_sets, probs):
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
    return entities


def evaluate_config_f05(entities, base_th, dense_th, margin, anchor, bridge, dc):
    sg = base_th * 0.85
    sum_f05 = 0.0
    total = 0
    for ent in entities:
        ps = ent["probs"]
        cids = ent["cand_ids"]
        is_s3 = ent["is_s3"]
        ts = ent["true_set"]
        n = len(ps)
        total += 1
        if n == 0 or ps[0] < sg:
            if not ts:
                sum_f05 += 1.0
            continue
        top = ps[0]
        eff_th = dense_th if n > dc else base_th
        accept_mask = (ps >= eff_th) & ((top - ps) <= margin)
        anchor_mask = ps >= anchor
        if np.any(anchor_mask & accept_mask):
            if not bool(np.any(accept_mask & is_s3)):
                accept_mask = accept_mask | ((~accept_mask) & is_s3 & (ps >= bridge))
            if not bool(np.any(accept_mask & ~is_s3)):
                accept_mask = accept_mask | ((~accept_mask) & ~is_s3 & (ps >= bridge))
        accepted = {cids[j] for j in range(n) if accept_mask[j]}
        np_set = len(accepted)
        if not ts:
            if not accepted:
                sum_f05 += 1.0
        elif np_set:
            tp = len(ts & accepted)
            if tp:
                p2 = tp / np_set
                r = tp / len(ts)
                d = 0.25 * p2 + r
                if d:
                    sum_f05 += 1.25 * p2 * r / d
    return sum_f05 / total if total else 0.0


def full_metrics(entities, base_th, dense_th, margin, anchor, bridge, dc):
    """Re-evaluate best config with full Evaluator for precision/recall/FP/FN."""
    sg = base_th * 0.85
    evaluator = Evaluator()
    for ent in entities:
        ps = ent["probs"]
        cids = ent["cand_ids"]
        is_s3 = ent["is_s3"]
        ts = ent["true_set"]
        n = len(ps)
        if n == 0 or ps[0] < sg:
            evaluator.update(ent["s1_id"], ts, set())
            continue
        top = ps[0]
        eff_th = dense_th if n > dc else base_th
        accept_mask = (ps >= eff_th) & ((top - ps) <= margin)
        anchor_mask = ps >= anchor
        if np.any(anchor_mask & accept_mask):
            if not bool(np.any(accept_mask & is_s3)):
                accept_mask = accept_mask | ((~accept_mask) & is_s3 & (ps >= bridge))
            if not bool(np.any(accept_mask & ~is_s3)):
                accept_mask = accept_mask | ((~accept_mask) & ~is_s3 & (ps >= bridge))
        accepted = {cids[j] for j in range(n) if accept_mask[j]}
        evaluator.update(ent["s1_id"], ts, accepted)
    return evaluator.compute()


def main():
    t_start = time.time()
    print("=" * 70)
    print("EXP_N008: Focused Bridge Sweep (~500 configs, ~2-3 min)")
    print("=" * 70)

    with open(CHAMPION_JSON) as f:
        champion = json.load(f)
    champion_f05 = champion["metric_macro_f05"]
    print(f"Champion baseline: {champion_f05:.6f}")

    # Load
    va = np.load(VAL_NPZ_N008, allow_pickle=True)
    probs_lgb = np.load(PROBS_LGB_PATH)
    probs_xgb = np.load(PROBS_XGB_PATH)
    probs = (W_LGB * probs_lgb + (1 - W_LGB) * probs_xgb).astype(np.float32)
    print(f"Probs loaded: {len(probs):,}, mean={probs.mean():.4f}")

    val_matches = load_val_matches()
    val_true_sets = {s1: set(mids) for s1, mids in val_matches.items()}

    print("Building entity structs...", flush=True)
    t0 = time.time()
    entities = build_entity_structs(
        va["s1_ids"], va["cand_ids"], va["sources"], val_true_sets, probs
    )
    print(f"  {len(entities):,} entities in {time.time()-t0:.1f}s")

    # ── Focused grid: ~500 configs, ~2.6 min ──
    # Based on confirmed best region (base_th~0.962, blend LGB0.45/XGB0.55)
    base_thresholds   = [0.958, 0.960, 0.962, 0.963, 0.965, 0.967, 0.970]
    dense_thresholds  = [0.962, 0.965, 0.967, 0.970, 0.972, 0.975]
    margin_deltas     = [0.020, 0.025, 0.030, 0.035, 0.040]
    anchor_gates      = [0.980, 0.983, 0.985, 0.988]
    bridge_thresholds = [0.925, 0.930, 0.935, 0.938, 0.940]
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
    total_cfg = len(configs)
    print(f"\nSweeping {total_cfg:,} configurations "
          f"(est. {total_cfg * 0.317 / 60:.1f} min)...", flush=True)

    best_f05 = 0.0
    best_params = {}
    log_every = max(1, total_cfg // 25)
    t0 = time.time()

    for i, (bt, dt, mg, ag, br, dc) in enumerate(configs):
        f05 = evaluate_config_f05(entities, bt, dt, mg, ag, br, dc)
        if f05 > best_f05:
            best_f05 = f05
            best_params = {
                "base_th": bt, "dense_th": dt, "margin": mg,
                "anchor": ag, "bridge": br, "density_cutoff": dc,
            }
        if (i + 1) % log_every == 0 or i == total_cfg - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total_cfg - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:,}/{total_cfg:,}] best={best_f05:.6f}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  params={best_params}",
                  flush=True)

    print(f"\nSweep done in {time.time()-t0:.1f}s")

    # Full metrics at best config
    print("\nComputing full metrics at best config...", flush=True)
    metrics = full_metrics(entities, **best_params)

    delta = metrics["macro_f_05"] - champion_f05
    verdict = "▲ NEW CHAMPION" if delta > 0 else "▼ no improvement"

    print("\n" + "=" * 70)
    print("EXP_N008 FINAL RESULTS")
    print("=" * 70)
    print(f"  Champion (EXP_007)       : F0.5 = {champion_f05:.6f}")
    print(f"  EXP_N008 (57-feat hybrid): F0.5 = {metrics['macro_f_05']:.6f}  "
          f"{verdict}  ({delta:+.6f})")
    print(f"  Precision                : {metrics['macro_precision']:.6f}")
    print(f"  Recall                   : {metrics['macro_recall']:.6f}")
    print(f"  Singleton Accuracy       : {metrics['singleton_accuracy']:.6f}")
    print(f"  Exact Set Accuracy       : {metrics['exact_set_accuracy']:.6f}")
    print(f"  False Positives          : {metrics['total_false_positives']:,}")
    print(f"  False Negatives          : {metrics['total_false_negatives']:,}")
    print(f"  Best params              : {best_params}")
    print(f"  Blend                    : LGB={W_LGB} / XGB={1-W_LGB}")
    print(f"  Total runtime            : {(time.time()-t_start)/60:.1f} min")
    print("=" * 70)

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
            "description": "Hybrid 57-feat (56+cosine) focused bridge sweep",
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
                      f"val_blocker_recall=96.44pct, 57_features"),
        })
    print(f"Registry appended: {REGISTRY_PATH}")

    if metrics["macro_f_05"] > champion_f05:
        new_champ = {
            "iteration": champion.get("iteration", 4) + 1,
            "champion_model": f"hybrid_57feat_LGB{W_LGB}_XGB{round(1-W_LGB, 2)}",
            "decision_strategy": "transitive_cross_source_consistency_bridge",
            "threshold": best_params["base_th"],
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
        print(f"   {champion_f05:.6f} → {metrics['macro_f_05']:.6f} "
              f"({delta:+.6f})")
    else:
        print(f"Champion retained: {champion_f05:.6f}")


if __name__ == "__main__":
    main()
