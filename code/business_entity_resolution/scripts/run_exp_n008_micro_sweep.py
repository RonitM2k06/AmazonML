#!/usr/bin/env python3
"""
EXP_N008: Micro-sweep around confirmed best params.

Confirmed best so far (at 4% of 9300 configs):
  base_th=0.958, dense_th=0.965, margin=0.025, anchor=0.980, bridge=0.935, dc=80
  F0.5 = 0.972537

This script sweeps a tight ±3 neighbourhood around those values:
~8 x 8 x 5 x 4 x 5 x 4 = 25,600 → filtered by dt>bt → ~3,200 valid configs
At 0.317s/config → ~17 min. Too slow.

ACTUALLY: restrict tightly to ±2 of each axis:
4 x 4 x 3 x 3 x 3 x 3 = 1,296 → ~432 valid → ~2.3 min. Perfect.
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
TRAIN_GT = os.path.join(project_root, "dataset", "train", "train_ground_truth.tsv")

SEED = 42
W_LGB = 0.45


def load_and_build():
    va = np.load(os.path.join(EXPERIMENTS_DIR, "val_pairs_n008.npz"), allow_pickle=True)
    probs_lgb = np.load(os.path.join(EXPERIMENTS_DIR, "probs_n008_lgb.npy"))
    probs_xgb = np.load(os.path.join(EXPERIMENTS_DIR, "probs_n008_xgb.npy"))
    probs = (W_LGB * probs_lgb + (1 - W_LGB) * probs_xgb).astype(np.float32)

    random.seed(SEED)
    with open(os.path.join(EXPERIMENTS_DIR, "validation_s1_ids.txt")) as f:
        val_pool = [l.strip() for l in f if l.strip()]
    selected = set(random.sample(val_pool, 25000))

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
            if len(val_matches) >= 25000:
                break

    val_true_sets = {s1: set(mids) for s1, mids in val_matches.items()}

    s1_idx_map = defaultdict(list)
    for i, s1 in enumerate(va["s1_ids"]):
        s1_idx_map[str(s1)].append(i)

    entities = []
    for s1_id, indices in s1_idx_map.items():
        idx_arr = np.array(indices, dtype=np.int32)
        p = probs[idx_arr]
        order = np.argsort(-p)
        entities.append({
            "s1_id": s1_id,
            "probs": p[order].astype(np.float32),
            "cand_ids": [str(va["cand_ids"][indices[j]]) for j in order],
            "is_s3": np.array(
                [str(va["sources"][indices[j]]) == "S3" for j in order], dtype=np.bool_
            ),
            "true_set": val_true_sets.get(s1_id, set()),
        })

    return entities, val_matches


def eval_f05(entities, bt, dt, mg, ag, br, dc):
    sg = bt * 0.85
    s = 0.0
    n = 0
    for e in entities:
        ps = e["probs"]
        cids = e["cand_ids"]
        is_s3 = e["is_s3"]
        ts = e["true_set"]
        n += 1
        N = len(ps)
        if N == 0 or ps[0] < sg:
            if not ts:
                s += 1.0
            continue
        top = ps[0]
        et = dt if N > dc else bt
        am = (ps >= et) & ((top - ps) <= mg)
        anm = ps >= ag
        if np.any(anm & am):
            if not bool(np.any(am & is_s3)):
                am = am | ((~am) & is_s3 & (ps >= br))
            if not bool(np.any(am & ~is_s3)):
                am = am | ((~am) & ~is_s3 & (ps >= br))
        acc = {cids[j] for j in range(N) if am[j]}
        np_ = len(acc)
        if not ts:
            if not acc:
                s += 1.0
        elif np_:
            tp = len(ts & acc)
            if tp:
                p2 = tp / np_
                r = tp / len(ts)
                d = 0.25 * p2 + r
                if d:
                    s += 1.25 * p2 * r / d
    return s / n if n else 0.0


def full_metrics(entities, bt, dt, mg, ag, br, dc):
    sg = bt * 0.85
    evaluator = Evaluator()
    for e in entities:
        ps = e["probs"]
        cids = e["cand_ids"]
        is_s3 = e["is_s3"]
        ts = e["true_set"]
        N = len(ps)
        if N == 0 or ps[0] < sg:
            evaluator.update(e["s1_id"], ts, set())
            continue
        top = ps[0]
        et = dt if N > dc else bt
        am = (ps >= et) & ((top - ps) <= mg)
        anm = ps >= ag
        if np.any(anm & am):
            if not bool(np.any(am & is_s3)):
                am = am | ((~am) & is_s3 & (ps >= br))
            if not bool(np.any(am & ~is_s3)):
                am = am | ((~am) & ~is_s3 & (ps >= br))
        acc = {cids[j] for j in range(N) if am[j]}
        evaluator.update(e["s1_id"], ts, acc)
    return evaluator.compute()


def main():
    t_start = time.time()
    print("=" * 70)
    print("EXP_N008: Micro-Sweep Around Confirmed Best")
    print("Confirmed best at 4pct sweep: F0.5=0.972537")
    print("  base_th=0.958, dense_th=0.965, margin=0.025, anchor=0.980, bridge=0.935, dc=80")
    print("=" * 70)

    champion_f05 = 0.951463
    print(f"Champion baseline: {champion_f05:.6f}")

    print("Loading data and building entity structs...", flush=True)
    entities, val_matches = load_and_build()
    print(f"  {len(entities):,} entities built", flush=True)

    # Tight neighbourhood: ±2 steps around each confirmed best param
    base_thresholds   = [0.955, 0.957, 0.958, 0.959, 0.960, 0.962]
    dense_thresholds  = [0.962, 0.963, 0.964, 0.965, 0.966, 0.967, 0.968]
    margin_deltas     = [0.020, 0.022, 0.025, 0.028, 0.030]
    anchor_gates      = [0.977, 0.978, 0.980, 0.982, 0.985]
    bridge_thresholds = [0.928, 0.930, 0.932, 0.935, 0.937, 0.938]
    density_cutoffs   = [70, 75, 80, 85, 90, 100]

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
    est_min = total * 0.317 / 60
    print(f"\nSweeping {total:,} configs (est. {est_min:.1f} min)...", flush=True)

    best_f05 = 0.0
    best_p = {}
    log_every = max(1, total // 20)
    t0 = time.time()

    for i, (bt, dt, mg, ag, br, dc) in enumerate(configs):
        f05 = eval_f05(entities, bt, dt, mg, ag, br, dc)
        if f05 > best_f05:
            best_f05 = f05
            best_p = {
                "base_th": bt, "dense_th": dt, "margin": mg,
                "anchor": ag, "bridge": br, "density_cutoff": dc,
            }
        if (i + 1) % log_every == 0 or i == total - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1:,}/{total:,}] best={best_f05:.6f}  "
                f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  "
                f"params={best_p}",
                flush=True,
            )

    print(f"\nSweep done in {time.time()-t0:.1f}s. Best F0.5={best_f05:.6f}", flush=True)
    print(f"Best params: {best_p}")

    # Full metrics
    print("\nComputing full metrics...", flush=True)
    m = full_metrics(entities, **best_p)

    delta = m["macro_f_05"] - champion_f05
    verdict = "NEW CHAMPION" if delta > 0 else "no improvement"

    print("\n" + "=" * 70)
    print("EXP_N008 FINAL RESULTS")
    print("=" * 70)
    print(f"  Champion (EXP_007)       : F0.5 = {champion_f05:.6f}")
    print(f"  EXP_N008 (57-feat hybrid): F0.5 = {m['macro_f_05']:.6f}  "
          f"{verdict}  ({delta:+.6f})")
    print(f"  Precision                : {m['macro_precision']:.6f}")
    print(f"  Recall                   : {m['macro_recall']:.6f}")
    print(f"  Singleton Accuracy       : {m['singleton_accuracy']:.6f}")
    print(f"  Exact Set Accuracy       : {m['exact_set_accuracy']:.6f}")
    print(f"  False Positives          : {m['total_false_positives']:,}")
    print(f"  False Negatives          : {m['total_false_negatives']:,}")
    print(f"  Best params              : {best_p}")
    print(f"  Blend                    : LGB={W_LGB} / XGB={1 - W_LGB}")
    print(f"  Total runtime            : {(time.time()-t_start)/60:.1f} min")
    print("=" * 70)

    # Registry
    registry_path = os.path.join(REPORTS_DIR, "neural_experiment_registry.csv")
    fnames = [
        "exp_id", "description", "macro_f_05", "macro_precision", "macro_recall",
        "singleton_accuracy", "exact_set_accuracy", "total_entities",
        "total_true_matches", "total_false_positives", "total_false_negatives",
        "threshold", "notes",
    ]
    write_hdr = not os.path.exists(registry_path)
    with open(registry_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fnames, extrasaction="ignore")
        if write_hdr:
            writer.writeheader()
        writer.writerow({
            "exp_id": "EXP_N008",
            "description": "Hybrid 57-feat micro-sweep final",
            "macro_f_05": m["macro_f_05"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "singleton_accuracy": m["singleton_accuracy"],
            "exact_set_accuracy": m["exact_set_accuracy"],
            "total_entities": m["total_entities"],
            "total_true_matches": m["total_true_matches"],
            "total_false_positives": m["total_false_positives"],
            "total_false_negatives": m["total_false_negatives"],
            "threshold": json.dumps(best_p),
            "notes": (
                "blend=LGB0.45+XGB0.55,"
                "cosine=paraphrase-multilingual-MiniLM-L12-v2,"
                "val_blocker_recall=96.44pct,57_features"
            ),
        })
    print(f"Registry appended: {registry_path}")

    # Champion update
    if m["macro_f_05"] > champion_f05:
        champion_json = os.path.join(REPORTS_DIR, "champion.json")
        champ = {
            "iteration": 5,
            "champion_model": "hybrid_57feat_LGB0.45_XGB0.55",
            "decision_strategy": "transitive_cross_source_consistency_bridge",
            "threshold": best_p["base_th"],
            "metric_macro_f05": m["macro_f_05"],
            "metric_macro_precision": m["macro_precision"],
            "metric_macro_recall": m["macro_recall"],
            "singleton_accuracy": m["singleton_accuracy"],
            "exact_set_accuracy": m["exact_set_accuracy"],
            "num_features": 57,
            "eval_population_s1": 25000,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "experiment": "EXP_N008",
            "neural_feature": "neural_cosine_sim",
            "neural_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "base_threshold": best_p["base_th"],
            "dense_threshold": best_p["dense_th"],
            "margin_delta": best_p["margin"],
            "anchor_gate": best_p["anchor"],
            "bridge_threshold": best_p["bridge"],
            "density_cutoff": best_p["density_cutoff"],
            "blend_w_lgb": W_LGB,
            "blend_w_xgb": round(1 - W_LGB, 2),
        }
        with open(champion_json, "w") as f:
            json.dump(champ, f, indent=2)
        print(f"NEW CHAMPION saved to {champion_json}")
        print(f"  {champion_f05:.6f} -> {m['macro_f_05']:.6f} ({delta:+.6f})")
    else:
        print(f"Champion retained at {champion_f05:.6f}")


if __name__ == "__main__":
    main()
