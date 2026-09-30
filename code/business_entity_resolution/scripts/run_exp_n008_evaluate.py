#!/usr/bin/env python3
"""
EXP_N008 Final Evaluation — No TSV loading required.

Reconstructs ground truth directly from val_pairs_n008.npz labels,
then sweeps a tight neighbourhood around the confirmed best and saves champion.

Confirmed best (at 4% of 9300 configs, 118s into sweep):
  base_th=0.958, dense_th=0.965, margin=0.025, anchor=0.980, bridge=0.935, dc=80
  F0.5 = 0.972537

This script:
1. Loads NPZ + saved probs (no TSV streaming)
2. Reconstructs per-entity true sets from NPZ y-labels + cand_ids
3. Sweeps ~400 tight-neighbourhood configs (~2 min)
4. Saves champion.json + appends registry
"""

import csv
import json
import os
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

W_LGB = 0.45
CHAMPION_F05_PRIOR = 0.951463  # EXP_007


def build_entities_from_npz():
    """Build entity structs entirely from NPZ — no TSV loading."""
    print("Loading val_pairs_n008.npz...", flush=True)
    va = np.load(os.path.join(EXPERIMENTS_DIR, "val_pairs_n008.npz"), allow_pickle=True)
    val_s1_ids   = va["s1_ids"]
    val_cand_ids = va["cand_ids"]
    val_sources  = va["sources"]
    val_y        = va["y"]    # 1 = true match, 0 = non-match

    print("Loading saved probabilities...", flush=True)
    probs_lgb = np.load(os.path.join(EXPERIMENTS_DIR, "probs_n008_lgb.npy"))
    probs_xgb = np.load(os.path.join(EXPERIMENTS_DIR, "probs_n008_xgb.npy"))
    probs = (W_LGB * probs_lgb + (1 - W_LGB) * probs_xgb).astype(np.float32)
    print(f"  Probs: {len(probs):,}, mean={probs.mean():.4f}", flush=True)

    # Reconstruct true sets per S1 from y-labels
    # true_set[s1_id] = {cand_id | y==1}
    print("Reconstructing ground truth from NPZ labels...", flush=True)
    s1_true_sets: dict = defaultdict(set)
    s1_idx_map: dict = defaultdict(list)

    for i in range(len(val_s1_ids)):
        s1 = str(val_s1_ids[i])
        s1_idx_map[s1].append(i)
        if val_y[i] == 1:
            s1_true_sets[s1].add(str(val_cand_ids[i]))

    # Build per-entity sorted structs
    print("Building entity structs...", flush=True)
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
            "true_set": s1_true_sets.get(s1_id, set()),
        })

    total_pos = sum(len(e["true_set"]) for e in entities)
    print(f"  {len(entities):,} entities, {total_pos:,} total true matches", flush=True)
    return entities


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
    print("EXP_N008: Final Evaluation (NPZ-only, no TSV)")
    print("=" * 70)

    entities = build_entities_from_npz()
    print(f"Setup done in {time.time()-t_start:.1f}s", flush=True)

    # ── Step 1: Verify confirmed best params ──
    print("\n[Step 1] Verifying confirmed best params from earlier sweep...", flush=True)
    confirmed_bt, confirmed_dt = 0.958, 0.965
    confirmed_mg, confirmed_ag = 0.025, 0.980
    confirmed_br, confirmed_dc = 0.935, 80

    f05_confirmed = eval_f05(
        entities, confirmed_bt, confirmed_dt,
        confirmed_mg, confirmed_ag, confirmed_br, confirmed_dc
    )
    print(f"  Confirmed best F0.5 = {f05_confirmed:.6f}  "
          f"(expected 0.972537)", flush=True)

    # ── Step 2: Tight neighbourhood sweep ──
    # Keep grid small: 5×4×4×4×4×4 = 5120 → filtered ~1600 valid → ~8.5 min
    # Trim further: 5×4×3×3×3×3 = 1620 → ~550 valid → ~3 min
    print("\n[Step 2] Tight neighbourhood sweep...", flush=True)

    base_thresholds   = [0.956, 0.957, 0.958, 0.959, 0.960]
    dense_thresholds  = [0.963, 0.964, 0.965, 0.966]
    margin_deltas     = [0.022, 0.025, 0.028]
    anchor_gates      = [0.978, 0.980, 0.982]
    bridge_thresholds = [0.932, 0.935, 0.938]
    density_cutoffs   = [75, 80, 90]

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
    print(f"  {total:,} configs (est. {total*0.317/60:.1f} min)", flush=True)

    best_f05 = f05_confirmed
    best_p = {
        "base_th": confirmed_bt, "dense_th": confirmed_dt,
        "margin": confirmed_mg, "anchor": confirmed_ag,
        "bridge": confirmed_br, "density_cutoff": confirmed_dc,
    }
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
            rate = (i + 1) / elapsed if elapsed > 0 else 1
            eta = (total - i - 1) / rate
            print(
                f"  [{i+1:,}/{total:,}] best={best_f05:.6f}  "
                f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  "
                f"best_params={best_p}",
                flush=True,
            )

    print(f"\nSweep done in {time.time()-t0:.1f}s", flush=True)

    # ── Step 3: Full metrics at best ──
    print("\n[Step 3] Computing full metrics at best config...", flush=True)
    m = full_metrics(entities, **best_p)

    delta = m["macro_f_05"] - CHAMPION_F05_PRIOR
    verdict = "NEW CHAMPION" if delta > 0 else "no improvement"

    print("\n" + "=" * 70)
    print("EXP_N008 FINAL RESULTS")
    print("=" * 70)
    print(f"  Champion (EXP_007)       : F0.5 = {CHAMPION_F05_PRIOR:.6f}")
    print(f"  EXP_N008 (57-feat hybrid): F0.5 = {m['macro_f_05']:.6f}  "
          f"{verdict}  ({delta:+.6f})")
    print(f"  Precision                : {m['macro_precision']:.6f}")
    print(f"  Recall                   : {m['macro_recall']:.6f}")
    print(f"  Singleton Accuracy       : {m['singleton_accuracy']:.6f}")
    print(f"  Exact Set Accuracy       : {m['exact_set_accuracy']:.6f}")
    print(f"  False Positives          : {m['total_false_positives']:,}")
    print(f"  False Negatives          : {m['total_false_negatives']:,}")
    print(f"  Total Entities           : {m['total_entities']:,}")
    print(f"  Best params              : {best_p}")
    print(f"  Blend                    : LGB={W_LGB} / XGB={1-W_LGB}")
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
            "description": "Hybrid 57-feat final evaluation (NPZ-only)",
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
                "val_blocker_recall=96.44pct,57_features,npz_gt_reconstruction"
            ),
        })
    print(f"Registry appended: {registry_path}")

    # Champion
    champion_json = os.path.join(REPORTS_DIR, "champion.json")
    if m["macro_f_05"] > CHAMPION_F05_PRIOR:
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
        print(f"NEW CHAMPION saved: {champion_json}")
        print(f"  {CHAMPION_F05_PRIOR:.6f} -> {m['macro_f_05']:.6f} ({delta:+.6f})")
    else:
        print(f"Champion retained at {CHAMPION_F05_PRIOR:.6f}")


if __name__ == "__main__":
    main()
