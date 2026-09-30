#!/usr/bin/env python3
"""
Autonomous Experiment 008:
Phonetic Skeleton & Address-Pair High-Recall Candidate Generation + Transitive Consistency Optimization.
Tests whether expanding the blocker recall ceiling from 92.45% to >95.5% (recovering hundreds of previously
missed cross-lingual and spelling variants) translates directly into a higher Entity-Level Macro F0.5.
"""

import os
import sys
import time
import random
import csv
import json
from collections import defaultdict, Counter
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["PYTHONUNBUFFERED"] = "1"

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
code_dir = os.path.join(project_root, "code", "business_entity_resolution")
sys.path.insert(0, os.path.join(code_dir, "src"))

from pair_features import FEATURE_NAMES
from build_pair_dataset import build_pair_dataset, save_pair_dataset
from run_phase5_experiment import load_train_and_val_samples
from train_matcher import train_matcher
from entity_decision import EntityDecisionEngine


def run_experiment_008():
    t_start = time.time()
    print("=" * 75, flush=True)
    print("AUTONOMOUS EXPERIMENT 008: HIGH-RECALL PHONETIC & ADDRESS-PAIR BLOCKER PIPELINE", flush=True)
    print("=" * 75, flush=True)

    dataset_dir = os.path.join(project_root, "dataset")
    validation_ids_path = os.path.join(code_dir, "experiments", "validation_s1_ids.txt")
    reports_dir = os.path.join(code_dir, "reports")
    models_dir = os.path.join(code_dir, "models")
    exp_dir = os.path.join(code_dir, "experiments")

    train_npz_path = os.path.join(exp_dir, "train_pairs_exp008.npz")
    val_npz_path = os.path.join(exp_dir, "val_pairs_exp008.npz")

    # 1. Load Samples
    print("\n[Step 1] Loading Train and Validation Samples...", flush=True)
    (
        train_s1, train_target_dict, train_matches,
        val_s1, val_target_dict, val_matches
    ) = load_train_and_val_samples(
        dataset_dir, validation_ids_path, val_sample_size=25000, train_sample_size=15000, seed=42
    )

    # 2. Extract Candidate Pairs & Extract 56-dim features
    if os.path.exists(train_npz_path) and os.path.exists(val_npz_path):
        print(f"\n[Step 2] Loading pre-extracted EXP_008 datasets from {train_npz_path}...", flush=True)
        tr_data = np.load(train_npz_path, allow_pickle=True)
        va_data = np.load(val_npz_path, allow_pickle=True)
        X_train = tr_data["X"]
        y_train = tr_data["y"]
        X_val = va_data["X"]
        y_val = va_data["y"]
        s1_ids_val = va_data["s1_ids"]
        cand_ids_val = va_data["cand_ids"]
        sources_val = va_data["sources"]
    else:
        print("\n[Step 2] Building Training Pairs with EXP_008 Blocker...", flush=True)
        X_train, y_train, train_meta, train_neg_cats = build_pair_dataset(
            train_s1, train_target_dict, train_matches,
            max_bucket_size=1000, max_candidates_per_s1=200, max_negatives_per_s1=15, seed=42
        )
        save_pair_dataset(train_npz_path, X_train, y_train, train_meta, train_neg_cats)

        print("\n[Step 2] Building Validation Pairs with EXP_008 Blocker...", flush=True)
        X_val, y_val, val_meta, val_neg_cats = build_pair_dataset(
            val_s1, val_target_dict, val_matches,
            max_bucket_size=1000, max_candidates_per_s1=200, max_negatives_per_s1=None, seed=42
        )
        save_pair_dataset(val_npz_path, X_val, y_val, val_meta, val_neg_cats)

        s1_ids_val = np.array([m["s1_id"] for m in val_meta], dtype=object)
        cand_ids_val = np.array([m["cand_id"] for m in val_meta], dtype=object)
        sources_val = np.array([m["source"] for m in val_meta], dtype=object)

    print(f"\nDatasets ready: Train X {X_train.shape}, Val X {X_val.shape}", flush=True)

    # 3. Train High-Efficiency Matcher
    print("\n[Step 3] Training Matchers on 56 Features...", flush=True)
    # Train LightGBM first (fastest, high accuracy)
    model_lgb, lgb_meta = train_matcher(
        "lightgbm", X_train, y_train, X_val, y_val, FEATURE_NAMES,
        num_trees=300, learning_rate=0.05, max_depth=6, random_state=42
    )
    print("Predicting LightGBM probabilities on validation set...", flush=True)
    p_lgb = model_lgb.predict_proba(X_val)[:, 1]

    # Train XGBoost
    model_xgb, xgb_meta = train_matcher(
        "xgboost", X_train, y_train, X_val, y_val, FEATURE_NAMES,
        num_trees=300, learning_rate=0.05, max_depth=6, random_state=42
    )
    print("Predicting XGBoost probabilities on validation set...", flush=True)
    p_xgb = model_xgb.predict_proba(X_val)[:, 1]

    p_blend = 0.50 * p_lgb + 0.50 * p_xgb

    # Save probabilities
    np.save(os.path.join(exp_dir, "probs_exp008_blend.npy"), p_blend)

    # 4. Entity-Level Decision & Threshold Sweeps
    print("\n[Step 4] Optimizing Entity-Level Decisions...", flush=True)
    s1_cand_indices = defaultdict(list)
    for i, s1 in enumerate(s1_ids_val):
        s1_cand_indices[s1].append(i)

    engine = EntityDecisionEngine()

    champion_path = os.path.join(reports_dir, "champion.json")
    with open(champion_path, "r", encoding="utf-8") as f:
        champ = json.load(f)
    champ_score = champ["metric_macro_f05"]
    print(f"Current Champion Macro F0.5: {champ_score:.6f} ({champ['champion_model']})", flush=True)

    # Grid search across decision hyperparameters
    best_f05 = 0.0
    best_params = {}
    best_metrics = {}

    anchor_gate = 0.985
    bridge_th = 0.938

    for base_th in [0.958, 0.960, 0.962, 0.964, 0.966]:
        for dense_th in [base_th + 0.004]:
            for delta in [0.025, 0.030, 0.035]:
                preds = {}
                for s1_id in val_matches.keys():
                    idxs = s1_cand_indices.get(s1_id)
                    if not idxs:
                        preds[s1_id] = set()
                        continue
                    sub_probs = p_blend[idxs]
                    top_prob = float(np.max(sub_probs))
                    if top_prob < 0.65:
                        preds[s1_id] = set()
                        continue

                    th = dense_th if len(idxs) >= 80 else base_th
                    sub_ids = cand_ids_val[idxs]
                    sub_src = sources_val[idxs]

                    # Density-aware + dynamic margin
                    mask = (sub_probs >= th) & ((top_prob - sub_probs) <= delta)
                    accepted = set(sub_ids[mask])

                    # Transitive consistency bridge
                    if top_prob >= anchor_gate:
                        has_s2 = any(c.startswith("S2-") for c in accepted)
                        has_s3 = any(c.startswith("S3-") for c in accepted)
                        if has_s2 and not has_s3:
                            s3_idxs = [i for i, src in enumerate(sub_src) if src == "S3" and sub_probs[i] >= bridge_th]
                            if s3_idxs:
                                best_s3_i = max(s3_idxs, key=lambda i: sub_probs[i])
                                accepted.add(sub_ids[best_s3_i])
                        elif has_s3 and not has_s2:
                            s2_idxs = [i for i, src in enumerate(sub_src) if src == "S2" and sub_probs[i] >= bridge_th]
                            if s2_idxs:
                                best_s2_i = max(s2_idxs, key=lambda i: sub_probs[i])
                                accepted.add(sub_ids[best_s2_i])

                    preds[s1_id] = accepted

                res = engine.evaluate_entity_predictions(preds, val_matches)
                f05 = res["macro_f_05"]
                if f05 > best_f05:
                    best_f05 = f05
                    best_params = {
                        "base_th": base_th,
                        "dense_th": dense_th,
                        "delta": delta,
                        "anchor_gate": anchor_gate,
                        "bridge_th": bridge_th,
                    }
                    best_metrics = res

    print("\n" + "=" * 75, flush=True)
    print(f"EXP_008 BEST EVALUATION RESULTS:", flush=True)
    print(f"  Macro F0.5:     {best_metrics['macro_f_05']:.6f}", flush=True)
    print(f"  Precision:      {best_metrics['macro_precision']:.6f}", flush=True)
    print(f"  Recall:         {best_metrics['macro_recall']:.6f}", flush=True)
    print(f"  SingleAcc:      {best_metrics['singleton_accuracy']:.6f}", flush=True)
    print(f"  Exact Set Acc:  {best_metrics['exact_set_accuracy']:.6f}", flush=True)
    print(f"  Parameters:     {best_params}", flush=True)
    print(f"  Prior Champion: {champ_score:.6f}", flush=True)
    delta_score = best_metrics['macro_f_05'] - champ_score
    print(f"  Score Delta:    {delta_score:+.6f}", flush=True)
    print("=" * 75, flush=True)

    promoted = False
    if best_metrics["macro_f_05"] > champ_score:
        promoted = True
        champ["iteration"] = champ["iteration"] + 1
        champ["champion_model"] = "EXP_008 Blend LGB_XGB (56 features + High-Recall Blocker)"
        champ["decision_strategy"] = "density_margin_transitive_bridge"
        champ["base_threshold"] = best_params["base_th"]
        champ["dense_threshold"] = best_params["dense_th"]
        champ["margin_delta"] = best_params["delta"]
        champ["anchor_gate"] = best_params["anchor_gate"]
        champ["bridge_threshold"] = best_params["bridge_th"]
        champ["metric_macro_f05"] = round(best_metrics["macro_f_05"], 6)
        champ["metric_macro_precision"] = round(best_metrics["macro_precision"], 6)
        champ["metric_macro_recall"] = round(best_metrics["macro_recall"], 6)
        champ["singleton_accuracy"] = round(best_metrics["singleton_accuracy"], 6)
        champ["exact_set_accuracy"] = round(best_metrics["exact_set_accuracy"], 6)
        champ["num_features"] = 56
        champ["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(champion_path, "w", encoding="utf-8") as f:
            json.dump(champ, f, indent=2)
        print(">>> PROMOTED NEW CHAMPION! Updated reports/champion.json <<<", flush=True)

    exp_registry_path = os.path.join(reports_dir, "experiment_registry.csv")
    with open(exp_registry_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "EXP_008",
            time.strftime("%Y-%m-%dT%H:%M:%S"),
            "Phonetic Skeleton & AddrPair Blocker + 56 Features",
            f"Base={best_params['base_th']} Dense={best_params['dense_th']} Delta={best_params['delta']}",
            f"{best_metrics['macro_f_05']:.6f}",
            f"{best_metrics['macro_precision']:.6f}",
            f"{best_metrics['macro_recall']:.6f}",
            f"{best_metrics['singleton_accuracy']:.6f}",
            "COMPLETED",
            str(promoted).upper(),
        ])
    print(f"Logged EXP_008 to {exp_registry_path}", flush=True)
    print(f"Total experiment time: {time.time()-t_start:.2f}s", flush=True)


if __name__ == "__main__":
    run_experiment_008()
