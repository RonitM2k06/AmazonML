#!/usr/bin/env python3
"""
Autonomous Experiment 003: Tree Regularization and Feature Subsampling Tuning.
Hypothesis:
Default XGBoost (depth=6, colsample_bytree=1.0) placed 74.5% of its feature importance weight
on a single feature (`addr_token_set_ratio`). By enforcing feature subsampling (`colsample_bytree=0.75`),
subsampling (`subsample=0.85`), and allowing deeper trees (`max_depth=7` or `8`), the model will learn
richer non-linear interactions across name similarity, postal code, and house numbers, reducing false
positive collisions between different entities sharing street tokens.
"""

import os
import sys
import time
import random
import csv
import json
from collections import defaultdict
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["PYTHONUNBUFFERED"] = "1"

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
code_dir = os.path.join(project_root, "code", "business_entity_resolution")
sys.path.insert(0, os.path.join(code_dir, "src"))

import xgboost as xgb
from entity_decision import EntityDecisionEngine


def run_regularization_experiment():
    print("=" * 70, flush=True)
    print("AUTONOMOUS EXPERIMENT 003: TREE REGULARIZATION & FEATURE SUBSAMPLING", flush=True)
    print("=" * 70, flush=True)

    # 1. Load ground truth
    dataset_dir = os.path.abspath(os.path.join(project_root, "dataset"))
    train_dir = os.path.join(dataset_dir, "train")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    validation_ids_path = os.path.join(code_dir, "experiments", "validation_s1_ids.txt")
    with open(validation_ids_path, "r", encoding="utf-8") as f:
        val_pool = [line.strip() for line in f if line.strip()]

    random.seed(42)
    selected_val_s1 = set(random.sample(val_pool, min(25000, len(val_pool))))

    val_matches = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1 = parts[0]
            if s1 in selected_val_s1:
                mids = parts[1].split(",") if len(parts) > 1 and parts[1].strip() else []
                val_matches[s1] = mids
                if len(val_matches) >= 25000:
                    break

    print(f"Loaded ground truth for {len(val_matches):,} validation S1 entities.", flush=True)

    # 2. Load pre-extracted pair arrays
    tr_data = np.load(os.path.join(code_dir, "experiments", "train_pairs.npz"), allow_pickle=True)
    va_data = np.load(os.path.join(code_dir, "experiments", "val_pairs.npz"), allow_pickle=True)
    X_train = tr_data["X"]
    y_train = tr_data["y"]
    X_val = va_data["X"]
    s1_ids = va_data["s1_ids"]
    cand_ids = va_data["cand_ids"]

    # Subsample validation set for early stopping
    np.random.seed(42)
    val_sub_idx = np.random.choice(len(X_val), 30000, replace=False)
    X_eval = X_val[val_sub_idx]
    y_eval = va_data["y"][val_sub_idx]

    s1_cand_indices = defaultdict(list)
    for i, s1 in enumerate(s1_ids):
        s1_cand_indices[s1].append(i)

    engine = EntityDecisionEngine()

    # Load current champion info
    champion_path = os.path.join(code_dir, "reports", "champion.json")
    with open(champion_path, "r", encoding="utf-8") as f:
        champ = json.load(f)

    champ_score = champ["metric_macro_f05"]
    print(f"Current Champion Macro F0.5: {champ_score:.6f} ({champ['champion_model']})", flush=True)

    # Configurations to evaluate
    configs = [
        {"name": "XGB_depth7_colsample0.80", "depth": 7, "colsample": 0.80, "subsample": 0.85, "lr": 0.07, "trees": 350},
        {"name": "XGB_depth8_colsample0.75", "depth": 8, "colsample": 0.75, "subsample": 0.85, "lr": 0.06, "trees": 400},
        {"name": "XGB_depth6_colsample0.80", "depth": 6, "colsample": 0.80, "subsample": 0.90, "lr": 0.08, "trees": 300},
    ]

    best_res = None
    best_cfg = None
    best_th = None
    best_probs = None

    for cfg in configs:
        print(f"\nTraining configuration: {cfg['name']}...", flush=True)
        t0 = time.time()
        model = xgb.XGBClassifier(
            n_estimators=cfg["trees"],
            learning_rate=cfg["lr"],
            max_depth=cfg["depth"],
            colsample_bytree=cfg["colsample"],
            subsample=cfg["subsample"],
            scale_pos_weight=1.0,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=35,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)
        t_train = time.time() - t0
        print(f"  Trained in {t_train:.2f}s", flush=True)

        # Chunked inference
        p_val = np.zeros(len(X_val), dtype=np.float32)
        chunk_size = 200000
        for i in range(0, len(X_val), chunk_size):
            chunk = np.ascontiguousarray(X_val[i : i + chunk_size], dtype=np.float32)
            p_val[i : i + chunk_size] = model.predict_proba(chunk)[:, 1]

        # Sweep thresholds
        for th in [0.94, 0.95, 0.955, 0.960, 0.965, 0.970]:
            preds = {}
            for s1_id in val_matches.keys():
                idxs = s1_cand_indices.get(s1_id)
                if not idxs:
                    preds[s1_id] = set()
                    continue
                sub_probs = p_val[idxs]
                top_prob = float(np.max(sub_probs))
                if top_prob < 0.65:
                    preds[s1_id] = set()
                    continue
                sub_ids = cand_ids[idxs]
                mask = sub_probs >= th
                preds[s1_id] = set(sub_ids[mask])
            res = engine.evaluate_entity_predictions(preds, val_matches)
            f05 = res["macro_f_05"]
            prec = res["macro_precision"]
            rec = res["macro_recall"]
            sa = res["singleton_accuracy"]
            print(f"  {cfg['name']:25s} | th={th:.3f} | F0.5: {f05:.6f} | Prec: {prec:.4f} | Rec: {rec:.4f} | SingleAcc: {sa:.4f}", flush=True)

            if best_res is None or f05 > best_res["macro_f_05"]:
                best_res = res
                best_cfg = cfg
                best_th = th
                best_probs = p_val

    print("\n" + "=" * 70, flush=True)
    print(f"BEST REGULARIZED MODEL: {best_cfg['name']} at threshold {best_th:.3f}", flush=True)
    print(f"  Macro F0.5: {best_res['macro_f_05']:.6f}", flush=True)
    print(f"  Precision:  {best_res['macro_precision']:.4f}", flush=True)
    print(f"  Recall:     {best_res['macro_recall']:.4f}", flush=True)
    print(f"  SingleAcc:  {best_res['singleton_accuracy']:.4f}", flush=True)
    print(f"  Champion:   {champ_score:.6f}", flush=True)
    delta = best_res["macro_f_05"] - champ_score
    print(f"  Delta:      {delta:+.6f}", flush=True)
    print("=" * 70, flush=True)

    # Also test ensembling the new tuned model with LGB and CatBoost
    print("\nTesting Ensemble with New Tuned XGBoost...", flush=True)
    p_lgb = np.load(os.path.join(code_dir, "experiments", "probs_lgb.npy"))
    p_cat = np.load(os.path.join(code_dir, "experiments", "probs_cat.npy"))
    p_new_blend = 0.45 * best_probs + 0.45 * p_lgb + 0.10 * p_cat

    for th in [0.955, 0.960, 0.965, 0.970]:
        preds = {}
        for s1_id in val_matches.keys():
            idxs = s1_cand_indices.get(s1_id)
            if not idxs:
                preds[s1_id] = set()
                continue
            sub_probs = p_new_blend[idxs]
            top_prob = float(np.max(sub_probs))
            if top_prob < 0.65:
                preds[s1_id] = set()
                continue
            sub_ids = cand_ids[idxs]
            mask = sub_probs >= th
            preds[s1_id] = set(sub_ids[mask])
        res_blend = engine.evaluate_entity_predictions(preds, val_matches)
        print(f"  New Tuned Blend | th={th:.3f} | F0.5: {res_blend['macro_f_05']:.6f} | Prec: {res_blend['macro_precision']:.4f} | Rec: {res_blend['macro_recall']:.4f} | SingleAcc: {res_blend['singleton_accuracy']:.4f}", flush=True)
        if res_blend["macro_f_05"] > best_res["macro_f_05"]:
            best_res = res_blend
            best_cfg["name"] = f"ensemble_{best_cfg['name']}_LGB_CAT"
            best_th = th

    # Log to registry & check promotion
    exp_registry_path = os.path.join(code_dir, "reports", "experiment_registry.csv")
    promoted = False
    if best_res["macro_f_05"] > champ_score:
        promoted = True
        champ["iteration"] = champ["iteration"] + 1
        champ["champion_model"] = best_cfg["name"]
        champ["threshold"] = best_th
        champ["metric_macro_f05"] = round(best_res["macro_f_05"], 6)
        champ["metric_macro_precision"] = round(best_res["macro_precision"], 6)
        champ["metric_macro_recall"] = round(best_res["macro_recall"], 6)
        champ["singleton_accuracy"] = round(best_res["singleton_accuracy"], 6)
        champ["exact_set_accuracy"] = round(best_res["exact_set_accuracy"], 6)
        champ["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(champion_path, "w", encoding="utf-8") as f:
            json.dump(champ, f, indent=2)
        print(">>> PROMOTED NEW CHAMPION! Updated reports/champion.json <<<", flush=True)
    else:
        print("Experiment did not beat Champion. Retaining baseline.", flush=True)

    with open(exp_registry_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "EXP_003",
            time.strftime("%Y-%m-%dT%H:%M:%S"),
            "Tree Regularization & Feature Subsampling",
            f"{best_cfg['name']} th={best_th}",
            f"{best_res['macro_f_05']:.6f}",
            f"{best_res['macro_precision']:.6f}",
            f"{best_res['macro_recall']:.6f}",
            f"{best_res['singleton_accuracy']:.6f}",
            "COMPLETED",
            str(promoted).upper(),
        ])
    print(f"Logged EXP_003 to {exp_registry_path}", flush=True)


if __name__ == "__main__":
    run_regularization_experiment()
