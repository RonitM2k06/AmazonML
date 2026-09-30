#!/usr/bin/env python3
"""
Autonomous Experiment 001: Model Ensembling (XGBoost + LightGBM + CatBoost).
Tests hypothesis that ensembling reduces singleton false positive variance,
improving Macro F0.5 beyond single model champion (0.950124).
"""

import os
import sys
import pickle
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

from evaluation import Evaluator


def run_ensemble_experiment():
    print("=" * 70)
    print("AUTONOMOUS EXPERIMENT 001: MODEL ENSEMBLING")
    print("=" * 70)

    # 1. Load validation ground truth
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

    print(f"Loaded ground truth for {len(val_matches):,} validation S1 entities.")

    # 2. Load validation pairs
    val_npz_path = os.path.join(code_dir, "experiments", "val_pairs.npz")
    va_data = np.load(val_npz_path, allow_pickle=True)
    X_val = va_data["X"]
    s1_ids = va_data["s1_ids"]
    cand_ids = va_data["cand_ids"]
    sources = va_data["sources"]
    print(f"Loaded {len(X_val):,} validation candidate pairs.")

    # Index by S1
    s1_cand_indices = defaultdict(list)
    for i, s1 in enumerate(s1_ids):
        s1_cand_indices[s1].append(i)

    # 3. Load precomputed probabilities (isolated processes guarantee C runtime stability)
    print("Loading precomputed model validation probabilities...")
    p_xgb = np.load(os.path.join(code_dir, "experiments", "probs_xgb.npy"))
    p_lgb = np.load(os.path.join(code_dir, "experiments", "probs_lgb.npy"))
    p_cat = np.load(os.path.join(code_dir, "experiments", "probs_cat.npy"))
    print(f"Probabilities loaded: XGB ({len(p_xgb):,}), LGB ({len(p_lgb):,}), CAT ({len(p_cat):,})")

    from entity_decision import EntityDecisionEngine
    decision_engine = EntityDecisionEngine()

    def eval_prob(p_arr, th=0.96, s2_th=None, s3_th=None):
        preds = {}
        for s1_id in val_matches.keys():
            idxs = s1_cand_indices.get(s1_id)
            if not idxs:
                preds[s1_id] = set()
                continue
            sub_probs = p_arr[idxs]
            top_prob = float(np.max(sub_probs))
            if top_prob < 0.65:
                preds[s1_id] = set()
                continue
            sub_ids = cand_ids[idxs]
            if s2_th is not None and s3_th is not None:
                sub_src = sources[idxs]
                mask = ((sub_src == "S2") & (sub_probs >= s2_th)) | ((sub_src == "S3") & (sub_probs >= s3_th))
            else:
                mask = sub_probs >= th
            preds[s1_id] = set(sub_ids[mask])
        return decision_engine.evaluate_entity_predictions(preds, val_matches)

    # Baseline verification
    res_xgb = eval_prob(p_xgb, 0.96)
    print(f"\nChampion Baseline (XGB at 0.96):")
    print(f"  Macro F0.5: {res_xgb['macro_f_05']:.6f} | Prec: {res_xgb['macro_precision']:.4f} | Rec: {res_xgb['macro_recall']:.4f} | SingleAcc: {res_xgb['singleton_accuracy']:.4f}")

    # Evaluate multiple blends
    blend_weights = [
        ("XGB_LGB (50/50)", 0.50, 0.50, 0.00),
        ("XGB_LGB_CAT (45/45/10)", 0.45, 0.45, 0.10),
        ("XGB_LGB_CAT (50/30/20)", 0.50, 0.30, 0.20),
        ("XGB_LGB_CAT (40/40/20)", 0.40, 0.40, 0.20),
        ("XGB_CAT (60/40)", 0.60, 0.00, 0.40),
    ]

    best_exp_res = None
    best_config = None

    thresholds = [0.93, 0.94, 0.95, 0.955, 0.96, 0.965, 0.97]

    print("\nSweeping Ensembles across Thresholds...")
    for label, wx, wl, wc in blend_weights:
        p_blend = wx * p_xgb + wl * p_lgb + wc * p_cat
        for th in thresholds:
            res = eval_prob(p_blend, th)
            print(f"  {label:24s} | th: {th:.3f} | F0.5: {res['macro_f_05']:.6f} | Prec: {res['macro_precision']:.4f} | Rec: {res['macro_recall']:.4f} | SingleAcc: {res['singleton_accuracy']:.4f}")
            if best_exp_res is None or res["macro_f_05"] > best_exp_res["macro_f_05"]:
                best_exp_res = res
                best_config = {
                    "label": label,
                    "weights": [wx, wl, wc],
                    "threshold": th,
                }

    print("\n" + "=" * 70)
    print(f"BEST ENSEMBLE CONFIGURATION: {best_config}")
    print(f"  Macro F0.5: {best_exp_res['macro_f_05']:.6f}")
    print(f"  Precision:  {best_exp_res['macro_precision']:.4f}")
    print(f"  Recall:     {best_exp_res['macro_recall']:.4f}")
    print(f"  SingleAcc:  {best_exp_res['singleton_accuracy']:.4f}")
    print(f"  Champion:   {res_xgb['macro_f_05']:.6f}")
    delta = best_exp_res["macro_f_05"] - res_xgb["macro_f_05"]
    print(f"  Delta:      {delta:+.6f}")
    print("=" * 70)

    # Check champion update
    champion_path = os.path.join(code_dir, "reports", "champion.json")
    with open(champion_path, "r", encoding="utf-8") as f:
        champ = json.load(f)

    exp_registry_path = os.path.join(code_dir, "reports", "experiment_registry.csv")

    promoted = False
    if best_exp_res["macro_f_05"] > champ["metric_macro_f05"]:
        promoted = True
        champ["iteration"] = 1
        champ["champion_model"] = f"ensemble_{best_config['label']}"
        champ["weights"] = best_config["weights"]
        champ["threshold"] = best_config["threshold"]
        champ["metric_macro_f05"] = round(best_exp_res["macro_f_05"], 6)
        champ["metric_macro_precision"] = round(best_exp_res["macro_precision"], 6)
        champ["metric_macro_recall"] = round(best_exp_res["macro_recall"], 6)
        champ["singleton_accuracy"] = round(best_exp_res["singleton_accuracy"], 6)
        champ["exact_set_accuracy"] = round(best_exp_res["exact_set_accuracy"], 6)
        champ["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(champion_path, "w", encoding="utf-8") as f:
            json.dump(champ, f, indent=2)
        print(">>> PROMOTED NEW CHAMPION! Updated reports/champion.json <<<")
    else:
        print("Ensemble did not beat Champion. Retaining baseline.")

    # Append to experiment_registry.csv
    with open(exp_registry_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "EXP_001",
            time.strftime("%Y-%m-%dT%H:%M:%S"),
            "Model Ensembling",
            f"Blend {best_config['label']} th={best_config['threshold']}",
            f"{best_exp_res['macro_f_05']:.6f}",
            f"{best_exp_res['macro_precision']:.6f}",
            f"{best_exp_res['macro_recall']:.6f}",
            f"{best_exp_res['singleton_accuracy']:.6f}",
            "COMPLETED",
            str(promoted).upper(),
        ])
    print(f"Logged EXP_001 to {exp_registry_path}")


if __name__ == "__main__":
    run_ensemble_experiment()
