#!/usr/bin/env python3
"""
Autonomous Experiment 002: Anchor-Conditioned Dual Thresholding.
Hypothesis:
When an S1 entity has at least one very high-confidence anchor match (>= anchor_gate, e.g. 0.985),
it is guaranteed to be a real, non-singleton entity. The catastrophic 0.0 penalty for false merging
on a singleton no longer applies to this entity!
Therefore, lowering the threshold for secondary candidates on anchored entities from 0.965 to a
calibrated secondary threshold (e.g., 0.92-0.95) will recover missed true matches without compromising
singleton accuracy (preserving the 96.84% singleton precision).
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

from evaluation import Evaluator
from entity_decision import EntityDecisionEngine


def run_anchor_experiment():
    print("=" * 70, flush=True)
    print("AUTONOMOUS EXPERIMENT 002: ANCHOR-CONDITIONED DUAL THRESHOLDING", flush=True)
    print("=" * 70, flush=True)

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

    print(f"Loaded ground truth for {len(val_matches):,} validation S1 entities.", flush=True)

    # 2. Load validation pairs
    val_npz_path = os.path.join(code_dir, "experiments", "val_pairs.npz")
    va_data = np.load(val_npz_path, allow_pickle=True)
    s1_ids = va_data["s1_ids"]
    cand_ids = va_data["cand_ids"]
    sources = va_data["sources"]
    print(f"Loaded {len(s1_ids):,} validation candidate pairs.", flush=True)

    # Index candidates by S1
    s1_cand_indices = defaultdict(list)
    for i, s1 in enumerate(s1_ids):
        s1_cand_indices[s1].append(i)

    # 3. Load ensemble probabilities (Current Champion: 45/45/10)
    print("Loading precomputed model validation probabilities...", flush=True)
    p_xgb = np.load(os.path.join(code_dir, "experiments", "probs_xgb.npy"))
    p_lgb = np.load(os.path.join(code_dir, "experiments", "probs_lgb.npy"))
    p_cat = np.load(os.path.join(code_dir, "experiments", "probs_cat.npy"))
    p_blend = 0.45 * p_xgb + 0.45 * p_lgb + 0.10 * p_cat

    decision_engine = EntityDecisionEngine()

    # Baseline Champion Evaluation (Global threshold 0.965)
    def eval_predictions(preds_dict):
        return decision_engine.evaluate_entity_predictions(preds_dict, val_matches)

    def predict_global(th=0.965):
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
            sub_ids = cand_ids[idxs]
            mask = sub_probs >= th
            preds[s1_id] = set(sub_ids[mask])
        return preds

    preds_champ = predict_global(0.965)
    res_champ = eval_predictions(preds_champ)
    print(f"\nChampion Baseline (Ensemble 45/45/10 at 0.965):", flush=True)
    print(f"  Macro F0.5: {res_champ['macro_f_05']:.6f} | Prec: {res_champ['macro_precision']:.4f} | Rec: {res_champ['macro_recall']:.4f} | SingleAcc: {res_champ['singleton_accuracy']:.4f}", flush=True)

    # Now evaluate Anchor-Conditioned Dual Thresholding
    # Strategy:
    # If top_prob >= anchor_gate:
    #     accept candidates with prob >= secondary_thresh
    # Else:
    #     accept candidates with prob >= base_thresh (e.g. 0.965 or 0.970)
    
    anchor_gates = [0.980, 0.985, 0.990]
    base_threshs = [0.965, 0.970]
    secondary_threshs = [0.910, 0.920, 0.930, 0.940, 0.950]

    best_res = None
    best_params = None

    print("\nSweeping Anchor-Conditioned Dual Thresholds...", flush=True)
    for a_gate in anchor_gates:
        for b_th in base_threshs:
            for s_th in secondary_threshs:
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

                    sub_ids = cand_ids[idxs]
                    # Apply dual threshold logic
                    if top_prob >= a_gate:
                        # Non-singleton entity with anchor confirmation
                        mask = sub_probs >= s_th
                    else:
                        mask = sub_probs >= b_th
                    preds[s1_id] = set(sub_ids[mask])

                res = eval_predictions(preds)
                print(f"  Anchor>={a_gate:.3f} | Base>={b_th:.3f} | Sec>={s_th:.3f} | F0.5: {res['macro_f_05']:.6f} | Prec: {res['macro_precision']:.4f} | Rec: {res['macro_recall']:.4f} | SingleAcc: {res['singleton_accuracy']:.4f}", flush=True)

                if best_res is None or res["macro_f_05"] > best_res["macro_f_05"]:
                    best_res = res
                    best_params = {
                        "anchor_gate": a_gate,
                        "base_thresh": b_th,
                        "secondary_thresh": s_th,
                    }

    print("\n" + "=" * 70, flush=True)
    print(f"BEST DUAL THRESHOLD CONFIGURATION: {best_params}", flush=True)
    print(f"  Macro F0.5: {best_res['macro_f_05']:.6f}", flush=True)
    print(f"  Precision:  {best_res['macro_precision']:.4f}", flush=True)
    print(f"  Recall:     {best_res['macro_recall']:.4f}", flush=True)
    print(f"  SingleAcc:  {best_res['singleton_accuracy']:.4f}", flush=True)
    print(f"  Champion:   {res_champ['macro_f_05']:.6f}", flush=True)
    delta = best_res["macro_f_05"] - res_champ["macro_f_05"]
    print(f"  Delta:      {delta:+.6f}", flush=True)
    print("=" * 70, flush=True)

    # Check champion update
    champion_path = os.path.join(code_dir, "reports", "champion.json")
    with open(champion_path, "r", encoding="utf-8") as f:
        champ = json.load(f)

    exp_registry_path = os.path.join(code_dir, "reports", "experiment_registry.csv")

    promoted = False
    if best_res["macro_f_05"] > champ["metric_macro_f05"]:
        promoted = True
        champ["iteration"] = 2
        champ["champion_model"] = "ensemble_XGB_LGB_CAT (45/45/10) + Anchor Dual Threshold"
        champ["decision_strategy"] = "anchor_conditioned_dual_threshold"
        champ["anchor_gate"] = best_params["anchor_gate"]
        champ["base_thresh"] = best_params["base_thresh"]
        champ["secondary_thresh"] = best_params["secondary_thresh"]
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
            "EXP_002",
            time.strftime("%Y-%m-%dT%H:%M:%S"),
            "Anchor-Conditioned Dual Thresholding",
            f"Anchor>={best_params['anchor_gate']} Base>={best_params['base_thresh']} Sec>={best_params['secondary_thresh']}",
            f"{best_res['macro_f_05']:.6f}",
            f"{best_res['macro_precision']:.6f}",
            f"{best_res['macro_recall']:.6f}",
            f"{best_res['singleton_accuracy']:.6f}",
            "COMPLETED",
            str(promoted).upper(),
        ])
    print(f"Logged EXP_002 to {exp_registry_path}", flush=True)


if __name__ == "__main__":
    run_anchor_experiment()
