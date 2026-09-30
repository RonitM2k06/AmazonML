#!/usr/bin/env python3
"""
Autonomous Experiment 004: Source-Specific Asymmetric Threshold Optimization.
Hypothesis:
Phase 2 noise forensics showed S3 contains significantly higher noise, abbreviation variance,
and transliteration noise than S2. Applying symmetric thresholds forces a suboptimal compromise.
By jointly optimizing decoupled thresholds (th_S2, th_S3), we can maximize precision on the noisier
source while preserving recall on the cleaner source.
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

from entity_decision import EntityDecisionEngine


def run_source_specific_experiment():
    print("=" * 70, flush=True)
    print("AUTONOMOUS EXPERIMENT 004: SOURCE-SPECIFIC ASYMMETRIC THRESHOLDS", flush=True)
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

    # 2. Load validation pairs
    va_data = np.load(os.path.join(code_dir, "experiments", "val_pairs.npz"), allow_pickle=True)
    s1_ids = va_data["s1_ids"]
    cand_ids = va_data["cand_ids"]
    sources = va_data["sources"]

    s1_cand_indices = defaultdict(list)
    for i, s1 in enumerate(s1_ids):
        s1_cand_indices[s1].append(i)

    # Load Champion probabilities (Blend 45/45/10)
    p_xgb = np.load(os.path.join(code_dir, "experiments", "probs_xgb.npy"))
    p_lgb = np.load(os.path.join(code_dir, "experiments", "probs_lgb.npy"))
    p_cat = np.load(os.path.join(code_dir, "experiments", "probs_cat.npy"))
    p_blend = 0.45 * p_xgb + 0.45 * p_lgb + 0.10 * p_cat

    engine = EntityDecisionEngine()

    # Load Champion
    champion_path = os.path.join(code_dir, "reports", "champion.json")
    with open(champion_path, "r", encoding="utf-8") as f:
        champ = json.load(f)
    champ_score = champ["metric_macro_f05"]
    print(f"Current Champion Macro F0.5: {champ_score:.6f} ({champ['champion_model']})", flush=True)

    s2_grid = [0.955, 0.960, 0.963, 0.965, 0.967, 0.970]
    s3_grid = [0.955, 0.960, 0.963, 0.965, 0.967, 0.970]

    best_res = None
    best_s2 = None
    best_s3 = None

    print("\nSweeping S2 and S3 threshold grid...", flush=True)
    for th_s2 in s2_grid:
        for th_s3 in s3_grid:
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
                sub_src = sources[idxs]
                mask = ((sub_src == "S2") & (sub_probs >= th_s2)) | ((sub_src == "S3") & (sub_probs >= th_s3))
                preds[s1_id] = set(sub_ids[mask])

            res = engine.evaluate_entity_predictions(preds, val_matches)
            f05 = res["macro_f_05"]
            prec = res["macro_precision"]
            rec = res["macro_recall"]
            sa = res["singleton_accuracy"]

            print(f"  S2_th: {th_s2:.3f} | S3_th: {th_s3:.3f} | F0.5: {f05:.6f} | Prec: {prec:.4f} | Rec: {rec:.4f} | SingleAcc: {sa:.4f}", flush=True)

            if best_res is None or f05 > best_res["macro_f_05"]:
                best_res = res
                best_s2 = th_s2
                best_s3 = th_s3

    print("\n" + "=" * 70, flush=True)
    print(f"BEST ASYMMETRIC THRESHOLDS: S2={best_s2:.3f}, S3={best_s3:.3f}", flush=True)
    print(f"  Macro F0.5: {best_res['macro_f_05']:.6f}", flush=True)
    print(f"  Precision:  {best_res['macro_precision']:.4f}", flush=True)
    print(f"  Recall:     {best_res['macro_recall']:.4f}", flush=True)
    print(f"  SingleAcc:  {best_res['singleton_accuracy']:.4f}", flush=True)
    print(f"  Champion:   {champ_score:.6f}", flush=True)
    delta = best_res["macro_f_05"] - champ_score
    print(f"  Delta:      {delta:+.6f}", flush=True)
    print("=" * 70, flush=True)

    # Check champion update
    promoted = False
    if best_res["macro_f_05"] > champ_score:
        promoted = True
        champ["iteration"] = champ["iteration"] + 1
        champ["champion_model"] = "ensemble_XGB_LGB_CAT (45/45/10)"
        champ["decision_strategy"] = "source_specific_asymmetric"
        champ["threshold_s2"] = best_s2
        champ["threshold_s3"] = best_s3
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

    exp_registry_path = os.path.join(code_dir, "reports", "experiment_registry.csv")
    with open(exp_registry_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "EXP_004",
            time.strftime("%Y-%m-%dT%H:%M:%S"),
            "Source-Specific Asymmetric Thresholds",
            f"th_S2={best_s2:.3f} th_S3={best_s3:.3f}",
            f"{best_res['macro_f_05']:.6f}",
            f"{best_res['macro_precision']:.6f}",
            f"{best_res['macro_recall']:.6f}",
            f"{best_res['singleton_accuracy']:.6f}",
            "COMPLETED",
            str(promoted).upper(),
        ])
    print(f"Logged EXP_004 to {exp_registry_path}", flush=True)


if __name__ == "__main__":
    run_source_specific_experiment()
