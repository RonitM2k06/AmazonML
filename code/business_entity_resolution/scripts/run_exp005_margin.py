#!/usr/bin/env python3
"""
Autonomous Experiment 005: Dynamic Margin-Relative Decision Optimization.
Hypothesis:
In multi-candidate sets for an S1 entity, accepting secondary candidates whose predicted matching
probability significantly lags behind the entity's top candidate (top_prob - cand_prob > delta)
frequently introduces near-duplicate confusion or branch-level false positives.
By enforcing a dynamic relative margin rule ((sub_probs >= base_th) & (top_prob - sub_probs <= delta)),
we aggressively prune ambiguity while retaining multi-source true matches.
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


def run_margin_experiment():
    print("=" * 70, flush=True)
    print("AUTONOMOUS EXPERIMENT 005: DYNAMIC MARGIN-RELATIVE DECISION OPTIMIZATION", flush=True)
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

    s1_cand_indices = defaultdict(list)
    for i, s1 in enumerate(s1_ids):
        s1_cand_indices[s1].append(i)

    # Load Champion probabilities (Blend 45/45/10)
    p_xgb = np.load(os.path.join(code_dir, "experiments", "probs_xgb.npy"))
    p_lgb = np.load(os.path.join(code_dir, "experiments", "probs_lgb.npy"))
    p_cat = np.load(os.path.join(code_dir, "experiments", "probs_cat.npy"))
    p_blend = 0.45 * p_xgb + 0.45 * p_lgb + 0.10 * p_cat

    engine = EntityDecisionEngine()

    champion_path = os.path.join(code_dir, "reports", "champion.json")
    with open(champion_path, "r", encoding="utf-8") as f:
        champ = json.load(f)
    champ_score = champ["metric_macro_f05"]
    print(f"Current Champion Macro F0.5: {champ_score:.6f} ({champ['champion_model']})", flush=True)

    # Evaluate optimal margin config
    base_th = 0.962
    delta = 0.030

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
        mask = (sub_probs >= base_th) & ((top_prob - sub_probs) <= delta)
        preds[s1_id] = set(sub_ids[mask])

    res = engine.evaluate_entity_predictions(preds, val_matches)
    f05 = res["macro_f_05"]
    prec = res["macro_precision"]
    rec = res["macro_recall"]
    sa = res["singleton_accuracy"]
    exact = res["exact_set_accuracy"]

    print("\n" + "=" * 70, flush=True)
    print(f"EVALUATION RESULTS FOR DYNAMIC MARGIN (Base={base_th}, Delta={delta}):", flush=True)
    print(f"  Macro F0.5: {f05:.6f}", flush=True)
    print(f"  Precision:  {prec:.4f}", flush=True)
    print(f"  Recall:     {rec:.4f}", flush=True)
    print(f"  SingleAcc:  {sa:.4f}", flush=True)
    print(f"  Exact Set:  {exact:.4f}", flush=True)
    print(f"  Champion:   {champ_score:.6f}", flush=True)
    diff = f05 - champ_score
    print(f"  Delta:      {diff:+.6f}", flush=True)
    print("=" * 70, flush=True)

    promoted = False
    if f05 > champ_score:
        promoted = True
        champ["iteration"] = champ["iteration"] + 1
        champ["decision_strategy"] = "dynamic_margin_relative"
        champ["base_threshold"] = base_th
        champ["margin_delta"] = delta
        champ["metric_macro_f05"] = round(f05, 6)
        champ["metric_macro_precision"] = round(prec, 6)
        champ["metric_macro_recall"] = round(rec, 6)
        champ["singleton_accuracy"] = round(sa, 6)
        champ["exact_set_accuracy"] = round(exact, 6)
        champ["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(champion_path, "w", encoding="utf-8") as f:
            json.dump(champ, f, indent=2)
        print(">>> PROMOTED NEW CHAMPION! Updated reports/champion.json <<<", flush=True)

    exp_registry_path = os.path.join(code_dir, "reports", "experiment_registry.csv")
    with open(exp_registry_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "EXP_005",
            time.strftime("%Y-%m-%dT%H:%M:%S"),
            "Dynamic Margin-Relative Decision Optimization",
            f"Ensemble (45/45/10) Base={base_th} Delta={delta}",
            f"{f05:.6f}",
            f"{prec:.6f}",
            f"{rec:.6f}",
            f"{sa:.6f}",
            "COMPLETED",
            str(promoted).upper(),
        ])
    print(f"Logged EXP_005 to {exp_registry_path}", flush=True)


if __name__ == "__main__":
    run_margin_experiment()
