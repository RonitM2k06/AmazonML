#!/usr/bin/env python3
"""
Autonomous Experiment 007: Inference-Safe Cross-Source Consistency & Transitive Bridge (Priority 1).
Hypothesis:
In business entity resolution across 3 sources (S1 reference, S2 noisy, S3 noisy), true business entities
frequently exist in both S2 and S3 (Phase 2 forensics showed > 85% of matches co-occur).
When an S1 entity achieves an ultra-high confidence match in one source (top_prob >= 0.985 in S2 or S3),
the existence of the business entity in the external world is confirmed.
If the opposite source has a candidate that fell marginally short of the aggressive 0.962 threshold due to
asymmetric noise (e.g., probability in [0.938, 0.962)), promoting that top candidate via transitive consistency
recovers missing cross-source matches with high precision, breaking past the 0.951000 Macro F0.5 barrier.
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


def run_transitive_bridge_experiment():
    print("=" * 70, flush=True)
    print("AUTONOMOUS EXPERIMENT 007: TRANSITIVE CROSS-SOURCE CONSISTENCY BRIDGE", flush=True)
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

    champion_path = os.path.join(code_dir, "reports", "champion.json")
    with open(champion_path, "r", encoding="utf-8") as f:
        champ = json.load(f)
    champ_score = champ["metric_macro_f05"]
    print(f"Current Champion Macro F0.5: {champ_score:.6f} ({champ['champion_model']})", flush=True)

    # Optimal Transitive Bridge Parameters
    base_th = 0.962
    dense_th = 0.966
    cutoff = 100
    delta = 0.030
    anchor_gate = 0.985
    bridge_th = 0.938

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

        th = dense_th if len(idxs) >= cutoff else base_th
        sub_ids = cand_ids[idxs]
        sub_src = sources[idxs]

        # Primary decision: density-aware + dynamic margin
        mask = (sub_probs >= th) & ((top_prob - sub_probs) <= delta)
        accepted = set(sub_ids[mask])

        # Priority 1: Transitive Cross-Source Consistency Bridge
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
    prec = res["macro_precision"]
    rec = res["macro_recall"]
    sa = res["singleton_accuracy"]
    exact = res["exact_set_accuracy"]

    print("\n" + "=" * 70, flush=True)
    print(f"EVALUATION RESULTS FOR TRANSITIVE CROSS-SOURCE CONSISTENCY BRIDGE:", flush=True)
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
        champ["decision_strategy"] = "transitive_cross_source_consistency_bridge"
        champ["base_threshold"] = base_th
        champ["dense_threshold"] = dense_th
        champ["density_cutoff"] = cutoff
        champ["margin_delta"] = delta
        champ["anchor_gate"] = anchor_gate
        champ["bridge_threshold"] = bridge_th
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
            "EXP_007",
            time.strftime("%Y-%m-%dT%H:%M:%S"),
            "Transitive Cross-Source Consistency Bridge",
            f"Anchor={anchor_gate} Bridge={bridge_th} Base={base_th} Delta={delta}",
            f"{f05:.6f}",
            f"{prec:.6f}",
            f"{rec:.6f}",
            f"{sa:.6f}",
            "COMPLETED",
            str(promoted).upper(),
        ])
    print(f"Logged EXP_007 to {exp_registry_path}", flush=True)


if __name__ == "__main__":
    run_transitive_bridge_experiment()
