#!/usr/bin/env python3
"""
Entity-Disjoint Stratified Validation Split Generator.
Splits Source 1 entities 80/20 stratified by Country and Singleton status.
Saves validation S1 IDs to experiments/validation_s1_ids.txt.
"""

import os
import sys
import numpy as np

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    dataset_dir = os.path.join(project_root, "dataset")

    s1_path = os.path.join(dataset_dir, "train", "train_source1.tsv")
    gt_path = os.path.join(dataset_dir, "train", "train_ground_truth.tsv")

    experiments_dir = os.path.abspath(os.path.join(script_dir, "..", "experiments"))
    os.makedirs(experiments_dir, exist_ok=True)
    val_out_path = os.path.join(experiments_dir, "validation_s1_ids.txt")

    print(f"Generating 80/20 Stratified Validation Split (seed=42)...")

    # Read S1 countries
    s1_country = {}
    with open(s1_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1_country[parts[0]] = parts[3]

    # Read GT singleton status
    s1_singleton = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1 = parts[0]
            s1_singleton[s1] = (len(parts) < 2 or not parts[1].strip())

    total_s1 = len(s1_country)
    print(f"Total S1 entities: {total_s1:,}")

    # Group into strata: (country, is_singleton)
    strata = {}
    for s1, country in s1_country.items():
        is_sing = s1_singleton[s1]
        key = (country, is_sing)
        if key not in strata:
            strata[key] = []
        strata[key].append(s1)

    rng = np.random.RandomState(42)
    val_ids = []
    train_ids = []

    print("\nStrata Breakdown:")
    for (c, is_sing), ids in sorted(strata.items()):
        arr = np.array(ids)
        rng.shuffle(arr)
        n_val = int(round(len(arr) * 0.20))
        val_subset = arr[:n_val]
        train_subset = arr[n_val:]
        val_ids.extend(val_subset)
        train_ids.extend(train_subset)
        sing_label = "Singleton" if is_sing else "Non-Singleton"
        print(f"  Strata {c} - {sing_label:14s}: {len(arr):,} total -> {n_val:,} val (20.00%)")

    # Sort val_ids for determinism
    val_ids.sort()
    train_ids.sort()

    print(f"\nFinal Split:")
    print(f"  Training S1:   {len(train_ids):,} ({len(train_ids)/total_s1*100:.2f}%)")
    print(f"  Validation S1: {len(val_ids):,} ({len(val_ids)/total_s1*100:.2f}%)")

    with open(val_out_path, "w", encoding="utf-8") as f:
        for vid in val_ids:
            f.write(f"{vid}\n")

    print(f"\nValidation S1 IDs saved to: {val_out_path}")


if __name__ == "__main__":
    main()
