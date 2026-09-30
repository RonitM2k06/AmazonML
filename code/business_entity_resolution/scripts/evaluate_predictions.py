#!/usr/bin/env python3
"""
CLI Evaluation Script for ML Challenge 2026 predictions.
Computes entity-level macro F_0.5, Precision, Recall, and diagnostic metrics.
"""

import argparse
import json
import os
import sys
import time

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from evaluation import evaluate_files


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate entity resolution predictions using macro F_0.5."
    )
    parser.add_argument(
        "--ground-truth",
        "-g",
        required=True,
        help="Path to ground truth TSV (e.g. dataset/train/train_ground_truth.tsv)",
    )
    parser.add_argument(
        "--predictions",
        "-p",
        required=True,
        help="Path to predictions TSV (e.g. output/matching_results.tsv)",
    )
    parser.add_argument(
        "--validation-ids",
        "-v",
        default=None,
        help="Optional path to validation S1 IDs text file to evaluate only on validation split",
    )
    parser.add_argument(
        "--output-json",
        "-o",
        default=None,
        help="Optional path to save evaluation results as JSON",
    )
    args = parser.parse_args()

    target_ids = None
    if args.validation_ids and os.path.isfile(args.validation_ids):
        print(f"Loading validation S1 IDs from {args.validation_ids}...")
        with open(args.validation_ids, "r", encoding="utf-8") as f:
            target_ids = {line.strip() for line in f if line.strip()}
        print(f"  Filtering to {len(target_ids):,} validation entities.")

    print(f"Evaluating {args.predictions} against {args.ground_truth}...")
    t0 = time.time()
    results = evaluate_files(args.ground_truth, args.predictions, target_s1_ids=target_ids)
    elapsed = time.time() - t0

    print("\n" + "=" * 55)
    print("EVALUATION RESULTS (Macro F_0.5)")
    print("=" * 55)
    print(f"Total Entities Evaluated:      {results['total_entities']:,}")
    print(f"Total True Matches:            {results['total_true_matches']:,}")
    print(f"Total Predicted Matches:       {results['total_predicted_matches']:,}")
    print(f"Macro F_0.5 Score:             {results['macro_f_05']:.6f}  <-- PRIMARY METRIC")
    print(f"Macro Precision:               {results['macro_precision']:.6f}")
    print(f"Macro Recall:                  {results['macro_recall']:.6f}")
    print(f"Exact Set Accuracy:            {results['exact_set_accuracy']:.6f}")
    print(f"Singleton Accuracy:            {results['singleton_accuracy']:.6f} ({results['total_singletons']:,} singletons)")
    print(f"Singleton False Positive Rate: {results['singleton_false_positive_rate']:.6f}")
    print(f"Total False Positives:         {results['total_false_positives']:,}")
    print(f"Total False Negatives:         {results['total_false_negatives']:,}")
    print(f"Evaluation Time:               {elapsed:.2f}s")
    print("=" * 55)

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
