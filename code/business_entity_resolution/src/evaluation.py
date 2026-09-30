"""
Official Evaluation Module for Amazon ML Challenge 2026: Business Entity Resolution.
Computes entity-level macro F_0.5, macro precision, macro recall, singleton metrics,
and error breakdowns.
"""

from typing import Dict, Iterable, Optional, Set, Tuple, Union
import numpy as np


def compute_entity_metrics(
    true_ids: Set[str], pred_ids: Set[str]
) -> Tuple[float, float, float, int, int, int]:
    """
    Compute F_0.5, precision, recall, tp, fp, fn for a single Source 1 entity.

    Formula:
        F_0.5 = (1.25 * precision * recall) / (0.25 * precision + recall)

    Special cases:
        - truth empty & pred empty (true singleton correctly predicted):
            precision = 1.0, recall = 1.0, F_0.5 = 1.0
        - truth empty & pred nonempty (singleton false merge):
            precision = 0.0, recall = 0.0, F_0.5 = 0.0
        - truth nonempty & pred empty (missed match):
            precision = 0.0, recall = 0.0, F_0.5 = 0.0
        - normal case:
            precision = TP / |pred|
            recall = TP / |truth|
            denom = 0.25 * precision + recall
            F_0.5 = (1.25 * precision * recall) / denom (0.0 if denom == 0)

    Returns:
        (f_05, precision, recall, tp, fp, fn)
    """
    n_true = len(true_ids)
    n_pred = len(pred_ids)

    # Both empty -> true singleton correctly identified
    if n_true == 0 and n_pred == 0:
        return 1.0, 1.0, 1.0, 0, 0, 0

    # One is empty and the other is not
    if n_true == 0 or n_pred == 0:
        tp = 0
        fp = n_pred
        fn = n_true
        return 0.0, 0.0, 0.0, tp, fp, fn

    # Both nonempty
    intersection = true_ids.intersection(pred_ids)
    tp = len(intersection)
    fp = n_pred - tp
    fn = n_true - tp

    if tp == 0:
        return 0.0, 0.0, 0.0, tp, fp, fn

    precision = tp / n_pred
    recall = tp / n_true

    denom = 0.25 * precision + recall
    if denom <= 0.0:
        f_05 = 0.0
    else:
        f_05 = (1.25 * precision * recall) / denom

    return f_05, precision, recall, tp, fp, fn


class Evaluator:
    """Streaming and batch evaluator for entity resolution predictions."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_entities = 0
        self.total_singletons = 0
        self.correct_singletons = 0
        self.exact_set_matches = 0

        self.sum_f_05 = 0.0
        self.sum_precision = 0.0
        self.sum_recall = 0.0

        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_true_matches = 0
        self.total_pred_matches = 0

    def update(self, s1_id: str, true_ids: Set[str], pred_ids: Set[str]):
        """Add an entity evaluation result."""
        f05, prec, rec, tp, fp, fn = compute_entity_metrics(true_ids, pred_ids)

        self.total_entities += 1
        self.sum_f_05 += f05
        self.sum_precision += prec
        self.sum_recall += rec

        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn
        self.total_true_matches += len(true_ids)
        self.total_pred_matches += len(pred_ids)

        is_true_singleton = len(true_ids) == 0
        if is_true_singleton:
            self.total_singletons += 1
            if len(pred_ids) == 0:
                self.correct_singletons += 1

        if true_ids == pred_ids:
            self.exact_set_matches += 1

    def compute(self) -> Dict[str, Union[int, float]]:
        """Compute final macro metrics and return dictionary."""
        if self.total_entities == 0:
            return {
                "macro_f_05": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "singleton_accuracy": 0.0,
                "singleton_false_positive_rate": 0.0,
                "exact_set_accuracy": 0.0,
                "total_entities": 0,
                "total_singletons": 0,
                "total_predicted_matches": 0,
                "total_true_matches": 0,
                "total_false_positives": 0,
                "total_false_negatives": 0,
            }

        macro_f05 = self.sum_f_05 / self.total_entities
        macro_prec = self.sum_precision / self.total_entities
        macro_rec = self.sum_recall / self.total_entities
        exact_acc = self.exact_set_matches / self.total_entities

        if self.total_singletons > 0:
            singleton_acc = self.correct_singletons / self.total_singletons
            singleton_fpr = 1.0 - singleton_acc
        else:
            singleton_acc = 1.0
            singleton_fpr = 0.0

        return {
            "macro_f_05": round(macro_f05, 6),
            "macro_precision": round(macro_prec, 6),
            "macro_recall": round(macro_rec, 6),
            "singleton_accuracy": round(singleton_acc, 6),
            "singleton_false_positive_rate": round(singleton_fpr, 6),
            "exact_set_accuracy": round(exact_acc, 6),
            "total_entities": self.total_entities,
            "total_singletons": self.total_singletons,
            "total_predicted_matches": self.total_pred_matches,
            "total_true_matches": self.total_true_matches,
            "total_false_positives": self.total_fp,
            "total_false_negatives": self.total_fn,
        }


def evaluate_mappings(
    ground_truth: Dict[str, Set[str]], predictions: Dict[str, Set[str]]
) -> Dict[str, Union[int, float]]:
    """Evaluate predictions dictionary against ground truth dictionary."""
    evaluator = Evaluator()
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        evaluator.update(s1_id, true_set, pred_set)
    return evaluator.compute()


def evaluate_files(
    ground_truth_path: str,
    predictions_path: str,
    target_s1_ids: Optional[Set[str]] = None,
) -> Dict[str, Union[int, float]]:
    """
    Evaluate prediction TSV file against ground truth TSV file in memory-efficient manner.
    Optionally filters by a subset of target S1 IDs (e.g. validation split).
    """
    # Stream predictions first or index them
    preds: Dict[str, Set[str]] = {}
    with open(predictions_path, "r", encoding="utf-8", errors="replace") as fp:
        next(fp)  # header
        for line in fp:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0]:
                continue
            s1 = parts[0]
            if target_s1_ids is not None and s1 not in target_s1_ids:
                continue
            mids = parts[1].split(",") if len(parts) > 1 and parts[1].strip() else []
            preds[s1] = set(mids)

    evaluator = Evaluator()
    with open(ground_truth_path, "r", encoding="utf-8", errors="replace") as fp:
        next(fp)  # header
        for line in fp:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0]:
                continue
            s1 = parts[0]
            if target_s1_ids is not None and s1 not in target_s1_ids:
                continue
            true_mids = parts[1].split(",") if len(parts) > 1 and parts[1].strip() else []
            true_set = set(true_mids)
            pred_set = preds.get(s1, set())
            evaluator.update(s1, true_set, pred_set)

    return evaluator.compute()
