"""
Unit tests for the evaluation module (Macro F_0.5, Precision, Recall, Singletons).
"""

import pytest
import math
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from evaluation import compute_entity_metrics, Evaluator, evaluate_mappings


def test_empty_truth_and_empty_pred_singleton():
    """True singleton correctly identified: F_0.5 = 1.0, P = 1.0, R = 1.0."""
    f05, prec, rec, tp, fp, fn = compute_entity_metrics(set(), set())
    assert f05 == 1.0
    assert prec == 1.0
    assert rec == 1.0
    assert tp == 0
    assert fp == 0
    assert fn == 0


def test_empty_truth_and_nonempty_pred_false_merge():
    """True singleton falsely merged: F_0.5 = 0.0, P = 0.0, R = 0.0."""
    f05, prec, rec, tp, fp, fn = compute_entity_metrics(set(), {"S2-101"})
    assert f05 == 0.0
    assert prec == 0.0
    assert rec == 0.0
    assert tp == 0
    assert fp == 1
    assert fn == 0


def test_nonempty_truth_and_empty_pred_missed():
    """True matches missed (predicted empty): F_0.5 = 0.0, P = 0.0, R = 0.0."""
    f05, prec, rec, tp, fp, fn = compute_entity_metrics({"S2-101", "S3-202"}, set())
    assert f05 == 0.0
    assert prec == 0.0
    assert rec == 0.0
    assert tp == 0
    assert fp == 0
    assert fn == 2


def test_perfect_match():
    """Perfect match: F_0.5 = 1.0, P = 1.0, R = 1.0."""
    ids = {"S2-101", "S3-202"}
    f05, prec, rec, tp, fp, fn = compute_entity_metrics(ids, ids)
    assert f05 == 1.0
    assert prec == 1.0
    assert rec == 1.0
    assert tp == 2
    assert fp == 0
    assert fn == 0


def test_complete_mismatch():
    """Both nonempty but disjoint: F_0.5 = 0.0, P = 0.0, R = 0.0."""
    truth = {"S2-101"}
    pred = {"S3-999"}
    f05, prec, rec, tp, fp, fn = compute_entity_metrics(truth, pred)
    assert f05 == 0.0
    assert prec == 0.0
    assert rec == 0.0
    assert tp == 0
    assert fp == 1
    assert fn == 1


def test_official_example():
    """
    Official example from Challenge Problem Statement:
    - Truth = [S2-00047, S3-00812]
    - Pred = [S2-00047, S2-00193, S3-00812]
    - Precision = 2/3 = 0.667, Recall = 1.0
    - F_0.5 = 5/7 = 0.714286
    """
    truth = {"S2-00047", "S3-00812"}
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    f05, prec, rec, tp, fp, fn = compute_entity_metrics(truth, pred)

    expected_prec = 2.0 / 3.0
    expected_rec = 1.0
    expected_f05 = (1.25 * expected_prec * expected_rec) / (0.25 * expected_prec + expected_rec)
    assert math.isclose(expected_f05, 5.0 / 7.0, rel_tol=1e-5)
    assert math.isclose(f05, expected_f05, rel_tol=1e-5)
    assert math.isclose(prec, expected_prec, rel_tol=1e-5)
    assert math.isclose(rec, expected_rec, rel_tol=1e-5)
    assert tp == 2
    assert fp == 1
    assert fn == 0


def test_macro_evaluation():
    """Test full macro aggregation across multiple entities."""
    gt = {
        "S1-1": {"S2-10", "S3-10"},           # perfect match: F0.5 = 1.0
        "S1-2": set(),                         # true singleton, correctly pred: F0.5 = 1.0
        "S1-3": set(),                         # true singleton, false merge: F0.5 = 0.0
        "S1-4": {"S2-20"},                     # missed: F0.5 = 0.0
    }
    preds = {
        "S1-1": {"S2-10", "S3-10"},
        "S1-2": set(),
        "S1-3": {"S2-99"},
        "S1-4": set(),
    }
    res = evaluate_mappings(gt, preds)
    # 4 entities: [1.0, 1.0, 0.0, 0.0] -> macro_f_05 = 0.5
    assert res["total_entities"] == 4
    assert res["macro_f_05"] == 0.5
    assert res["total_singletons"] == 2
    assert res["singleton_accuracy"] == 0.5
    assert res["singleton_false_positive_rate"] == 0.5
    assert res["exact_set_accuracy"] == 0.5
