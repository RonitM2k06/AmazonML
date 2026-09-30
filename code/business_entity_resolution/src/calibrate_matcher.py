"""
Probability Calibration Engine for Business Entity Resolution Matchers.
Implements Platt scaling (sigmoid calibration) using pure NumPy gradient descent,
with ZERO external dependencies and zero risk of OpenMP thread deadlocks.
"""

import os
import sys
import pickle
from typing import Dict, Optional, Tuple, Union
import numpy as np


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier score loss: mean squared difference."""
    return float(np.mean((y_true - y_prob) ** 2))


def log_loss_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute binary cross-entropy log loss."""
    eps = 1e-15
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


class PlattScaler:
    """
    Platt Scaling (Sigmoidal calibration on model logits).
    Fits P(y=1|p) = 1 / (1 + exp(-(A * logit(p) + B))) via pure NumPy GD.
    """

    def __init__(self):
        self.a = 1.0
        self.b = 0.0

    def fit(self, raw_probs: np.ndarray, y: np.ndarray, epochs: int = 100, lr: float = 0.05):
        eps = 1e-7
        p = np.clip(raw_probs, eps, 1.0 - eps)
        logits = np.log(p / (1.0 - p))

        a = 1.0
        b = 0.0
        for _ in range(epochs):
            z = np.clip(a * logits + b, -20.0, 20.0)
            sig = 1.0 / (1.0 + np.exp(-z))
            err = sig - y
            grad_a = np.mean(err * logits)
            grad_b = np.mean(err)
            a -= lr * grad_a
            b -= lr * grad_b

        self.a = float(a)
        self.b = float(b)
        return self

    def predict_proba(self, raw_probs: np.ndarray) -> np.ndarray:
        eps = 1e-7
        p = np.clip(raw_probs, eps, 1.0 - eps)
        logits = np.log(p / (1.0 - p))
        z = np.clip(self.a * logits + self.b, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-z))


class CalibratedModelWrapper:
    """Wraps a base model with a probability calibrator."""

    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw_probs = self.base_model.predict_proba(X)[:, 1]
        cal_probs = self.calibrator.predict_proba(raw_probs)
        return np.column_stack([1.0 - cal_probs, cal_probs])


def calibrate_model(
    base_model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "sigmoid",
) -> Tuple[object, Dict[str, float]]:
    """
    Calibrate a trained classifier on validation features and labels.
    """
    raw_probs = base_model.predict_proba(X_val)[:, 1]

    calibrator = PlattScaler().fit(raw_probs, y_val)
    cal_probs = calibrator.predict_proba(raw_probs)

    raw_brier = brier_score(y_val, raw_probs)
    raw_logloss = log_loss_score(y_val, raw_probs)

    cal_brier = brier_score(y_val, cal_probs)
    cal_logloss = log_loss_score(y_val, cal_probs)

    wrapper = CalibratedModelWrapper(base_model, calibrator)

    metrics = {
        "method": method,
        "raw_brier_score": round(raw_brier, 6),
        "calibrated_brier_score": round(cal_brier, 6),
        "raw_log_loss": round(raw_logloss, 6),
        "calibrated_log_loss": round(cal_logloss, 6),
        "brier_improvement": round((raw_brier - cal_brier) / max(1e-9, raw_brier) * 100, 2),
    }

    return wrapper, metrics


def save_calibrated_model(output_path: str, model_obj, metrics: Optional[Dict] = None):
    """Save calibrated model object to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model_obj, f)
    print(f"Saved calibrated model to {output_path}")
