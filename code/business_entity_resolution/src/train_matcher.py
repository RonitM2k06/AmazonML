"""
Matcher Training Engine for Business Entity Resolution.
Supports LightGBM, CatBoost, and XGBoost training on identical candidate pair datasets,
class imbalance handling (scale_pos_weight), feature importance analysis,
and evaluation of pairwise and entity-level Macro F0.5.
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"

import time
import pickle
import json
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import psutil

# Ensure UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def compute_pairwise_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute precision, recall, f0.5, and log_loss using pure numpy."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = 0.25 * prec + rec
    f05 = (1.25 * prec * rec / denom) if denom > 0 else 0.0
    
    eps = 1e-15
    p = np.clip(y_prob, eps, 1.0 - eps)
    loss = float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1.0 - p)))
    return round(prec, 6), round(rec, 6), round(f05, 6), round(loss, 6)


def get_current_ram_mb() -> float:
    """Return memory used by current process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def train_matcher(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    scale_pos_weight: Optional[float] = None,
    num_trees: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    random_state: int = 42,
) -> Tuple[object, Dict[str, Union[float, int, str]]]:
    """
    Train a gradient boosting model (LightGBM, CatBoost, or XGBoost)
    with identical candidate features and tracking metrics.
    """
    model_type = model_type.lower()
    t_start = time.time()
    ram_start = get_current_ram_mb()

    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    implied_scale_pos = round(neg_count / max(1, pos_count), 2)
    spw = scale_pos_weight if scale_pos_weight is not None else implied_scale_pos

    print(f"\nTraining {model_type.upper()} Classifier...")
    print(f"  Train Samples: {len(y_train):,} ({pos_count:,} pos, {neg_count:,} neg, ratio 1:{implied_scale_pos})")
    print(f"  Validation Samples: {len(y_val):,}")
    print(f"  Feature Dimensions: {X_train.shape[1]}")
    print(f"  Scale Pos Weight: {spw}")

    # Subsample validation set for early stopping to guarantee low memory and fast tree iterations
    if len(X_val) > 30000:
        np.random.seed(random_state)
        val_sub_idx = np.random.choice(len(X_val), 30000, replace=False)
        X_val_eval = X_val[val_sub_idx]
        y_val_eval = y_val[val_sub_idx]
    else:
        X_val_eval = X_val
        y_val_eval = y_val

    if model_type == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=num_trees,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=min(63, 2**max_depth - 1),
            scale_pos_weight=spw,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val_eval, y_val_eval)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

    elif model_type == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=num_trees,
            learning_rate=learning_rate,
            max_depth=max_depth,
            scale_pos_weight=spw,
            random_state=random_state,
            n_jobs=-1,
            early_stopping_rounds=30,
            eval_metric="logloss",
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val_eval, y_val_eval)],
            verbose=False,
        )

    elif model_type == "catboost":
        import catboost as cb
        model = cb.CatBoostClassifier(
            iterations=num_trees,
            learning_rate=learning_rate,
            depth=max_depth,
            scale_pos_weight=spw,
            random_seed=random_state,
            thread_count=-1,
            early_stopping_rounds=30,
            verbose=False,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val_eval, y_val_eval),
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose lightgbm, xgboost, or catboost.")

    train_time = round(time.time() - t_start, 2)
    ram_peak = round(get_current_ram_mb() - ram_start, 2)

    # Chunked inference to maintain low RAM and measure throughput
    t_inf_start = time.time()
    val_probs_list = []
    chunk_size = 250000
    for c_start in range(0, len(X_val), chunk_size):
        chunk = X_val[c_start : c_start + chunk_size]
        val_probs_list.append(model.predict_proba(chunk)[:, 1])
    val_probs = np.concatenate(val_probs_list)
    inference_time = round(time.time() - t_inf_start, 4)
    throughput = round(len(X_val) / max(0.0001, inference_time), 1)

    # Standard pairwise metrics at 0.50 threshold
    val_preds_50 = (val_probs >= 0.50).astype(int)
    pair_prec, pair_rec, pair_f05, loss = compute_pairwise_metrics(y_val, val_preds_50, val_probs)

    # Feature importances
    if hasattr(model, "feature_importances_"):
        raw_importances = model.feature_importances_
        norm_imp = [round(float(imp / max(1e-9, np.sum(raw_importances)) * 100), 2) for imp in raw_importances]
        feat_imp_dict = dict(zip(feature_names, norm_imp))
    else:
        feat_imp_dict = {}

    stats = {
        "model_type": model_type,
        "train_time_sec": train_time,
        "inference_time_sec": inference_time,
        "throughput_pairs_per_sec": throughput,
        "ram_delta_mb": ram_peak,
        "num_features": X_train.shape[1],
        "pair_precision_05": pair_prec,
        "pair_recall_05": pair_rec,
        "pair_f05_05": pair_f05,
        "val_log_loss": loss,
        "feature_importances": feat_imp_dict,
    }

    print(f"  Done in {train_time}s. Inference: {throughput:,.1f} pairs/s. Pair F0.5 (at 0.50): {pair_f05}")

    return model, stats


def save_matcher(output_path: str, model_obj, stats: Optional[Dict] = None):
    """Persist trained model artifact to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model_obj, f)
    if stats:
        stats_path = output_path + ".stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
    print(f"Saved model to {output_path}")
