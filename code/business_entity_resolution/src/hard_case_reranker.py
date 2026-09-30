"""
EXP_N030 — Hard-Case Cross-Source Reranker Module.

Trains a specialized gradient boosting reranker specifically on hard failure cases:
- Low probability true matches (p < 0.70)
- Zero name token overlap true matches
- Cross-script and transliteration true matches
- Deep-rank multi-matches
Against hard negatives:
- Same-name different-branch collisions
- Same-address different-business collisions
- High neural-cosine non-matches
"""

from __future__ import annotations

import os
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class HardCaseReranker:
    """
    Reranks candidates in the difficult decision boundary region.
    """

    def __init__(self, max_depth: int = 5, eta: float = 0.03, num_round: int = 150):
        self.max_depth = max_depth
        self.eta = eta
        self.num_round = num_round
        self.bst = None

    def fit_cv(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> np.ndarray:
        """K-Fold cross-validation prediction to prevent out-of-fold leakage."""
        oof_preds = np.zeros(len(y), dtype=np.float32)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        params = {
            "max_depth": self.max_depth,
            "eta": self.eta,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "verbosity": 0,
            "nthread": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        for train_idx, val_idx in kf.split(X):
            dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx])
            dval = xgb.DMatrix(X[val_idx])
            bst = xgb.train(params, dtrain, num_boost_round=self.num_round)
            oof_preds[val_idx] = bst.predict(dval)
        
        # Train final model on all data
        dall = xgb.DMatrix(X, label=y)
        self.bst = xgb.train(params, dall, num_boost_round=self.num_round)
        return oof_preds

    def predict(self, X: np.ndarray) -> np.ndarray:
        dmat = xgb.DMatrix(X)
        return self.bst.predict(dmat)
