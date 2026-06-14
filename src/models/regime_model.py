"""Regime / probability model: calibrated P(Down), P(Sideways), P(Up).

This is output #3 — the Up/Down/Sideways probability triple. Same ensemble philosophy as
the direction model but multiclass, with isotonic calibration so the three probabilities
are individually meaningful (the scorecard scores them with a multiclass Brier score).

Classes follow close-vs-open with a neutral band (config ``scoring.sideways_pct``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

CLASSES = ["Down", "Sideways", "Up"]


def _base_ensemble() -> VotingClassifier:
    linear = Pipeline(
        [("scale", StandardScaler()),
         ("lr", LogisticRegression(max_iter=1000, C=0.5))]
    )
    xgb = XGBClassifier(
        max_depth=3, n_estimators=250, learning_rate=0.05, reg_lambda=1.0,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", tree_method="hist", random_state=11, n_jobs=1,
    )
    lgbm = LGBMClassifier(
        max_depth=3, n_estimators=250, learning_rate=0.05, reg_lambda=1.0,
        min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=11, n_jobs=1, verbose=-1,
    )
    return VotingClassifier(
        estimators=[("linear", linear), ("xgb", xgb), ("lgbm", lgbm)],
        voting="soft", weights=[1.0, 1.0, 1.0],
    )


class RegimeModel:
    def __init__(self) -> None:
        self.clf: CalibratedClassifierCV | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RegimeModel":
        # Ensure a stable class order regardless of training-set frequencies.
        y_cat = pd.Categorical(y, categories=CLASSES)
        self.clf = CalibratedClassifierCV(_base_ensemble(), method="isotonic", cv=4)
        self.clf.fit(X, y_cat)
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with columns Down/Sideways/Up (probabilities)."""
        assert self.clf is not None, "model not fitted"
        proba = self.clf.predict_proba(X)
        # Map model's class order back to canonical CLASSES order.
        order = list(self.clf.classes_)
        idx = [order.index(c) for c in CLASSES]
        return pd.DataFrame(proba[:, idx], columns=CLASSES, index=X.index)
