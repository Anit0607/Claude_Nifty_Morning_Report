"""Direction model: P(close > open) for the session.

A calibrated soft-voting ensemble of a linear baseline (StandardScaler + LogisticRegression)
and two gradient-boosting models (XGBoost + LightGBM). The ensemble is wrapped in
CalibratedClassifierCV (isotonic) so the emitted probability is trustworthy — calibration
is what the scorecard's Brier term rewards, and what option sellers actually need.

Honest expectation: daily close-vs-open direction is close to a coin flip on an efficient
index; the edge is small (low-50s%). The value is a *calibrated* probability plus magnitude,
not a magic hit rate.
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


def _base_ensemble() -> VotingClassifier:
    linear = Pipeline(
        [("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, C=0.5))]
    )
    xgb = XGBClassifier(
        max_depth=3, n_estimators=250, learning_rate=0.05, reg_lambda=1.0,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", tree_method="hist", random_state=7, n_jobs=1,
    )
    lgbm = LGBMClassifier(
        max_depth=3, n_estimators=250, learning_rate=0.05, reg_lambda=1.0,
        min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=7, n_jobs=1, verbose=-1,
    )
    return VotingClassifier(
        estimators=[("linear", linear), ("xgb", xgb), ("lgbm", lgbm)],
        voting="soft", weights=[1.0, 1.0, 1.0],
    )


class DirectionModel:
    def __init__(self) -> None:
        self.clf: CalibratedClassifierCV | None = None
        self._mean_ret_up = 0.0   # avg |ret_co| on up days, for magnitude estimate
        self._mean_ret_dn = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series, ret_co: pd.Series | None = None) -> "DirectionModel":
        self.clf = CalibratedClassifierCV(_base_ensemble(), method="isotonic", cv=4)
        self.clf.fit(X, y)
        if ret_co is not None:
            up = ret_co[y == 1]
            dn = ret_co[y == 0]
            self._mean_ret_up = float(up.mean()) if len(up) else 0.0
            self._mean_ret_dn = float(dn.mean()) if len(dn) else 0.0
        return self

    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        assert self.clf is not None, "model not fitted"
        return self.clf.predict_proba(X)[:, 1]

    def expected_magnitude(self, p_up: float) -> float:
        """Expected (close-open)/open **in the predicted direction**.

        Returns the average up-move when the call is bullish (p_up>=0.5) and the average
        down-move when bearish — so the sign always agrees with the directional call.
        (The old probability-weighted blend could flip sign on low-conviction days, which
        is confusing: "Bullish" but a negative expected move.)
        """
        return self._mean_ret_up if p_up >= 0.5 else self._mean_ret_dn
