"""Fit a ModelBundle from a built feature frame.

Shared by scripts/train_initial.py (full history) and the backtest harness (each
walk-forward refit), so the trained object is identical in both paths.
"""
from __future__ import annotations

import pandas as pd

from src.features.builder import FEATURE_COLS
from src.models.direction_model import DirectionModel
from src.models.range_model import RangeModel
from src.models.regime_model import RegimeModel
from src.models.registry import ModelBundle


def fit_bundle(train_frame: pd.DataFrame, feature_cols: list[str] | None = None,
               metadata: dict | None = None) -> ModelBundle:
    feature_cols = feature_cols or FEATURE_COLS
    X = train_frame[feature_cols]

    direction = DirectionModel().fit(X, train_frame["y_dir"], train_frame["ret_co"])
    regime = RegimeModel().fit(X, train_frame["y_reg"])
    range_model = RangeModel().fit(train_frame)

    meta = {
        "n_train": int(len(train_frame)),
        "train_start": str(train_frame.index.min().date()),
        "train_end": str(train_frame.index.max().date()),
        "feature_cols": feature_cols,
    }
    if metadata:
        meta.update(metadata)

    return ModelBundle(
        direction=direction,
        regime=regime,
        range_model=range_model,
        feature_cols=feature_cols,
        metadata=meta,
    )
