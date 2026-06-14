"""The scorecard: quantifies how good predictions are vs. reality.

Operates on a DataFrame of per-prediction records (one row per session). Required columns:

    dir_pred, dir_actual                       # 0/1  (close>open)
    pred_high, pred_low, act_high, act_low     # prices
    p_down, p_sideways, p_up                   # regime probabilities (sum~1)
    reg_actual                                 # "Down" / "Sideways" / "Up"
    conf_overall                               # 0-100 overall confidence
    trade_r                                    # avg P&L across personas in R-multiples (optional)

Produces five sub-scores (0-100) and a weighted composite using config weights:
    direction, trade_pnl, range_hit, calibration, confidence_reliability

"Satisfactory" = rolling composite >= threshold AND critical floors held
(direction accuracy and calibration). Weights, threshold and floors live in
config ``scoring`` so Agent 2 (and the user) can tune them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import load_settings
from src.models.regime_model import CLASSES

# 3-class Brier for a uniform (no-skill) forecast against a one-hot outcome.
_BRIER_NOSKILL = 2.0 / 3.0


@dataclass
class Scorecard:
    composite: float
    direction: float
    trade_pnl: float
    range_hit: float
    calibration: float
    confidence_reliability: float
    n: int
    satisfactory: bool
    detail: dict


def _multiclass_brier(records: pd.DataFrame) -> float:
    probs = records[["p_down", "p_sideways", "p_up"]].to_numpy(dtype=float)
    actual = pd.Categorical(records["reg_actual"], categories=CLASSES).codes
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(actual)), actual] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def _direction_score(records: pd.DataFrame) -> float:
    return float((records["dir_pred"] == records["dir_actual"]).mean())


def _range_components(records: pd.DataFrame) -> tuple[float, float]:
    contained = ((records["act_high"] <= records["pred_high"]) &
                 (records["act_low"] >= records["pred_low"])).mean()
    pred_range = (records["pred_high"] - records["pred_low"]).replace(0, np.nan)
    act_range = (records["act_high"] - records["act_low"]).clip(lower=0)
    efficiency = (act_range / pred_range).clip(upper=1.0).mean()  # penalize over-wide bands
    return float(contained), float(efficiency)


def _confidence_reliability(records: pd.DataFrame) -> float:
    """Do higher-confidence calls actually score better? Returns 0-1."""
    if records["conf_overall"].nunique() < 2 or len(records) < 6:
        return 0.5  # not enough spread to judge; neutral
    correct = (records["dir_pred"] == records["dir_actual"]).astype(float)
    med = records["conf_overall"].median()
    high = correct[records["conf_overall"] >= med].mean()
    low = correct[records["conf_overall"] < med].mean()
    if math.isnan(high) or math.isnan(low):
        return 0.5
    return float(np.clip(0.5 + (high - low), 0.0, 1.0))


def compute(records: pd.DataFrame) -> Scorecard:
    settings = load_settings()
    sc = settings["scoring"]
    w = sc["weights"]

    direction = _direction_score(records)
    contained, efficiency = _range_components(records)
    range_hit = contained * efficiency
    brier = _multiclass_brier(records)
    calibration = max(0.0, 1.0 - brier / _BRIER_NOSKILL)
    conf_rel = _confidence_reliability(records)

    have_trade = "trade_r" in records and records["trade_r"].notna().any()
    if have_trade:
        mean_r = float(records["trade_r"].dropna().mean())
        trade_pnl = 0.5 + 0.5 * math.tanh(mean_r)  # breakeven->0.5, +1R->~0.88
    else:
        trade_pnl = float("nan")

    # Assemble weighted composite over available components (renormalize if trade missing).
    parts = {
        "direction": (direction, w["direction"]),
        "range_hit": (range_hit, w["range_hit"]),
        "calibration": (calibration, w["calibration"]),
        "confidence_reliability": (conf_rel, w["confidence_reliability"]),
    }
    if have_trade:
        parts["trade_pnl"] = (trade_pnl, w["trade_pnl"])

    wsum = sum(weight for _, weight in parts.values())
    composite = 100.0 * sum(val * weight for val, weight in parts.values()) / wsum

    floors = sc["satisfactory"]["floors"]
    satisfactory = (
        composite >= sc["satisfactory"]["composite_threshold"]
        and direction >= floors["direction"]
        and calibration >= floors["calibration"]
    )

    return Scorecard(
        composite=round(composite, 2),
        direction=round(direction * 100, 2),
        trade_pnl=round(trade_pnl * 100, 2) if have_trade else float("nan"),
        range_hit=round(range_hit * 100, 2),
        calibration=round(calibration * 100, 2),
        confidence_reliability=round(conf_rel * 100, 2),
        n=len(records),
        satisfactory=bool(satisfactory),
        detail={
            "range_contained": round(contained * 100, 2),
            "range_efficiency": round(efficiency * 100, 2),
            "brier": round(brier, 4),
            "mean_trade_r": round(float(records["trade_r"].dropna().mean()), 3) if have_trade else None,
        },
    )
