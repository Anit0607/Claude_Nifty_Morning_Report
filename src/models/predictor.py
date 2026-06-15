"""Shared prediction path used by BOTH the backtest and the live Agent 1.

Given a fitted ModelBundle and a built feature frame (one or many rows), produce the
raw model outputs: P(up), regime probabilities, expected close-vs-open magnitude, and
the High/Low band. Centralizing this guarantees the backtested model and the live model
behave identically — no train/serve skew.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_settings
from src.models.confidence import (
    direction_confidence,
    overall_confidence,
    range_confidence,
    regime_confidence,
)
from src.models.registry import ModelBundle


def predict_outputs(bundle: ModelBundle, frame: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of model outputs aligned to ``frame`` index.

    Columns: p_up, dir_pred, exp_magnitude, p_down, p_sideways, p_up_reg,
    pred_high, pred_low, expected_move, conf_direction, conf_regime, conf_range,
    conf_overall.
    """
    X = frame[bundle.feature_cols]

    p_up = bundle.direction.predict_proba_up(X)
    regime = bundle.regime.predict_proba(X)                 # cols Down/Sideways/Up
    band = bundle.range_model.predict_band(frame)           # pred_high/pred_low/expected_move

    out = pd.DataFrame(index=frame.index)
    out["p_up"] = p_up
    out["dir_pred"] = (out["p_up"] >= 0.5).astype(int)
    out["exp_magnitude"] = [bundle.direction.expected_magnitude(p) for p in p_up]
    out["p_down"] = regime["Down"].to_numpy()
    out["p_sideways"] = regime["Sideways"].to_numpy()
    out["p_up_reg"] = regime["Up"].to_numpy()
    out["pred_high"] = band["pred_high"]
    out["pred_low"] = band["pred_low"]
    out["expected_move"] = band["expected_move"]

    # Asymmetric tilt: tighten the COUNTER-trend side of the band, scaled by directional
    # conviction. On a bearish call we expect little upside, so pull the high in; on a
    # bullish call, pull the low in. The trend side keeps full width.
    tilt_max = float(load_settings()["tunable"]["range"].get("asym_tilt", 0.0))
    if tilt_max > 0:
        open_ = frame["open"].to_numpy(dtype=float)
        pu = out["p_up"].to_numpy(dtype=float)
        tilt = tilt_max * (np.abs(pu - 0.5) * 2.0)            # 0..tilt_max by conviction
        ph, pl = out["pred_high"].to_numpy(dtype=float), out["pred_low"].to_numpy(dtype=float)
        bearish = pu < 0.5
        ph = np.where(bearish, open_ + (ph - open_) * (1 - tilt), ph)   # tighten upside
        pl = np.where(~bearish, open_ - (open_ - pl) * (1 - tilt), pl)  # tighten downside
        out["pred_high"], out["pred_low"] = ph, pl

    # Confidence (row-wise; small N so a loop is fine and keeps the logic readable).
    conf_dir, conf_reg, conf_rng, conf_all = [], [], [], []
    for i, idx in enumerate(frame.index):
        cd = direction_confidence(
            float(p_up[i]), float(frame.at[idx, "gap_pct"]),
            float(frame.at[idx, "markov_p_up"]), float(frame.at[idx, "markov_p_down"]),
        )
        cr = regime_confidence(
            float(regime.at[idx, "Down"]), float(regime.at[idx, "Sideways"]),
            float(regime.at[idx, "Up"]),
        )
        cg = range_confidence(
            float(frame.at[idx, "vix_regime"]), float(frame.at[idx, "vix_change"]),
        )
        conf_dir.append(cd)
        conf_reg.append(cr)
        conf_rng.append(cg)
        conf_all.append(overall_confidence(cd, cr, cg))
    out["conf_direction"] = conf_dir
    out["conf_regime"] = conf_reg
    out["conf_range"] = conf_rng
    out["conf_overall"] = conf_all
    return out
