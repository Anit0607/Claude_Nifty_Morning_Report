"""Assemble the model-ready feature matrix and prediction targets.

Critical correctness rule: every feature must be knowable at the 09:15 open of the day
being predicted. So features derived from a full day's OHLC (the range-based vol
estimators, ATR, realized vol) are **lagged by one session** — they describe the last
*completed* day. Point-in-time-correct signals (today's gap vs prev close, today's VIX
level, Markov probabilities conditioned on yesterday's state, day-of-week) are used as-is.

Targets (all measured from today's open, since a trader acts at the open):
    y_dir   : 1 if close > open else 0                      (direction)
    y_reg   : Down / Sideways / Up by close-vs-open vs band (regime / probability model)
    ret_co  : (close - open) / open                         (signed magnitude)
    up_exc  : (high - open) / open      >= 0                (upper excursion, range)
    dn_exc  : (open - low)  / open      >= 0                (lower excursion, range)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_settings
from src.data.intraday_history import OR_FEATURE_COLS
from src.features.regime import add_regime_features, _VIX_BINS
from src.features.volatility import add_volatility_features
from src.features.technical import add_technical_features
from src.features.garch import conditional_vol_series

# Features fed to the trained models. Order is fixed so the live path matches training.
FEATURE_COLS = [
    "gap_pct",
    "gap_zscore",
    "vix",
    "vix_change",
    "vix_regime",
    "vix_daily_move",
    "prev_ret",
    "prev_sigma_gkyz",
    "prev_sigma_yz",
    "prev_realized_vol",
    "prev_atr_pct",
    "garch_sigma",
    "markov_p_down",
    "markov_p_sideways",
    "markov_p_up",
    "dow",
    "mom_5",
    "mom_20",
    "dist_sma20",
    "rsi_14",
    "is_thursday",
]


def build_feature_frame(df: pd.DataFrame, *, with_garch: bool = True,
                        opening_range: pd.DataFrame | None = None,
                        require_targets: bool = True) -> pd.DataFrame:
    """Return a frame with features + targets, ready for training/backtest.

    If ``opening_range`` (a date-indexed OR table) is supplied, its columns are merged in
    and become available as extra features (see ``OR_FEATURE_COLS``).

    ``require_targets=False`` keeps rows whose outcome (y_dir) is not yet known — used live
    by Agent 1 to build today's feature row before the session has closed.
    """
    settings = load_settings()
    sideways_pct = settings["scoring"]["sideways_pct"]

    feat = add_volatility_features(df)
    feat = add_regime_features(feat)
    feat = add_technical_features(feat)

    # --- VIX features lagged to the last completed session ---
    # Today's VIX close is NOT known at the 09:15 open (it reflects today's move), so we
    # use the previous session's VIX level/change. Live, Agent 1 feeds the same (prior
    # close) value, keeping training and serving consistent and leakage-free.
    feat["vix"] = feat["prev_vix"]
    feat["vix_change"] = feat["prev_vix"].diff()
    feat["vix_daily_move"] = (feat["prev_vix"] / 100.0) / np.sqrt(252.0)
    feat["vix_regime"] = pd.cut(feat["prev_vix"], bins=_VIX_BINS, labels=False, right=False)

    # --- lagged (last completed day) volatility features ---
    feat["prev_ret"] = np.log(feat["close"] / feat["prev_close"]).shift(1)
    feat["prev_sigma_gkyz"] = feat["sigma_gkyz"].shift(1)
    feat["prev_sigma_yz"] = feat["sigma_yz"].shift(1)
    feat["prev_realized_vol"] = feat["realized_vol"].shift(1)
    feat["prev_atr_pct"] = (feat["atr"] / feat["close"]).shift(1)

    # --- conditional volatility forecast for today (info up to yesterday) ---
    if with_garch:
        feat["garch_sigma"] = conditional_vol_series(df)
    else:
        feat["garch_sigma"] = feat["prev_realized_vol"]

    feat["dow"] = feat.index.dayofweek

    # --- targets (measured from today's open) ---
    feat["ret_co"] = (feat["close"] - feat["open"]) / feat["open"]
    feat["y_dir"] = (feat["close"] > feat["open"]).astype(int)
    feat["up_exc"] = (feat["high"] - feat["open"]) / feat["open"]
    feat["dn_exc"] = (feat["open"] - feat["low"]) / feat["open"]

    reg = pd.Series("Sideways", index=feat.index, dtype=object)
    reg[feat["ret_co"] > sideways_pct] = "Up"
    reg[feat["ret_co"] < -sideways_pct] = "Down"
    feat["y_reg"] = reg

    feature_cols = list(FEATURE_COLS)
    if opening_range is not None:
        feat = feat.join(opening_range[OR_FEATURE_COLS], how="left")
        feature_cols += OR_FEATURE_COLS

    cols = feature_cols + ["ret_co", "y_dir", "y_reg", "up_exc", "dn_exc",
                           "open", "close", "high", "low", "prev_close"]
    subset = feature_cols + (["y_dir"] if require_targets else [])
    out = feat[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=subset)
    return out


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Select just the FEATURE_COLS (in fixed order) from a built frame."""
    return frame[FEATURE_COLS]


if __name__ == "__main__":  # quick manual check
    from src.data.historical import load_training_frame

    ff = build_feature_frame(load_training_frame())
    print(f"feature frame rows: {len(ff)}  cols: {len(ff.columns)}")
    print(ff[FEATURE_COLS].describe().T.to_string())
    print("\nregime balance:")
    print(ff["y_reg"].value_counts())
    print("direction balance:")
    print(ff["y_dir"].value_counts())
