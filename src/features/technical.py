"""Momentum / trend / mean-reversion features (beyond the existing skills).

Motivated by the cross-sectional and time-series momentum literature (trend persistence)
and short-horizon mean reversion. All values are **lagged one session** so they are known
at the 09:15 open — no lookahead. The weekly-expiry flag captures option-pinning, which
biases expiry sessions toward the "Sideways" regime.

Expiry note: Nifty weekly expiry was Thursday for most of 2018-2026 (with brief regime
changes). We approximate it with a Thursday flag — a documented simplification; the live
path can later use the exact expiry calendar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append momentum/trend/mean-reversion features + expiry flag (all open-time-safe)."""
    out = df.copy()
    close = out["close"]

    out["mom_5"] = np.log(close / close.shift(5)).shift(1)
    out["mom_20"] = np.log(close / close.shift(20)).shift(1)
    out["dist_sma20"] = (close / close.rolling(20).mean() - 1.0).shift(1)
    out["rsi_14"] = rsi(close, 14).shift(1)
    out["is_thursday"] = (out.index.dayofweek == 3).astype(int)  # weekly-expiry proxy
    return out
