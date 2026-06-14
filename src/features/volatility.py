"""Range-based volatility estimators and realized-vol features.

Research (range-based GARCH for Indian intraday vol) finds range estimators —
especially Garman-Klass-Yang-Zhang (GKYZ) and Yang-Zhang (YZ) — forecast Nifty
volatility better than close-to-close returns, because they use the full OHLC bar
and the overnight jump. All estimators below return **daily variance** (fraction^2);
take ``sqrt`` for a daily sigma, multiply by ``sqrt(252)`` to annualize.

Inputs are the training/daily frame from ``historical.py`` (columns: open, high,
low, close, prev_close). Every function returns a pandas Series aligned to the frame
index, so they compose cleanly in ``features/builder.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_LN2 = np.log(2.0)


def _log(a, b):
    """Safe natural log of a/b for positive price series."""
    return np.log(a / b)


def parkinson_var(df: pd.DataFrame) -> pd.Series:
    """Parkinson (1980): uses the high-low range only."""
    hl = _log(df["high"], df["low"])
    return (hl**2) / (4.0 * _LN2)


def garman_klass_var(df: pd.DataFrame) -> pd.Series:
    """Garman-Klass (1980): high-low + open-close, no overnight gap."""
    hl = _log(df["high"], df["low"])
    co = _log(df["close"], df["open"])
    return 0.5 * hl**2 - (2.0 * _LN2 - 1.0) * co**2


def rogers_satchell_var(df: pd.DataFrame) -> pd.Series:
    """Rogers-Satchell (1991): drift-independent OHLC estimator."""
    ho = _log(df["high"], df["open"])
    hc = _log(df["high"], df["close"])
    lo = _log(df["low"], df["open"])
    lc = _log(df["low"], df["close"])
    return hc * ho + lc * lo


def gkyz_var(df: pd.DataFrame) -> pd.Series:
    """Garman-Klass-Yang-Zhang: GK plus the overnight (prev_close->open) jump.

    Best single-day estimator for the Indian market per the range-GARCH literature.
    """
    overnight = _log(df["open"], df["prev_close"]) ** 2
    hl = _log(df["high"], df["low"])
    co = _log(df["close"], df["open"])
    return overnight + 0.5 * hl**2 - (2.0 * _LN2 - 1.0) * co**2


def yang_zhang_var(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Yang-Zhang (2000) rolling estimator: overnight + open-close + Rogers-Satchell.

    YZ is minimum-variance and handles both drift and the opening jump. It is a
    *windowed* estimator; ``window`` rows of history are required.
    """
    o_c_prev = _log(df["open"], df["prev_close"])      # overnight return
    c_o = _log(df["close"], df["open"])                # open-to-close return

    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    overnight_var = o_c_prev.rolling(n).var(ddof=1)
    openclose_var = c_o.rolling(n).var(ddof=1)
    rs = rogers_satchell_var(df).rolling(n).mean()

    return overnight_var + k * openclose_var + (1.0 - k) * rs


def true_range(df: pd.DataFrame) -> pd.Series:
    """Wilder's True Range in price points (not log)."""
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["prev_close"]).abs()
    lc = (df["low"] - df["prev_close"]).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range (points) — Wilder smoothing via EMA."""
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / window, adjust=False).mean()


def realized_vol(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Rolling close-to-close daily volatility (fraction)."""
    ret = _log(df["close"], df["prev_close"])
    return ret.rolling(window).std(ddof=1)


def vix_implied_daily_move(df: pd.DataFrame) -> pd.Series:
    """Expected 1-day move (fraction) implied by India VIX.

    VIX is an annualized volatility in percent; de-annualize by sqrt(252).
    """
    return (df["vix"] / 100.0) / np.sqrt(252.0)


def add_volatility_features(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Append all volatility features (as daily sigmas / points) to a copy of df."""
    out = df.copy()
    out["sigma_parkinson"] = np.sqrt(parkinson_var(df).clip(lower=0))
    out["sigma_gk"] = np.sqrt(garman_klass_var(df).clip(lower=0))
    out["sigma_rs"] = np.sqrt(rogers_satchell_var(df).clip(lower=0))
    out["sigma_gkyz"] = np.sqrt(gkyz_var(df).clip(lower=0))
    out["sigma_yz"] = np.sqrt(yang_zhang_var(df, window).clip(lower=0))
    out["atr"] = atr(df, window)
    out["realized_vol"] = realized_vol(df, window)
    out["vix_daily_move"] = vix_implied_daily_move(df)
    # Smoothed range proxies (rolling means) used by the range model blend.
    out["sigma_gkyz_avg"] = out["sigma_gkyz"].rolling(window).mean()
    return out
