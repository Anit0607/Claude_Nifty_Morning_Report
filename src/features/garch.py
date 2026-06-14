"""Asymmetric GARCH conditional volatility (EGARCH + TGARCH).

The Nifty exhibits a leverage effect (down moves raise vol more than up moves), so
asymmetric models fit better than plain GARCH — consistent with the GARCH/RNN and
range-GARCH studies on Nifty. We fit both EGARCH(1,1) and TGARCH/GJR(1,1) on daily
close-to-close log returns and average them for a more robust conditional-vol estimate.

Two use modes:
  * ``conditional_vol_series`` — in-sample conditional sigma for every day, used as a
    training feature. Parameters are full-sample, but sigma_t depends only on info up to
    t-1, which is acceptable for a feature (strict walk-forward lives in the backtest).
  * ``forecast_next_sigma`` — fit on a trailing window and forecast the *next* day's
    sigma, used live by Agent 1.

All sigmas are returned as **daily fractions** (e.g. 0.009 == 0.9% per day).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# arch wants returns scaled (it warns when |y| is tiny). We work in percent then /100.
_SCALE = 100.0


def _log_returns_pct(df: pd.DataFrame) -> pd.Series:
    ret = np.log(df["close"] / df["prev_close"]) * _SCALE
    return ret.dropna()


# Both asymmetric specifications, fitted and then averaged for robustness:
#   EGARCH(1,1)         — leverage via log-variance (sign term)
#   TGARCH/GJR(1,1)     — threshold term on the conditional std dev (power=1.0, o=1)
_SPECS = (
    {"vol": "EGARCH", "p": 1, "o": 1, "q": 1},
    {"vol": "GARCH", "p": 1, "o": 1, "q": 1, "power": 1.0},  # TGARCH (Zakoian)
)


def _fit(returns_pct: pd.Series, spec: dict):
    from arch import arch_model

    model = arch_model(returns_pct, mean="Constant", dist="studentst", **spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(disp="off")


def conditional_vol_series(df: pd.DataFrame) -> pd.Series:
    """In-sample conditional daily sigma (fraction): mean of EGARCH and TGARCH.

    Falls back to rolling realized vol if both GARCH fits fail (e.g. too little data).
    """
    returns = _log_returns_pct(df)
    sigmas = []
    for spec in _SPECS:
        try:
            res = _fit(returns, spec)
            sigmas.append(np.asarray(res.conditional_volatility) / _SCALE)
        except Exception:
            continue
    if not sigmas:
        fallback = (np.log(df["close"] / df["prev_close"])).rolling(14).std(ddof=1)
        return fallback.rename("garch_sigma")
    avg = np.mean(np.column_stack(sigmas), axis=1)
    return pd.Series(avg, index=returns.index, name="garch_sigma").reindex(df.index)


def forecast_next_sigma(df: pd.DataFrame, lookback: int = 750) -> float:
    """Forecast next session's daily sigma (fraction): mean of EGARCH and TGARCH forecasts.

    Used live: fit on the most recent ``lookback`` sessions, forecast 1 step ahead.
    """
    returns = _log_returns_pct(df).iloc[-lookback:]
    if len(returns) < 100:
        return float(np.log(df["close"] / df["prev_close"]).tail(14).std(ddof=1))
    forecasts = []
    for spec in _SPECS:
        try:
            res = _fit(returns, spec)
            fc = res.forecast(horizon=1, reindex=False)
            forecasts.append(float(np.sqrt(fc.variance.values[-1, 0]) / _SCALE))
        except Exception:
            continue
    if not forecasts:
        return float(returns.std(ddof=1) / _SCALE)
    return float(np.mean(forecasts))
