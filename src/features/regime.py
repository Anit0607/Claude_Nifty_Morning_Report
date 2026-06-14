"""Regime features: Markov transition state, VIX regime, gap classification.

These carry forward the *useful logic* from the existing skills (Markov regime model,
VIX bands, gap classification) but expressed as engineered numeric features that feed
the trained models — not as the model itself.

Markov note: the transition matrix is estimated on the supplied history. For training
features it is full-sample (a mild, documented simplification); live and in the
backtest harness it is estimated on past data only, so there is no forward leakage in
the evaluated path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Daily close-to-close return state thresholds (fraction).
_UP_THR = 0.004      # > +0.4%  => Up
_DOWN_THR = -0.004   # < -0.4%  => Down
STATES = ["Down", "Sideways", "Up"]

# India VIX regime bins (level). Ordinal 0..3.
_VIX_BINS = [0, 13, 16, 20, np.inf]
_VIX_LABELS = ["Low", "Medium", "High", "Extreme"]


def daily_state(returns: pd.Series) -> pd.Series:
    """Classify close-to-close returns into Down / Sideways / Up."""
    s = pd.Series("Sideways", index=returns.index, dtype=object)
    s[returns > _UP_THR] = "Up"
    s[returns < _DOWN_THR] = "Down"
    return s


def transition_matrix(states: pd.Series) -> pd.DataFrame:
    """Estimate a 3x3 first-order Markov transition matrix P(next | current)."""
    idx = {st: i for i, st in enumerate(STATES)}
    counts = np.ones((3, 3))  # Laplace smoothing so no row is empty
    cur = states.map(idx).to_numpy()
    for a, b in zip(cur[:-1], cur[1:]):
        if np.isnan(a) or np.isnan(b):
            continue
        counts[int(a), int(b)] += 1
    mat = counts / counts.sum(axis=1, keepdims=True)
    return pd.DataFrame(mat, index=STATES, columns=STATES)


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append gap, VIX-regime and Markov next-state probabilities to a copy of df."""
    out = df.copy()

    # --- Gap (open vs previous close) ---
    out["gap_pct"] = (out["open"] - out["prev_close"]) / out["prev_close"]
    # Gap normalized by recent daily volatility => "how unusual is this gap".
    recent_sigma = np.log(out["close"] / out["prev_close"]).rolling(14).std(ddof=1)
    out["gap_zscore"] = out["gap_pct"] / recent_sigma.replace(0, np.nan)

    # --- VIX regime (ordinal) and 1-day VIX change ---
    out["vix_regime"] = pd.cut(out["vix"], bins=_VIX_BINS, labels=False, right=False)
    out["vix_change"] = out["vix"] - out["prev_vix"]

    # --- Markov: yesterday's state -> probabilities for today ---
    ret = np.log(out["close"] / out["prev_close"])
    states = daily_state(ret)
    out["state"] = states
    prev_state = states.shift(1)

    tmat = transition_matrix(states)
    for target in STATES:
        col = f"markov_p_{target.lower()}"
        out[col] = prev_state.map(tmat[target]).astype(float)

    return out
