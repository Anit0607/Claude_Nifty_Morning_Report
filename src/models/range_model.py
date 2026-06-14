"""Range model: expected intraday High/Low band around the open.

Approach (robust + interpretable, per the range-GARCH / India-VIX literature):

1. Estimate today's expected daily move (fraction) as a weighted blend of forward and
   range-based volatility signals: India-VIX-implied move, EGARCH forecast, Yang-Zhang,
   and ATR%. Blend weights live in config (``tunable.range.blend_weights``) so Agent 2
   can retune them.

2. Calibrate two scalars k_up, k_dn on training data so the predicted band
   ``[open*(1 - k_dn*move), open*(1 + k_up*move)]`` covers the actual high/low excursions
   at the target probability (``tunable.range.band_quantile``, e.g. 0.80). Asymmetry
   (k_up != k_dn) captures the index's tendency to fall faster than it rises.

k_up/k_dn are stored in the bundle; the blend reads live config so retuning needs no refit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_settings

# Volatility signals (in fraction units) blended into the expected move, in the
# same order as config ``tunable.range.blend_weights``.
_BLEND_COLS = ["vix_daily_move", "garch_sigma", "prev_sigma_yz", "prev_atr_pct"]


def expected_move(frame: pd.DataFrame) -> pd.Series:
    """Weighted blend of vol signals -> expected daily move (fraction)."""
    weights = np.asarray(load_settings()["tunable"]["range"]["blend_weights"], dtype=float)
    weights = weights / weights.sum()
    cols = frame[_BLEND_COLS].to_numpy()
    return pd.Series(cols @ weights, index=frame.index, name="expected_move")


class RangeModel:
    def __init__(self) -> None:
        self.k_up: float = 1.0
        self.k_dn: float = 1.0

    def fit(self, frame: pd.DataFrame) -> "RangeModel":
        """Calibrate k_up/k_dn so the band hits the target coverage on training data."""
        q = float(load_settings()["tunable"]["range"]["band_quantile"])
        move = expected_move(frame)
        # Ratio of actual excursion to expected move; the q-quantile of this ratio is
        # the multiplier that yields q coverage.
        up_ratio = (frame["up_exc"] / move).replace([np.inf, -np.inf], np.nan).dropna()
        dn_ratio = (frame["dn_exc"] / move).replace([np.inf, -np.inf], np.nan).dropna()
        self.k_up = float(np.quantile(up_ratio, q))
        self.k_dn = float(np.quantile(dn_ratio, q))
        return self

    def predict_band(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return predicted high/low (price) and expected_move for each row."""
        move = expected_move(frame)
        open_ = frame["open"]
        high = open_ * (1.0 + self.k_up * move)
        low = open_ * (1.0 - self.k_dn * move)
        return pd.DataFrame(
            {"pred_high": high, "pred_low": low, "expected_move": move}, index=frame.index
        )
