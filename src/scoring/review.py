"""Score a logged prediction against the actual session outcome.

Turns one ``predictions.jsonl`` row + the day's real OHLC into a record with the columns
the scorecard consumes, plus a futures-bracket P&L in R (consistent with the backtest's
``trade_r`` proxy). Agent 2 appends these to ``outcomes.jsonl`` and scores rolling windows.
"""
from __future__ import annotations

from src.config import load_settings


def _futures_r(dir_pred: int, open_: float, high: float, low: float, close: float,
               atr_points: float, settings: dict) -> float:
    """Same intraday-futures proxy as the backtest harness."""
    fp = settings["personas"]["futures_trader"]
    sl = max(1e-9, fp["sl_atr_mult"] * atr_points)
    tgt = fp["target_atr_mult"] * atr_points
    if dir_pred == 1:  # long
        if low <= open_ - sl:
            return -1.0
        if high >= open_ + tgt:
            return tgt / sl
        return (close - open_) / sl
    else:              # short
        if high >= open_ + sl:
            return -1.0
        if low <= open_ - tgt:
            return tgt / sl
        return (open_ - close) / sl


def score_prediction(pred: dict, actual: dict) -> dict:
    """Build a scored record. ``actual`` has open/high/low/close for the session."""
    settings = load_settings()
    sideways_pct = settings["scoring"]["sideways_pct"]

    open_ = float(pred["open"])
    high, low, close = float(actual["high"]), float(actual["low"]), float(actual["close"])
    ret_co = (close - open_) / open_
    reg_actual = "Up" if ret_co > sideways_pct else ("Down" if ret_co < -sideways_pct else "Sideways")

    atr_points = pred.get("atr_points")
    trade_r = (_futures_r(int(pred["dir_pred"]), open_, high, low, close, float(atr_points), settings)
               if atr_points else float("nan"))

    return {
        "date": pred["date"],
        "model_version": pred.get("model_version"),
        "dir_pred": int(pred["dir_pred"]),
        "dir_actual": int(close > open_),
        "pred_high": float(pred["pred_high"]),
        "pred_low": float(pred["pred_low"]),
        "act_high": high,
        "act_low": low,
        "p_down": float(pred["p_down"]),
        "p_sideways": float(pred["p_sideways"]),
        "p_up": float(pred["p_up_reg"]),          # scorecard's regime up-probability
        "reg_actual": reg_actual,
        "conf_overall": float(pred["conf_overall"]),
        "trade_r": trade_r,
        "ret_co": ret_co,
    }
