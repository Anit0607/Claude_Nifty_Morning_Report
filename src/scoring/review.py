"""Score a logged prediction against the actual session outcome.

Turns one ``predictions.jsonl`` row + the day's real OHLC into a record with the columns
the scorecard consumes, plus a futures-bracket P&L in R (consistent with the backtest's
``trade_r`` proxy). Agent 2 appends these to ``outcomes.jsonl`` and scores rolling windows.
"""
from __future__ import annotations

import pandas as pd

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


def score_pending_from_frame(daily: pd.DataFrame, before, write: bool = True):
    """Score any unscored predictions whose session is complete, using a daily OHLC frame.

    Lets Agent 1 keep the scorecard current from its own (reliable) morning run, without
    depending on Agent 2: each morning it scores yesterday (and any earlier unscored day)
    from the Dhan daily frame it already fetched. Only sessions strictly before ``before``
    (today's target date) are scored — never the in-progress day. With ``write=False`` it
    scores nothing and just returns the current standings (used on dry-runs).

    Returns (Scorecard | None, latest_outcome_dict | None).
    """
    from src.scoring import scorecard
    from src.storage.logs import OUTCOMES, PREDICTIONS, append_jsonl, read_jsonl

    preds = read_jsonl(PREDICTIONS)
    if not preds.empty and write:
        preds = preds.drop_duplicates(subset="date", keep="last")
        done = set(read_jsonl(OUTCOMES)["date"]) if not read_jsonl(OUTCOMES).empty else set()
        before_ts = pd.Timestamp(before).normalize()
        for _, p in preds.sort_values("date").iterrows():
            if p["date"] in done:
                continue
            d = pd.Timestamp(p["date"]).normalize()
            if d >= before_ts or d not in daily.index:
                continue  # in-progress today, or session not in the frame
            row = daily.loc[d]
            actual = {"open": float(row["open"]), "high": float(row["high"]),
                      "low": float(row["low"]), "close": float(row["close"])}
            append_jsonl(OUTCOMES, score_prediction(p.to_dict(), actual))

    outs = read_jsonl(OUTCOMES)
    if outs.empty:
        return None, None
    return scorecard.compute(outs), outs.iloc[-1].to_dict()
