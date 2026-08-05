"""Replay a logged trade plan against intraday data to compute REAL P&L.

This closes the loop that previously required the user to hand-report every fill. Given the
machine-readable ``sim`` spec Agent 1 logs with each plan (strikes, security ids, entry
premiums, stop/target rules), we fetch 5-minute candles for those exact instruments from
Dhan and walk the session bar-by-bar applying the plan's own rules:

  * option spreads — combined premium vs the % stop / % target, plus the absolute ₹ hard
    stop; otherwise exit at the close.
  * futures — hard points stop, then a step-trailing stop once in profit; otherwise exit at
    the close.

P&L is in rupees using the configured lot size. The result is what the plan *as written*
would have earned; the user's own deviations (late entry, manual exit) are not modelled —
those remain the only thing worth reporting by hand.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from src.config import load_settings

_FNO_SEG = "NSE_FNO"


def _intraday(client, security_id: int, date_str: str, instrument: str) -> pd.DataFrame | None:
    """5-min candles for one instrument on one session (empty/None if unavailable)."""
    try:
        d = dt.date.fromisoformat(date_str)
        df = client.intraday(date_str, (d + dt.timedelta(days=1)).isoformat(), interval="5",
                             security_id=security_id, segment=_FNO_SEG, instrument=instrument)
        if df is None or df.empty:
            return None
        same_day = df[df.index.normalize() == pd.Timestamp(d)]
        return same_day if not same_day.empty else None
    except Exception:
        return None


def _simulate_option_spread(client, spec: dict, date_str: str, lot: int) -> dict | None:
    """Walk the session applying % stop / % target / hard ₹ stop to the combined premium."""
    legs = spec.get("legs") or []
    frames, entries = [], []
    for leg in legs:
        sid = leg.get("security_id")
        if not sid:
            return None                      # can't replay without the instrument id
        df = _intraday(client, int(sid), date_str, "OPTIDX")
        if df is None:
            return None
        frames.append(df)
        entries.append(float(leg.get("entry") or df["open"].iloc[0]))

    entry_total = sum(entries)
    if entry_total <= 0:
        return None

    # Short spread: we profit when premium falls. (All current spreads are SELL legs.)
    sl_level = entry_total * (1 + float(spec["sl_premium_pct"]))
    tgt_level = entry_total * (1 - float(spec["target_premium_pct"]))
    hard_stop = float(spec.get("hard_stop_inr") or 0)

    idx = frames[0].index
    for ts in idx:
        # Worst case within the bar (premium high) is checked before the favourable case,
        # so a bar that touches both is scored as the stop — the conservative reading.
        try:
            worst = sum(float(f.loc[ts, "high"]) for f in frames)
            best = sum(float(f.loc[ts, "low"]) for f in frames)
        except KeyError:
            continue
        loss_at_worst = (worst - entry_total) * lot
        if worst >= sl_level or (hard_stop and loss_at_worst >= hard_stop):
            exit_px = min(worst, entry_total + hard_stop / lot) if hard_stop else worst
            reason = "hard_stop" if (hard_stop and loss_at_worst >= hard_stop) else "sl"
            return {"pnl_inr": round((entry_total - exit_px) * lot, 2), "exit": ts.strftime("%H:%M"),
                    "reason": reason, "entry_premium": round(entry_total, 2)}
        if best <= tgt_level:
            return {"pnl_inr": round((entry_total - tgt_level) * lot, 2), "exit": ts.strftime("%H:%M"),
                    "reason": "target", "entry_premium": round(entry_total, 2)}

    close_total = sum(float(f["close"].iloc[-1]) for f in frames)
    return {"pnl_inr": round((entry_total - close_total) * lot, 2), "exit": "close",
            "reason": "eod", "entry_premium": round(entry_total, 2)}


def _simulate_futures(spec: dict, intraday_index: pd.DataFrame, lot: int) -> dict | None:
    """Hard points stop, then step-trailing once in profit; else exit at the close.

    Uses the index intraday series as a proxy for the future (they track within a few
    points), so no futures instrument id is needed.
    """
    if intraday_index is None or intraday_index.empty:
        return None
    long = spec.get("direction") == "LONG"
    entry = float(spec["entry"])
    sl_pts = float(spec["sl_points"])
    step = float(spec.get("trail_step") or sl_pts)

    stop = entry - sl_pts if long else entry + sl_pts
    best = entry
    for ts, bar in intraday_index.iterrows():
        hi, lo = float(bar["high"]), float(bar["low"])
        # Stop first (conservative when a bar spans both directions).
        if (long and lo <= stop) or (not long and hi >= stop):
            return {"pnl_inr": round(((stop - entry) if long else (entry - stop)) * lot, 2),
                    "exit": ts.strftime("%H:%M"), "reason": "stop/trail"}
        # Ratchet the trailing stop by whole steps of favourable movement.
        if long and hi > best:
            best = hi
            stop = max(stop, entry + (int((best - entry) / step) - 1) * step) if best - entry >= step else stop
        elif not long and lo < best:
            best = lo
            stop = min(stop, entry - (int((entry - best) / step) - 1) * step) if entry - best >= step else stop

    close = float(intraday_index["close"].iloc[-1])
    return {"pnl_inr": round(((close - entry) if long else (entry - close)) * lot, 2),
            "exit": "close", "reason": "eod"}


def simulate_plans(client, prediction: dict) -> list[dict]:
    """Simulate every taken plan in a logged prediction. Returns trade-ledger rows."""
    plans = prediction.get("plans") or []
    date_str = prediction["date"]
    lot = int(load_settings()["personas"]["lot_size"])
    rows: list[dict] = []

    index_intraday = None
    for plan in plans:
        if not plan.get("take_trade"):
            rows.append({"date": date_str, "persona": plan.get("persona_key", "unknown"),
                         "outcome": "skip", "pnl_inr": None, "source": "simulated",
                         "note": "plan not taken"})
            continue
        spec = plan.get("sim") or {}
        result = None
        if spec.get("kind") == "option_spread":
            result = _simulate_option_spread(client, spec, date_str, lot)
        elif spec.get("kind") == "futures":
            if index_intraday is None:
                cfg = load_settings()["market"]["dhan"]
                index_intraday = _intraday(client, cfg["nifty_security_id"], date_str, "INDEX")
                if index_intraday is None:   # index segment, not FNO
                    try:
                        d = dt.date.fromisoformat(date_str)
                        index_intraday = client.intraday(
                            date_str, (d + dt.timedelta(days=1)).isoformat(), interval="5")
                        index_intraday = index_intraday[
                            index_intraday.index.normalize() == pd.Timestamp(d)]
                    except Exception:
                        index_intraday = None
            result = _simulate_futures(spec, index_intraday, lot)

        if result is None:
            rows.append({"date": date_str, "persona": plan.get("persona_key", "unknown"),
                         "outcome": "unknown", "pnl_inr": None, "source": "simulated",
                         "note": "intraday data unavailable"})
            continue
        pnl = result["pnl_inr"]
        # Exactly breakeven (a trail-to-entry stop-out) is a scratch, not a loss — counting
        # it as one would understate the win-rate.
        outcome = "win" if pnl > 0 else ("loss" if pnl < 0 else "scratch")
        rows.append({
            "date": date_str, "persona": plan.get("persona_key", "unknown"),
            "outcome": outcome, "pnl_inr": pnl, "source": "simulated",
            "note": f"auto: exit {result['exit']} ({result['reason']})",
        })
    return rows
