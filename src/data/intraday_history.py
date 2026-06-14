"""Historical intraday backfill from Dhan -> per-day opening-range features.

Dhan serves 5-min intraday history several years back, so we can build a proper
opening-range (OR) training/backtest set — the feature the research says actually drives
intraday direction, and which free daily data cannot provide.

At Agent 1's 09:20 decision time the just-completed first 5-min candle (09:15-09:20) is
known. We extract, per session, features from that first candle:

    or_ret        first-candle return (close-open)/open      -> early momentum / gap-and-go
    or_range      first-candle (high-low)/open               -> early realized range
    or_close_pos  (close-low)/(high-low) in [0,1]            -> where it closed in the bar
    or_vol_z      first-candle volume vs 20-day average      -> participation/conviction

These are merged onto the daily frame by date and used as extra model features.
"""
from __future__ import annotations

import datetime as dt
import time

import numpy as np
import pandas as pd

from src.config import project_path
from src.data.dhan_client import DhanClient

OR_CACHE = project_path("data", "historical", "opening_range.csv")
# or_vol_z is excluded from the model feature set: the index reports intraday volume
# only for recent sessions, so it is mostly missing over the historical window. It is
# still written to the cache for live use, where volume is available.
OR_FEATURE_COLS = ["or_ret", "or_range", "or_close_pos"]


def fetch_opening_range(start: str = "2020-01-01", chunk_days: int = 30,
                        sleep: float = 0.25) -> pd.DataFrame:
    """Fetch 5-min history in chunks and keep only the first (09:15) candle per day."""
    client = DhanClient()
    today = dt.date.today()
    cur = dt.date.fromisoformat(start)
    first_candles = []

    while cur <= today:
        nxt = min(cur + dt.timedelta(days=chunk_days), today)
        try:
            df = client.intraday(cur.isoformat(), nxt.isoformat(), interval="5")
            if len(df):
                opens = df[df.index.time == dt.time(9, 15)]
                if len(opens):
                    first_candles.append(opens)
        except Exception as exc:  # keep going; log the gap
            print(f"  [skip {cur}..{nxt}] {str(exc)[:100]}")
        cur = nxt + dt.timedelta(days=1)
        time.sleep(sleep)

    if not first_candles:
        raise RuntimeError("No intraday data fetched from Dhan.")

    first = pd.concat(first_candles).sort_index()
    first = first[~first.index.duplicated()]
    first.index = first.index.normalize()
    first.index.name = "date"
    return first


def build_opening_range_table(refresh: bool = False) -> pd.DataFrame:
    """Return a date-indexed OR feature table, building+caching from Dhan if needed."""
    if OR_CACHE.exists() and not refresh:
        return pd.read_csv(OR_CACHE, index_col=0, parse_dates=True)

    first = fetch_opening_range()
    rng = (first["high"] - first["low"]).replace(0, np.nan)

    tbl = pd.DataFrame(index=first.index)
    tbl["or_ret"] = (first["close"] - first["open"]) / first["open"]
    tbl["or_range"] = (first["high"] - first["low"]) / first["open"]
    tbl["or_close_pos"] = ((first["close"] - first["low"]) / rng).astype(float).fillna(0.5)
    vol = first["volume"].astype(float)
    tbl["or_vol_z"] = (vol / vol.rolling(20, min_periods=5).mean()).clip(upper=5.0)

    OR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(OR_CACHE)
    return tbl


if __name__ == "__main__":
    table = build_opening_range_table(refresh=True)
    print(f"opening-range table: {len(table)} sessions "
          f"({table.index.min().date()} -> {table.index.max().date()})")
    print(table.describe().T.to_string())
