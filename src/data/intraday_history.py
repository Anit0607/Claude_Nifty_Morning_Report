"""Historical intraday backfill from Dhan -> per-day opening-range features.

Dhan serves 5-min intraday history several years back, so we can build a proper
opening-range (OR) training/backtest set — the feature the research says actually drives
intraday direction, and which free daily data cannot provide.

At Agent 1's 09:20 decision time the just-completed first 5-min candle (09:15-09:20) is
known. We extract, per session, features from that first candle:

    or_ret        first-candle return (close-open)/open      -> early momentum / gap-and-go
    or_range      first-candle (high-low)/open               -> early realized range
    or_close_pos  (close-low)/(high-low) in [0,1]            -> where it closed in the bar

The cache (``opening_range.csv``) is committed to the repo and kept current incrementally
via :func:`update_opening_range_cache`, so training/backtests always see recent sessions
without re-downloading 6 years each run.
"""
from __future__ import annotations

import datetime as dt
import time

import pandas as pd

from src.config import project_path
from src.data.dhan_client import DhanClient

OR_CACHE = project_path("data", "historical", "opening_range.csv")
OR_FEATURE_COLS = ["or_ret", "or_range", "or_close_pos"]


def _fetch_first_candles(client: DhanClient, start: str, end: str,
                         chunk_days: int = 30, sleep: float = 0.25) -> pd.DataFrame:
    """Fetch 5-min history in chunks; keep only the first (09:15) candle per day."""
    cur = dt.date.fromisoformat(start)
    end_d = dt.date.fromisoformat(end)
    collected = []
    while cur <= end_d:
        nxt = min(cur + dt.timedelta(days=chunk_days), end_d)
        try:
            df = client.intraday(cur.isoformat(), nxt.isoformat(), interval="5")
            if len(df):
                opens = df[df.index.time == dt.time(9, 15)]
                if len(opens):
                    collected.append(opens)
        except Exception as exc:
            print(f"  [skip {cur}..{nxt}] {str(exc)[:100]}")
        cur = nxt + dt.timedelta(days=1)
        time.sleep(sleep)
    if not collected:
        return pd.DataFrame()
    first = pd.concat(collected).sort_index()
    first = first[~first.index.duplicated()]
    first.index = first.index.normalize()
    first.index.name = "date"
    return first


def _or_features(first: pd.DataFrame) -> pd.DataFrame:
    """Per-day opening-range features from first-candle OHLC."""
    rng = (first["high"] - first["low"]).replace(0, pd.NA)
    tbl = pd.DataFrame(index=first.index)
    tbl["or_ret"] = (first["close"] - first["open"]) / first["open"]
    tbl["or_range"] = (first["high"] - first["low"]) / first["open"]
    tbl["or_close_pos"] = ((first["close"] - first["low"]) / rng).astype(float).fillna(0.5)
    return tbl


def _save(tbl: pd.DataFrame) -> None:
    OR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tbl.sort_index().to_csv(OR_CACHE)


def build_opening_range_table(refresh: bool = False, start: str = "2020-01-01") -> pd.DataFrame:
    """Return the date-indexed OR feature table, building+caching from Dhan if needed."""
    if OR_CACHE.exists() and not refresh:
        return pd.read_csv(OR_CACHE, index_col=0, parse_dates=True)
    first = _fetch_first_candles(DhanClient(), start, dt.date.today().isoformat())
    if first.empty:
        raise RuntimeError("No intraday data fetched from Dhan.")
    tbl = _or_features(first)
    _save(tbl)
    return tbl


def update_opening_range_cache(client: DhanClient | None = None) -> pd.DataFrame:
    """Incrementally append any missing recent sessions to the cache. Cheap + idempotent.

    Fetches only from the day after the last cached date through today. Safe to call at the
    start of each agent run so the OR cache (and thus training data) stays current.
    """
    if not OR_CACHE.exists():
        return build_opening_range_table()

    tbl = pd.read_csv(OR_CACHE, index_col=0, parse_dates=True)
    last = tbl.index.max().date()
    today = dt.date.today()
    if last >= today:
        return tbl

    client = client or DhanClient()
    first = _fetch_first_candles(client, (last + dt.timedelta(days=1)).isoformat(), today.isoformat())
    if first.empty:
        return tbl

    new_tbl = _or_features(first)
    merged = pd.concat([tbl, new_tbl])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    _save(merged)
    added = len(merged) - len(tbl)
    if added:
        print(f"  [OR cache] appended {added} session(s); now through {merged.index.max().date()}")
    return merged


if __name__ == "__main__":
    table = build_opening_range_table(refresh=True)
    print(f"opening-range table: {len(table)} sessions "
          f"({table.index.min().date()} -> {table.index.max().date()})")
    print(table.describe().T.to_string())
