"""Historical daily data for model training.

Free source (yfinance): Nifty 50 (``^NSEI``) OHLC and India VIX (``^INDIAVIX``) close.
Data is cached to ``data/historical`` so repeated runs / backtests do not re-hit the
network. Live/intraday data comes from Dhan (see ``dhan_client.py``); this module is
strictly for the training corpus.

The canonical training frame returned by :func:`load_training_frame` has one row per
trading day with columns:

    open, high, low, close, volume      # Nifty 50 daily OHLCV
    prev_close                          # previous session close
    vix, prev_vix                       # India VIX close (and previous)

All downstream features are derived from this frame.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import load_settings, project_path

_NIFTY_COLS = ["open", "high", "low", "close", "volume"]


def _cache_path(name: str) -> Path:
    settings = load_settings()
    cache_dir = project_path(settings["data"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{name}.csv"


def _download_yf(ticker: str, years: int) -> pd.DataFrame:
    """Download daily OHLCV for a ticker via yfinance. Returns a clean lowercase frame."""
    import yfinance as yf

    period = f"{years}y"
    raw = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker!r}")

    # yfinance may return a MultiIndex (column, ticker) when given a single ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.rename(columns=str.lower)
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    raw.index.name = "date"
    return raw


def refresh_cache(years: int | None = None) -> None:
    """Force a fresh download of Nifty + India VIX and overwrite the cache."""
    settings = load_settings()
    years = years or settings["data"]["historical_years"]
    sym = settings["market"]["symbol_yf"]
    vix = settings["market"]["vix_yf"]

    nifty = _download_yf(sym, years)[_NIFTY_COLS]
    nifty.to_csv(_cache_path("nifty"))

    vix_df = _download_yf(vix, years)[["close"]].rename(columns={"close": "vix"})
    vix_df.to_csv(_cache_path("vix"))


def _read_cache(name: str) -> pd.DataFrame | None:
    path = _cache_path(name)
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


def load_training_frame(refresh: bool = False) -> pd.DataFrame:
    """Return the merged daily training frame, downloading/caching if needed.

    Parameters
    ----------
    refresh:
        If True, re-download from yfinance before loading.
    """
    if refresh or _read_cache("nifty") is None or _read_cache("vix") is None:
        refresh_cache()

    nifty = _read_cache("nifty")
    vix = _read_cache("vix")
    assert nifty is not None and vix is not None  # populated by refresh_cache above

    df = nifty.join(vix, how="left")
    # India VIX has occasional gaps vs the index calendar; forward-fill is the
    # standard treatment (VIX is persistent day-to-day).
    df["vix"] = df["vix"].ffill()

    df["prev_close"] = df["close"].shift(1)
    df["prev_vix"] = df["vix"].shift(1)

    # Drop the first row (no previous session) and any rows still missing core data.
    df = df.dropna(subset=["open", "high", "low", "close", "prev_close"])
    return df


if __name__ == "__main__":  # quick manual check
    frame = load_training_frame()
    print(f"rows: {len(frame)}  range: {frame.index.min().date()} -> {frame.index.max().date()}")
    print(frame.tail(3).to_string())
