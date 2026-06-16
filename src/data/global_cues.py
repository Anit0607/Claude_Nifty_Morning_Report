"""Overnight global cues — best-effort context for the report.

These are informational/narrative inputs (US close, Asian open) that set the backdrop for
the Indian open. They are fetched via yfinance and are strictly best-effort: any failure
returns None for that field rather than breaking Agent 1. Their directional content is
already largely embedded in the opening gap, so they inform the *narrative*, not the core
model.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GlobalCues:
    sp500_change_pct: float | None = None
    nasdaq_change_pct: float | None = None
    nikkei_change_pct: float | None = None
    hangseng_change_pct: float | None = None

    def summary(self) -> str:
        def fmt(name: str, v: float | None) -> str | None:
            return f"{name} {v:+.2f}%" if v is not None else None
        parts = [fmt("S&P", self.sp500_change_pct), fmt("Nasdaq", self.nasdaq_change_pct),
                 fmt("Nikkei", self.nikkei_change_pct), fmt("HangSeng", self.hangseng_change_pct)]
        live = [p for p in parts if p]
        return " | ".join(live) if live else "global cues unavailable"


def _last_change_pct(ticker: str, attempts: int = 3) -> float | None:
    """Last daily % change for a ticker. yfinance is throttled on cloud IPs, so retry a
    few times; returns None if still unavailable (these cues are best-effort/cosmetic)."""
    import time

    import yfinance as yf
    for i in range(attempts):
        try:
            hist = yf.download(ticker, period="5d", interval="1d",
                               auto_adjust=False, progress=False)
            if hist is not None and len(hist) >= 2:
                closes = hist["Close"].squeeze()
                return round(float((closes.iloc[-1] / closes.iloc[-2] - 1.0) * 100), 2)
        except Exception:
            pass
        time.sleep(1.5 * (i + 1))
    return None


def fetch_global_cues() -> GlobalCues:
    return GlobalCues(
        sp500_change_pct=_last_change_pct("^GSPC"),
        nasdaq_change_pct=_last_change_pct("^IXIC"),
        nikkei_change_pct=_last_change_pct("^N225"),
        hangseng_change_pct=_last_change_pct("^HSI"),
    )
