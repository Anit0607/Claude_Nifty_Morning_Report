"""Dhan v2 REST client — live market data for Agent 1 (read-only; never places orders).

Endpoints used:
  * /charts/historical   — daily OHLC history
  * /charts/intraday     — intraday candles (e.g. the 09:15-09:20 opening-range bar)
  * /marketfeed/ltp      — last traded price (Nifty, India VIX)
  * /optionchain         — option chain (OI/IV) for a given expiry
  * /optionchain/expirylist

Auth: header ``access-token``; the quote/option-chain endpoints additionally need
``client-id``. Token is read from the environment (``.env`` locally / CI secret) and
expires daily during the testing phase.

This module ONLY reads market data. It must never call order/trade endpoints.
"""
from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests

from src.config import get_env, load_settings


class DhanError(RuntimeError):
    pass


class DhanClient:
    def __init__(self) -> None:
        cfg = load_settings()["market"]["dhan"]
        self.base = cfg["base_url"]
        self.cfg = cfg
        self.token = get_env("DHAN_ACCESS_TOKEN", required=True)
        self.client_id = get_env("DHAN_CLIENT_ID", required=True)
        self._session = requests.Session()

    # ---- low-level ----
    def _headers(self, with_client: bool = False) -> dict[str, str]:
        h = {
            "access-token": self.token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if with_client:
            h["client-id"] = self.client_id
        return h

    def _post(self, path: str, body: dict, with_client: bool = False, timeout: int = 20) -> Any:
        url = f"{self.base}{path}"
        resp = self._session.post(url, json=body, headers=self._headers(with_client), timeout=timeout)
        if resp.status_code != 200:
            raise DhanError(f"{path} -> HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        if isinstance(data, dict) and data.get("status") == "failure":
            raise DhanError(f"{path} -> API failure: {data}")
        return data

    # ---- market data ----
    def daily_history(self, from_date: str, to_date: str,
                      security_id: int | None = None, segment: str | None = None,
                      instrument: str = "INDEX") -> pd.DataFrame:
        """Daily OHLC(V) history. Dates 'YYYY-MM-DD'. Returns a date-indexed frame."""
        body = {
            "securityId": str(security_id or self.cfg["nifty_security_id"]),
            "exchangeSegment": segment or self.cfg["index_segment"],
            "instrument": instrument,
            "expiryCode": 0,
            "oi": False,
            "fromDate": from_date,
            "toDate": to_date,
        }
        return _candles_to_frame(self._post("/charts/historical", body))

    def intraday(self, from_date: str, to_date: str, interval: str = "5",
                 security_id: int | None = None, segment: str | None = None,
                 instrument: str = "INDEX") -> pd.DataFrame:
        """Intraday candles. ``interval`` in minutes: '1','5','15','25','60'."""
        body = {
            "securityId": str(security_id or self.cfg["nifty_security_id"]),
            "exchangeSegment": segment or self.cfg["index_segment"],
            "instrument": instrument,
            "interval": str(interval),
            "oi": False,
            "fromDate": from_date,
            "toDate": to_date,
        }
        return _candles_to_frame(self._post("/charts/intraday", body))

    def ltp(self, securities: dict[str, list[int]] | None = None) -> dict:
        """Last traded price. Default: Nifty + India VIX on IDX_I."""
        if securities is None:
            seg = self.cfg["index_segment"]
            securities = {seg: [self.cfg["nifty_security_id"], self.cfg["vix_security_id"]]}
        return self._post("/marketfeed/ltp", securities, with_client=True)

    def expiry_list(self) -> list[str]:
        body = {"UnderlyingScrip": self.cfg["nifty_security_id"],
                "UnderlyingSeg": self.cfg["underlying_seg"]}
        data = self._post("/optionchain/expirylist", body, with_client=True)
        return data.get("data", data) if isinstance(data, dict) else data

    def option_chain(self, expiry: str) -> dict:
        """Option chain for a given expiry 'YYYY-MM-DD'. Rate-limited (~1 / 3s)."""
        body = {"UnderlyingScrip": self.cfg["nifty_security_id"],
                "UnderlyingSeg": self.cfg["underlying_seg"], "Expiry": expiry}
        return self._post("/optionchain", body, with_client=True)


def _candles_to_frame(data: dict) -> pd.DataFrame:
    """Dhan candle response (parallel arrays) -> tidy date/time-indexed OHLCV frame."""
    if not data or "close" not in data:
        raise DhanError(f"Unexpected candle response: {str(data)[:300]}")
    df = pd.DataFrame({
        "open": data.get("open", []),
        "high": data.get("high", []),
        "low": data.get("low", []),
        "close": data.get("close", []),
        "volume": data.get("volume", []),
    })
    ts = data.get("timestamp") or data.get("start_Time") or data.get("startTime")
    if ts is not None and len(ts) == len(df):
        # Dhan timestamps are epoch seconds; localize to IST for readability.
        df.index = (pd.to_datetime(ts, unit="s", utc=True)
                    .tz_convert("Asia/Kolkata").tz_localize(None))
        df.index.name = "datetime"
    return df


if __name__ == "__main__":  # connectivity smoke test (read-only)
    import datetime as dt

    c = DhanClient()
    today = dt.date.today()
    start = today - dt.timedelta(days=14)
    print("Daily history (last ~14d):")
    hist = c.daily_history(start.isoformat(), today.isoformat())
    print(hist.tail(5).to_string())
    print("\nLTP (Nifty + VIX):")
    print(c.ltp())
