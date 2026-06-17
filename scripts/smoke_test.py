"""Offline smoke test — runs the FULL Agent 1 pipeline against mock Dhan data (no token).

This exists because live bugs (a missing column, a bad strike, a schema mismatch) kept
surfacing only on the user's morning run — the local Dhan token is expired by the time we
develop, so the live path went untested. This test mocks the Dhan client and exercises the
real ``run_agent1.run()`` end-to-end, asserting the report builds with all sections and
sane trade strikes. Run before every push:

    python -m scripts.smoke_test     # exit 0 = pass
"""
from __future__ import annotations

import datetime as dt
import sys

import numpy as np
import pandas as pd

from src.config import load_settings


class FakeDhan:
    """Minimal stand-in for DhanClient returning deterministic synthetic data."""

    def __init__(self) -> None:
        self.cfg = load_settings()["market"]["dhan"]
        self.client_id = "TEST"

    def daily_history(self, frm, to, security_id=None, segment=None, instrument="INDEX"):
        n = 820
        idx = pd.bdate_range(end=dt.date.today(), periods=n)
        rng = np.random.default_rng(1)
        if security_id == self.cfg["vix_security_id"]:
            v = 13 + np.abs(rng.normal(0, 1.5, n))
            return pd.DataFrame({"open": v, "high": v + 0.5, "low": v - 0.5,
                                 "close": v, "volume": 0}, index=idx)
        close = 24000 + np.cumsum(rng.normal(0, 80, n))
        high = close + np.abs(rng.normal(0, 60, n))
        low = close - np.abs(rng.normal(0, 60, n))
        open_ = close + rng.normal(0, 40, n)
        return pd.DataFrame({"open": open_, "high": high, "low": low,
                             "close": close, "volume": 1e6}, index=idx)

    def intraday(self, frm, to, interval="5", **kw):
        days = pd.bdate_range(end=dt.date.today(), periods=5)
        rng = np.random.default_rng(2)
        rows, idxs = [], []
        for d in days:
            for h, m in [(9, 15), (9, 20), (9, 25)]:
                base = 24000 + rng.normal(0, 50)
                rows.append([base, base + 12, base - 12, base + rng.normal(0, 6), 1e5])
                idxs.append(pd.Timestamp(d) + pd.Timedelta(hours=h, minutes=m))
        return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"],
                            index=pd.DatetimeIndex(idxs))

    def ltp(self, securities=None):
        return {"data": {"IDX_I": {
            str(self.cfg["nifty_security_id"]): {"last_price": 24044.0},
            str(self.cfg["vix_security_id"]): {"last_price": 13.0}}}}

    def expiry_list(self):
        return [(dt.date.today() + dt.timedelta(days=3)).isoformat()]

    def option_chain(self, expiry):
        oc = {}
        for k in range(23000, 25001, 50):
            dist = abs(k - 24050)
            prem = max(1.0, 200 - dist * 0.12)
            oi = max(1, 6000 - dist)
            oc[f"{k}.000000"] = {
                "ce": {"last_price": prem, "oi": oi, "implied_volatility": 12.0},
                "pe": {"last_price": prem, "oi": oi, "implied_volatility": 12.5}}
        return {"data": {"last_price": 24044.0, "oc": oc}}


def main() -> None:
    import scripts.run_agent1 as a1

    # Swap in the fake Dhan + neutralize external side-effects.
    a1.DhanClient = lambda: FakeDhan()
    a1.update_opening_range_cache = lambda *a, **k: None
    a1.fetch_global_cues = lambda: type("G", (), {"summary": lambda self: "mock cues"})()

    report = a1.run(dry_run=True)

    required = ["EXPECTED DAY RANGE", "CLOSE vs OPEN DIRECTION",
                "PROBABILITY MODEL", "TRADER-SPECIFIC PLANS", "CONFIDENCE"]
    for section in required:
        assert section in report, f"FAIL: report missing section '{section}'"

    # Sanity: every option strike mentioned should be within 5% of the ~24044 open
    # (guards against the far-OI-wall strike-selection bug).
    import re
    for m in re.finditer(r"(?:sell|Buy|SELL|BUY)\s+(\d{5})\s*(?:CE|PE)", report):
        strike = int(m.group(1))
        assert 22800 <= strike <= 25300, f"FAIL: insane strike {strike} (far from open)"

    print("✅ SMOKE TEST PASSED — pipeline runs, all sections present, strikes sane.")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
