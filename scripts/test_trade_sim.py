"""Offline end-to-end test of the trade simulator (no Dhan token needed).

Feeds `simulate_plans` a synthetic prediction (the same shape Agent 1 now logs) plus a mock
client serving intraday candles, and checks the rupee P&L for three scenarios:
  1. strangle that decays -> profit at target
  2. strangle on a trend day -> capped by the hard Rs stop
  3. futures long that runs then fades -> trailing stop locks profit

    python -m scripts.test_trade_sim
"""
from __future__ import annotations

import datetime as dt
import sys

import pandas as pd

from src.scoring.trade_sim import simulate_plans

_TODAY = dt.date.today().isoformat()
_IDX = pd.date_range(f"{_TODAY} 09:20", periods=8, freq="5min")


def _frame(lows, highs):
    return pd.DataFrame({"open": highs, "high": highs, "low": lows, "close": lows}, index=_IDX)


class MockClient:
    """Serves canned intraday candles per security id."""

    def __init__(self, series: dict[int, pd.DataFrame], index: pd.DataFrame):
        self.series, self.index = series, index
        self.cfg = {"nifty_security_id": 13, "index_segment": "IDX_I"}

    def intraday(self, frm, to, interval="5", security_id=None, segment=None, instrument="INDEX"):
        if security_id in self.series:
            return self.series[security_id]
        return self.index


def _plan_strangle(ce_entry, pe_entry):
    return {"persona": "Intraday Option Non-Directional Seller",
            "persona_key": "non_directional_seller", "take_trade": True,
            "sim": {"kind": "option_spread",
                    "legs": [{"side": "SELL", "right": "CE", "strike": 24350,
                              "security_id": 111, "entry": ce_entry},
                             {"side": "SELL", "right": "PE", "strike": 24000,
                              "security_id": 222, "entry": pe_entry}],
                    "sl_premium_pct": 0.55, "target_premium_pct": 0.55,
                    "hard_stop_inr": 3000}}


def main() -> None:
    lot = 75

    # --- 1. Premium decays from 136 -> ~55 : should hit the 55% target ---
    ce = _frame([80, 74, 66, 58, 50, 44, 40, 38], [82, 78, 70, 62, 54, 48, 44, 42])
    pe = _frame([56, 52, 46, 40, 34, 30, 26, 24], [58, 55, 50, 44, 38, 33, 29, 27])
    idx = _frame([24100] * 8, [24200] * 8)
    rows = simulate_plans(MockClient({111: ce, 222: pe}, idx),
                          {"date": _TODAY, "plans": [_plan_strangle(80, 56)]})
    print("1) decaying strangle  ->", rows[0]["outcome"], f"Rs {rows[0]['pnl_inr']}", rows[0]["note"])

    # --- 2. Trend day: put premium explodes -> hard Rs3,000 stop must cap the loss ---
    ce_t = _frame([80, 70, 60, 50, 42, 36, 30, 26], [82, 74, 64, 54, 46, 40, 34, 30])
    pe_t = _frame([56, 70, 95, 130, 170, 210, 250, 290], [58, 76, 104, 142, 184, 226, 268, 310])
    rows = simulate_plans(MockClient({111: ce_t, 222: pe_t}, idx),
                          {"date": _TODAY, "plans": [_plan_strangle(80, 56)]})
    r = rows[0]
    print("2) trend-day strangle ->", r["outcome"], f"Rs {r['pnl_inr']}", r["note"],
          "  [hard stop caps at -3000]" if r["pnl_inr"] and r["pnl_inr"] >= -3000 else "  [!! EXCEEDED CAP]")

    # --- 3. Futures long: +200 then fades -> trailing locks profit ---
    fut_idx = pd.DataFrame(
        {"open": [24000] * 8,
         "high": [24030, 24080, 24140, 24200, 24200, 24200, 24200, 24200],
         "low": [23995, 24020, 24070, 24130, 24050, 24000, 23980, 23980],
         "close": [24020, 24070, 24130, 24190, 24060, 24010, 23990, 23990]}, index=_IDX)
    fut_plan = {"persona": "Intraday Futures Trader", "persona_key": "futures", "take_trade": True,
                "sim": {"kind": "futures", "direction": "LONG", "entry": 24000.0,
                        "sl_points": 50.0, "target_points": 50.0, "trail_step": 50.0}}
    rows = simulate_plans(MockClient({}, fut_idx), {"date": _TODAY, "plans": [fut_plan]})
    print("3) futures long+trail ->", rows[0]["outcome"], f"Rs {rows[0]['pnl_inr']}", rows[0]["note"])

    # --- 4. A skipped plan should log as 'skip', not a trade ---
    skipped = {"persona": "Intraday Futures Trader", "persona_key": "futures",
               "take_trade": False, "sim": {}}
    rows = simulate_plans(MockClient({}, idx), {"date": _TODAY, "plans": [skipped]})
    print("4) skipped plan       ->", rows[0]["outcome"], "(no P&L)" if rows[0]["pnl_inr"] is None else "!!")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
