"""Actual-trade ledger — what the user really earned, from their reported fills.

Separate from the model scorecard (whose trade_pnl is a futures-only proxy). Records live
in data/logs/trades.jsonl: one row per (date, persona) with an outcome
(win/loss/no_entry/skip) and optional pnl_inr. This summarizes real win-rate and rupee P&L
per persona and overall — so range-day seller wins (invisible to the futures proxy) are
captured.

    python -m scripts.trade_ledger                     # print the ledger
    python -m scripts.trade_ledger --add DATE PERSONA OUTCOME [PNL_INR] ["note"]
"""
from __future__ import annotations

import sys

from src.storage.logs import TRADES, log_trade, read_jsonl

_TAKEN = {"win", "loss"}   # outcomes that count as an actual trade taken


def summary() -> None:
    df = read_jsonl(TRADES)
    if df.empty:
        print("no trades logged yet")
        return

    taken = df[df["outcome"].isin(_TAKEN)]
    wins = int((taken["outcome"] == "win").sum())
    losses = int((taken["outcome"] == "loss").sum())
    n = wins + losses
    win_rate = (wins / n * 100) if n else 0.0
    pnl = df["pnl_inr"].dropna().sum() if "pnl_inr" in df else 0.0

    print(f"=== ACTUAL TRADE LEDGER ({df['date'].nunique()} sessions) ===")
    print(f"  trades taken : {n}  (wins {wins} / losses {losses})  win-rate {win_rate:.0f}%")
    print(f"  net P&L (known fills only): Rs {pnl:,.2f}")
    print("  by persona:")
    for persona, g in df.groupby("persona"):
        gt = g[g["outcome"].isin(_TAKEN)]
        w = int((gt["outcome"] == "win").sum())
        tot = len(gt)
        rupees = g["pnl_inr"].dropna().sum() if "pnl_inr" in g else 0.0
        extra = f", Rs {rupees:,.0f}" if rupees else ""
        skipped = int((g["outcome"] == "skip").sum()) + int((g["outcome"] == "no_entry").sum())
        print(f"    {persona:26s}: {w}/{tot} win"
              f"{f' ({w/tot*100:.0f}%)' if tot else ''}{extra}"
              f"{f'  [{skipped} skip/no-entry]' if skipped else ''}")


def _add(args: list[str]) -> None:
    date, persona, outcome = args[0], args[1], args[2]
    pnl = float(args[3]) if len(args) > 3 and args[3] not in ("", "-", "none") else None
    note = args[4] if len(args) > 4 else ""
    log_trade({"date": date, "persona": persona, "outcome": outcome, "pnl_inr": pnl, "note": note})
    print(f"logged: {date} {persona} {outcome} {pnl if pnl is not None else ''}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) > 1 and sys.argv[1] == "--add":
        _add(sys.argv[2:])
    else:
        summary()
