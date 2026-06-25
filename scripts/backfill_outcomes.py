"""Backfill outcomes for past predictions using yfinance actuals (no Dhan token needed).

Agent 2 normally scores each day live via Dhan, but if its evening runs didn't fire, the
predictions sit unscored. This re-scores every unscored prediction against yfinance daily
OHLC and writes them to outcomes.jsonl, then prints the rolling scorecard. Safe to re-run
(skips already-scored dates).

    python -m scripts.backfill_outcomes
"""
from __future__ import annotations

import sys

import pandas as pd

from src.data.historical import load_training_frame
from src.scoring import scorecard
from src.scoring.review import score_prediction
from src.storage.logs import OUTCOMES, PREDICTIONS, append_jsonl, read_jsonl


def main() -> None:
    daily = load_training_frame(refresh=True)  # fresh yfinance daily OHLC
    preds = read_jsonl(PREDICTIONS)
    if preds.empty:
        print("no predictions to score")
        return
    preds = preds.drop_duplicates(subset="date", keep="last").sort_values("date")

    outs = read_jsonl(OUTCOMES)
    done = set(outs["date"]) if not outs.empty else set()

    scored = 0
    for _, p in preds.iterrows():
        if p["date"] in done:
            continue
        d = pd.Timestamp(p["date"]).normalize()
        if d not in daily.index:
            print(f"  {p['date']}: no actual in yfinance yet — skipped")
            continue
        row = daily.loc[d]
        actual = {"open": float(row["open"]), "high": float(row["high"]),
                  "low": float(row["low"]), "close": float(row["close"])}
        rec = score_prediction(p.to_dict(), actual)
        append_jsonl(OUTCOMES, rec)
        scored += 1
        tr = rec["trade_r"]
        tr_s = f"{tr:+.2f}R" if isinstance(tr, float) and tr == tr else "n/a"
        ok = "✅" if rec["dir_pred"] == rec["dir_actual"] else "❌"
        contained = rec["act_high"] <= rec["pred_high"] and rec["act_low"] >= rec["pred_low"]
        print(f"  {p['date']}: dir {ok} (pred {rec['dir_pred']}/act {rec['dir_actual']}), "
              f"range {'in' if contained else 'OUT'}, futures {tr_s}")

    print(f"\nscored {scored} new outcome(s)")
    allout = read_jsonl(OUTCOMES)
    if not allout.empty:
        card = scorecard.compute(allout)
        print(f"\n=== ROLLING SCORECARD (n={card.n}) ===")
        print(f"  composite              : {card.composite}  (satisfactory={card.satisfactory})")
        print(f"  direction              : {card.direction}%")
        print(f"  range_hit              : {card.range_hit}   (contained {card.detail['range_contained']}%)")
        print(f"  calibration            : {card.calibration}   (brier {card.detail['brier']})")
        print(f"  trade_pnl              : {card.trade_pnl}   (mean {card.detail['mean_trade_r']}R)")
        print(f"  confidence_reliability : {card.confidence_reliability}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
