"""Agent 2 — Reviewer / Auto-Healer (runs ~20:00 IST).

Pipeline:
  1. fetch actual session outcomes (Dhan) for any unscored predictions
  2. score them -> outcomes.jsonl
  3. judge satisfactory over the rolling window
  4. diagnose the weak component(s)
  5. if unsatisfactory (and enough history): run champion/challenger self-improvement
  6. send an evening review to Telegram + log a metrics snapshot

Run:
  python -m scripts.run_agent2              # normal nightly review
  python -m scripts.run_agent2 --dry-run    # don't send to Telegram
  python -m scripts.run_agent2 --force-improve   # run the improvement loop regardless
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.config import load_settings
from src.data.dhan_client import DhanClient
from src.delivery import telegram
from src.improve import champion_challenger
from src.improve.diagnose import diagnose
from src.improve.judge import judge
from src.models.registry import get_active_version
from src.scoring.review import score_prediction
from src.storage.logs import (OUTCOMES, PREDICTIONS, log_metrics, log_outcome, read_jsonl)


def _fetch_actual(client: DhanClient, date_str: str) -> dict | None:
    # Dhan rejects a zero-width range, so query a few days and select the target date.
    import datetime as dt
    d = dt.date.fromisoformat(date_str)
    df = client.daily_history(date_str, (d + dt.timedelta(days=4)).isoformat())
    if df.empty:
        return None
    target = pd.Timestamp(d)
    match = df[df.index.normalize() == target]
    if match.empty:
        return None
    r = match.iloc[0]
    return {"open": float(r["open"]), "high": float(r["high"]),
            "low": float(r["low"]), "close": float(r["close"])}


def _score_new_outcomes(client: DhanClient) -> tuple[pd.DataFrame, list[dict]]:
    preds = read_jsonl(PREDICTIONS)
    if not preds.empty:
        preds = preds.drop_duplicates(subset="date", keep="last")
    outs = read_jsonl(OUTCOMES)
    done = set(outs["date"]) if not outs.empty else set()
    new: list[dict] = []
    for _, p in preds.iterrows():
        if p["date"] in done:
            continue
        actual = _fetch_actual(client, p["date"])
        if actual is None:   # outcome not available yet (same-day / holiday)
            continue
        rec = score_prediction(p.to_dict(), actual)
        log_outcome(rec)
        new.append(rec)
    return read_jsonl(OUTCOMES), new


def _build_review(new: list[dict], j, findings: list[str], decision: dict | None) -> str:
    L = ["🌙 NIFTY Quant — Evening Review (Agent 2)", ""]
    if new:
        L.append(f"— SCORED {len(new)} NEW SESSION(S) —")
        for r in new[-3:]:
            ok = "✅" if r["dir_pred"] == r["dir_actual"] else "❌"
            contained = (r["act_high"] <= r["pred_high"] and r["act_low"] >= r["pred_low"])
            tr = r.get("trade_r")
            tr_s = f"{tr:+.2f}R" if isinstance(tr, (int, float)) and tr == tr else "n/a"
            L.append(f"  {r['date']}: dir {ok} (pred {r['dir_pred']}/act {r['dir_actual']}), "
                     f"range {'in' if contained else 'out'}, futures {tr_s}")
        L.append("")
    else:
        L.append("No new outcomes to score this run.\n")

    if not j.enough_data:
        L.append(f"Not enough scored sessions yet to judge (need ≥10). Building history.")
    else:
        c = j.card
        verdict = "✅ SATISFACTORY" if c.satisfactory else "⚠️ BELOW BAR"
        L.append(f"— ROLLING SCORECARD (last {min(j.window, c.n)}) — {verdict}")
        threshold = load_settings()["scoring"]["satisfactory"]["composite_threshold"]
        L.append(f"  Composite {c.composite}/100  (threshold {threshold})")
        L.append(f"  Direction {c.direction}% | Range {c.range_hit} | Calibration {c.calibration} "
                 f"| TradeP&L {c.trade_pnl} | ConfRel {c.confidence_reliability}")
        L.append("")
        if findings:
            L.append("— DIAGNOSIS —")
            L.extend(f"  • {f}" for f in findings)
            L.append("")

    if decision is not None:
        L.append("— SELF-IMPROVEMENT (champion/challenger) —")
        L.append(f"  Champion {decision['champion_composite']} vs best challenger "
                 f"{decision['best_challenger_composite']}")
        if decision["promoted"]:
            L.append(f"  ✅ PROMOTED new model {decision['new_version']} "
                     f"(learned: {decision['learned']})")
        else:
            L.append("  ⏸️ No challenger beat the champion by the margin — model unchanged.")
        L.append("")

    L.append(f"Active model: {get_active_version()}")
    return "\n".join(L)


def run(dry_run: bool = False, force_improve: bool = False) -> str:
    client = DhanClient()
    outcomes, new = _score_new_outcomes(client)
    j = judge(outcomes)
    findings = diagnose(j.card) if j.card else []

    decision = None
    if j.should_improve or force_improve:
        print("[running champion/challenger improvement loop...]")
        decision = champion_challenger.try_improve()
        log_metrics({"type": "improvement", **decision})

    if j.card is not None:
        log_metrics({"type": "scorecard", "n": j.card.n, "composite": j.card.composite,
                     "direction": j.card.direction, "range_hit": j.card.range_hit,
                     "calibration": j.card.calibration, "trade_pnl": j.card.trade_pnl,
                     "satisfactory": j.card.satisfactory})

    report = _build_review(new, j, findings, decision)
    if not dry_run and telegram.is_configured():
        telegram.send_message(report)
        print("[sent to Telegram]")
    print("\n" + report)
    return report


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-improve", action="store_true")
    a = ap.parse_args()
    run(dry_run=a.dry_run, force_improve=a.force_improve)
