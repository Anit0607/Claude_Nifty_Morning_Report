"""Agent 1 — Pre-Market Predictor (runs ~09:20 IST).

Pipeline: assemble live data (Dhan) -> build today's feature row -> champion model
prediction -> option-chain features -> trade plans -> deterministic report -> Telegram
(+ log the prediction).

Run:
  python -m scripts.run_agent1            # live: send to Telegram if configured, else print
  python -m scripts.run_agent1 --dry-run  # never send; print report (uses latest session)

On a non-trading day (or before 09:20), the latest available session's opening-range candle
is used so the full pipeline can be exercised end-to-end.
"""
from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd

from src.data.dhan_client import DhanClient
from src.data.global_cues import fetch_global_cues
from src.data.historical import load_training_frame
from src.data.intraday_history import (OR_FEATURE_COLS, build_opening_range_table,
                                        update_opening_range_cache)
from src.features.builder import build_feature_frame
from src.features.options import extract_option_features
from src.features.volatility import atr
from src.models.predictor import predict_outputs
from src.models.registry import load_bundle, get_active_version
from src.report.builder import build_report
from src.storage.logs import PREDICTIONS, log_prediction, read_jsonl
from src.trade.engine import build_trade_plans
from src.delivery import dispatch


def _latest_session(client: DhanClient) -> tuple[pd.Timestamp, dict]:
    """Most recent session with a 09:15 candle -> (date, opening-range dict)."""
    today = dt.date.today()
    intra = client.intraday((today - dt.timedelta(days=7)).isoformat(), today.isoformat(), interval="5")
    opens = intra[intra.index.time == dt.time(9, 15)]
    if opens.empty:
        raise RuntimeError("No opening-range candle available in the last 7 days.")
    last = opens.iloc[-1]
    date = pd.Timestamp(opens.index[-1]).normalize()
    rng = max(float(last["high"] - last["low"]), 1e-9)
    or_row = {
        "or_ret": float((last["close"] - last["open"]) / last["open"]),
        "or_range": float((last["high"] - last["low"]) / last["open"]),
        "or_close_pos": float((last["close"] - last["low"]) / rng),
    }
    return date, {"open": float(last["open"]), **or_row}


def _daily_frame(client: DhanClient, target_date: pd.Timestamp) -> pd.DataFrame:
    """Authoritative recent daily frame for live features.

    yfinance is unreliable on cloud/CI IPs (Yahoo throttles datacentre ranges → stale or
    missing rows, which corrupts prev_close/gap). So we source daily OHLC + India VIX from
    Dhan, which is reliable on the runner. Falls back to yfinance only if Dhan fails.
    """
    from src.data.historical import load_training_frame
    try:
        start = (target_date - pd.Timedelta(days=1100)).date().isoformat()
        end = target_date.date().isoformat()
        nifty = client.daily_history(start, end)
        nifty.index = pd.DatetimeIndex(nifty.index).normalize()
        df = nifty[["open", "high", "low", "close", "volume"]].copy()
        try:
            vix = client.daily_history(start, end, security_id=client.cfg["vix_security_id"])
            vix.index = pd.DatetimeIndex(vix.index).normalize()
            df["vix"] = vix["close"].reindex(df.index).ffill()
        except Exception:
            df["vix"] = float("nan")
        if df["vix"].isna().all():  # VIX from Dhan failed -> borrow yfinance VIX history
            df["vix"] = load_training_frame()["vix"].reindex(df.index).ffill()
        df = df[~df.index.duplicated(keep="last")].sort_index().dropna(
            subset=["open", "high", "low", "close"])
        # Match load_training_frame's schema — downstream (ATR, gap, VIX features) needs these.
        df["prev_close"] = df["close"].shift(1)
        df["prev_vix"] = df["vix"].shift(1)
        if len(df) < 300:
            raise RuntimeError(f"insufficient Dhan daily history ({len(df)} rows)")
        print(f"[daily] Dhan daily frame: {len(df)} sessions through {df.index.max().date()}")
        return df
    except Exception as exc:
        print(f"[warn] Dhan daily frame failed ({str(exc)[:100]}); falling back to yfinance")
        return load_training_frame()


def run(dry_run: bool = False, force: bool = False) -> str:
    import datetime as _dt
    _now_ist = _dt.datetime.utcnow() + _dt.timedelta(hours=5, minutes=30)
    as_of = _now_ist.strftime("%Y-%m-%d %H:%M") + " IST"
    _open = _now_ist.replace(hour=9, minute=20, second=0, microsecond=0)
    if _now_ist > _open + _dt.timedelta(minutes=20):
        as_of += "  ⚠️ LATE run — levels reflect this time, NOT the 9:20 open"
    client = DhanClient()

    # --- keep the opening-range cache current (cheap, incremental) ---
    try:
        update_opening_range_cache(client)
    except Exception as exc:
        print(f"[warn] OR cache update failed: {str(exc)[:120]}")

    # --- live opening range / today's open ---
    target_date, orc = _latest_session(client)
    open_price = orc["open"]

    # Backup-run guard: if this session was already reported (e.g. the punctual external
    # trigger ran at 9:22), skip so a late GitHub-scheduled run can't re-send a stale report.
    # --force bypasses it for manual testing.
    if not dry_run and not force:
        _existing = read_jsonl(PREDICTIONS)
        if not _existing.empty and str(target_date.date()) in set(_existing["date"]):
            print(f"[Agent 1] {target_date.date()} already reported — skipping (backup run). "
                  f"Use --force to re-send.")
            return ""

    # --- live VIX (best-effort) ---
    try:
        ltp = client.ltp()["data"]["IDX_I"]
        live_vix = float(ltp[str(client.cfg["vix_security_id"])]["last_price"])
    except Exception:
        live_vix = None

    # --- assemble daily frame (Dhan-sourced; yfinance fallback) + synthetic today row ---
    daily = _daily_frame(client, target_date)
    prior = daily[daily.index < target_date]

    # Score yesterday (and any unscored past day) from this same daily frame, so the
    # scorecard stays current from Agent 1's reliable run — no dependency on Agent 2.
    from src.scoring.review import score_pending_from_frame
    perf_card, last_outcome = score_pending_from_frame(daily, target_date, write=not dry_run)
    vix_val = live_vix if live_vix is not None else float(prior["vix"].iloc[-1])
    today_row = pd.DataFrame({
        "open": open_price, "high": open_price, "low": open_price, "close": open_price,
        "volume": 0.0, "vix": vix_val, "prev_close": float("nan"), "prev_vix": float("nan"),
    }, index=[target_date])
    d = pd.concat([prior, today_row])
    d["prev_close"] = d["close"].shift(1)
    d["prev_vix"] = d["vix"].shift(1)

    # --- opening-range table incl. today ---
    ortab = build_opening_range_table()
    ortab.loc[target_date, OR_FEATURE_COLS] = [orc["or_ret"], orc["or_range"], orc["or_close_pos"]]

    # --- features + prediction ---
    frame = build_feature_frame(d, opening_range=ortab, require_targets=False)
    row = frame.loc[[target_date]]
    bundle = load_bundle()
    preds = predict_outputs(bundle, row).iloc[0]
    atr_points = float(atr(prior).iloc[-1])

    # --- option-chain features (best-effort) ---
    opt = None
    try:
        expiry = client.expiry_list()[0]
        opt = extract_option_features(client.option_chain(expiry), expiry)
    except Exception as exc:
        print(f"[warn] option chain unavailable: {str(exc)[:120]}")

    # --- trade plans ---
    bias = "Bullish" if preds["p_up"] >= 0.5 else "Bearish"
    plans = build_trade_plans(
        open_price=open_price, expected_move=float(preds["expected_move"]),
        pred_high=float(preds["pred_high"]), pred_low=float(preds["pred_low"]),
        p_up=float(preds["p_up"]), p_down=float(preds["p_down"]),
        p_sideways=float(preds["p_sideways"]), p_up_reg=float(preds["p_up_reg"]),
        conf_direction=float(preds["conf_direction"]), conf_regime=float(preds["conf_regime"]),
        conf_range=float(preds["conf_range"]), atr_points=atr_points, opt=opt,
    )

    # --- report ---
    report = build_report(
        date_str=str(target_date.date()), mode="LIVE" if not dry_run else "DRY-RUN",
        open_price=open_price, prev_close=float(d.loc[target_date, "prev_close"]),
        gap_pct=float(row["gap_pct"].iloc[0]), india_vix=live_vix,
        or_ret=orc["or_ret"], or_range=orc["or_range"],
        pred_low=float(preds["pred_low"]), pred_high=float(preds["pred_high"]),
        expected_move_pts=open_price * float(preds["expected_move"]),
        p_up=float(preds["p_up"]), exp_magnitude_pts=open_price * float(preds["exp_magnitude"]),
        bias=bias, p_down=float(preds["p_down"]), p_sideways=float(preds["p_sideways"]),
        p_up_reg=float(preds["p_up_reg"]), conf_direction=float(preds["conf_direction"]),
        conf_regime=float(preds["conf_regime"]), conf_range=float(preds["conf_range"]),
        conf_overall=float(preds["conf_overall"]), plans=plans, opt=opt,
        global_summary=fetch_global_cues().summary(), as_of=as_of,
    )

    # --- log prediction (skip on dry-run; dedup handled by the early backup guard) ---
    if dry_run:
        print("[dry-run — prediction not logged]")
    else:
        log_prediction({
        "date": str(target_date.date()), "model_version": get_active_version(),
        "open": open_price, "p_up": float(preds["p_up"]),
        "p_down": float(preds["p_down"]), "p_sideways": float(preds["p_sideways"]),
        "p_up_reg": float(preds["p_up_reg"]),
        "pred_high": float(preds["pred_high"]), "pred_low": float(preds["pred_low"]),
        "expected_move": float(preds["expected_move"]),
        "dir_pred": int(preds["dir_pred"]), "conf_overall": float(preds["conf_overall"]),
        "or_ret": orc["or_ret"], "india_vix": live_vix, "atr_points": atr_points,
    })

    # --- append recent-performance block (scored from Agent 1's own run) ---
    if perf_card is not None:
        pl = ["", "— RECENT PERFORMANCE —"]
        if last_outcome:
            ok = "✅" if last_outcome["dir_pred"] == last_outcome["dir_actual"] else "❌"
            tr = last_outcome.get("trade_r")
            tr_s = f"{tr:+.2f}R" if isinstance(tr, float) and tr == tr else "n/a"
            pl.append(f"Last scored {last_outcome['date']}: direction {ok}, futures {tr_s}")
        pl.append(f"Rolling (n={perf_card.n}): direction {perf_card.direction}% | "
                  f"trade {perf_card.trade_pnl} (avg {perf_card.detail.get('mean_trade_r')}R) | "
                  f"range_hit {perf_card.range_hit} | composite {perf_card.composite}"
                  f" ({'satisfactory' if perf_card.satisfactory else 'below bar'})")
        report = report + "\n" + "\n".join(pl)

    # --- deliver ---
    if not dry_run and dispatch.is_configured():
        dispatch.send(report, subject=f"NIFTY Pre-Market Outlook — {target_date.date()}")
    elif not dry_run:
        print("[no delivery channel configured — printing only]")
    print("\n" + report)
    return report


if __name__ == "__main__":
    import sys
    try:  # ensure emoji/box-drawing chars print on Windows consoles (cp1252)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="never send; print report")
    ap.add_argument("--force", action="store_true", help="re-send even if already reported today")
    args = ap.parse_args()
    try:
        run(dry_run=args.dry_run, force=args.force)
    except Exception as exc:
        # Never fail silently — alert so a missing morning report is noticed.
        try:
            dispatch.send(f"{type(exc).__name__}: {str(exc)[:300]}", subject="⚠️ Agent 1 FAILED")
        except Exception:
            pass
        raise
