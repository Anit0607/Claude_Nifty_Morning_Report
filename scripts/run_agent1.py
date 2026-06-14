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
from src.delivery import telegram


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


def run(dry_run: bool = False) -> str:
    client = DhanClient()

    # --- keep the opening-range cache current (cheap, incremental) ---
    try:
        update_opening_range_cache(client)
    except Exception as exc:
        print(f"[warn] OR cache update failed: {str(exc)[:120]}")

    # --- live opening range / today's open ---
    target_date, orc = _latest_session(client)
    open_price = orc["open"]

    # --- live VIX (best-effort) ---
    try:
        ltp = client.ltp()["data"]["IDX_I"]
        live_vix = float(ltp[str(client.cfg["vix_security_id"])]["last_price"])
    except Exception:
        live_vix = None

    # --- assemble daily frame: real history strictly before target, + synthetic today row ---
    daily = load_training_frame()
    prior = daily[daily.index < target_date]
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
        global_summary=fetch_global_cues().summary(),
    )

    # --- log prediction (skip on dry-run; one record per date) ---
    existing = read_jsonl(PREDICTIONS)
    already = (not existing.empty) and (str(target_date.date()) in set(existing["date"]))
    if dry_run:
        print("[dry-run — prediction not logged]")
    elif already:
        print(f"[prediction for {target_date.date()} already logged — not duplicating]")
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

    # --- deliver ---
    if not dry_run and telegram.is_configured():
        telegram.send_message(report)
        print("[sent to Telegram]")
    else:
        if not dry_run:
            print("[telegram not configured — printing only]")
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
    run(**{"dry_run": ap.parse_args().dry_run})
