"""Walk-forward backtest over the historical feature frame.

Expanding-window, periodic-refit design (refit monthly, predict the days in between)
so it is honest about label leakage — models never see the day they predict — while
staying fast enough to run in CI. The shared ``fit_bundle`` / ``predict_outputs`` paths
mean the backtested behaviour matches live Agent 1.

A simple intraday-futures P&L proxy supplies the scorecard's ``trade_r`` term here; the
full 4-persona simulation replaces it once the trade engine exists (Phase 1/2). Both are
champion/challenger-comparable because they feed the same scorecard.

Note: the EGARCH feature uses full-sample parameters (a minor, documented simplification);
classifier/range fitting is strictly walk-forward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_settings
from src.features.builder import build_feature_frame
from src.features.volatility import atr
from src.models.predictor import predict_outputs
from src.models.train import fit_bundle
from src.scoring import scorecard


def _futures_pnl_r(row: pd.Series, atr_pts: float, settings: dict) -> float:
    """Intraday futures proxy: trade the model's direction, bracket by ATR. Returns R."""
    fp = settings["personas"]["futures_trader"]
    sl_pts = max(1e-9, fp["sl_atr_mult"] * atr_pts)
    tgt_pts = fp["target_atr_mult"] * atr_pts
    long = row["dir_pred"] == 1
    o, h, l, c = row["open"], row["act_high"], row["act_low"], row["close"]

    if long:
        if l <= o - sl_pts:
            return -1.0                      # stopped out
        if h >= o + tgt_pts:
            return tgt_pts / sl_pts          # target hit
        return (c - o) / sl_pts              # exit at close
    else:
        if h >= o + sl_pts:
            return -1.0
        if l <= o - tgt_pts:
            return tgt_pts / sl_pts
        return (o - c) / sl_pts


def run_backtest(initial_train: int = 750, refit_every: int = 21,
                 frame: pd.DataFrame | None = None,
                 feature_cols: list[str] | None = None) -> tuple[pd.DataFrame, scorecard.Scorecard]:
    settings = load_settings()
    if frame is None:
        from src.data.historical import load_training_frame
        frame = build_feature_frame(load_training_frame())

    atr_series = atr(frame).shift(1)  # ATR known at open (prev day)
    records = []

    i = initial_train
    n = len(frame)
    while i < n:
        train = frame.iloc[:i]
        block = frame.iloc[i : i + refit_every]
        if train["y_reg"].nunique() < 3 or len(block) == 0:
            i += refit_every
            continue

        bundle = fit_bundle(train, feature_cols=feature_cols)
        preds = predict_outputs(bundle, block)

        for idx in block.index:
            rec = {
                "date": idx,
                "open": block.at[idx, "open"],
                "close": block.at[idx, "close"],
                "act_high": block.at[idx, "high"],
                "act_low": block.at[idx, "low"],
                "dir_pred": int(preds.at[idx, "dir_pred"]),
                "dir_actual": int(block.at[idx, "y_dir"]),
                "p_up": float(preds.at[idx, "p_up"]),
                "pred_high": float(preds.at[idx, "pred_high"]),
                "pred_low": float(preds.at[idx, "pred_low"]),
                "p_down": float(preds.at[idx, "p_down"]),
                "p_sideways": float(preds.at[idx, "p_sideways"]),
                "p_up_reg": float(preds.at[idx, "p_up_reg"]),
                "reg_actual": block.at[idx, "y_reg"],
                "conf_overall": float(preds.at[idx, "conf_overall"]),
            }
            a = atr_series.get(idx, np.nan)
            rec["trade_r"] = _futures_pnl_r(pd.Series(rec), float(a), settings) if np.isfinite(a) else np.nan
            records.append(rec)

        i += refit_every

    rec_df = pd.DataFrame(records).set_index("date")
    # The scorecard's regime Brier needs columns p_down/p_sideways/p_up where p_up is the
    # regime up-probability. Build a clean scoring frame (direction prob is unused there).
    scoring_frame = rec_df.assign(p_up=rec_df["p_up_reg"])
    card = scorecard.compute(scoring_frame)
    return rec_df, card


if __name__ == "__main__":
    recs, card = run_backtest()
    print(f"backtest predictions: {len(recs)}  ({recs.index.min().date()} -> {recs.index.max().date()})")
    print("\n=== SCORECARD (full backtest) ===")
    for k, v in card.__dict__.items():
        if k != "detail":
            print(f"  {k:24s}: {v}")
    print("  detail:")
    for k, v in card.detail.items():
        print(f"      {k:20s}: {v}")
