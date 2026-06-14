"""One-time base training: build the dataset, backtest, and save the champion bundle.

Run:  python -m scripts.train_initial             (cached data, uses opening-range if present)
      python -m scripts.train_initial --refresh    (re-download daily history first)
      python -m scripts.train_initial --no-opening-range

If the Dhan opening-range cache exists (see src/data/intraday_history.py), those features
are included — they lift directional accuracy materially — and the bundle's feature_cols
reflect that. Agent 1 fetches today's opening-range candle live to match.

Saves a versioned ModelBundle (fit on full history) with the walk-forward backtest
scorecard embedded in metadata, and points active.txt at it.
"""
from __future__ import annotations

import argparse

from src.backtest.harness import run_backtest
from src.data.historical import load_training_frame
from src.data.intraday_history import OR_CACHE, OR_FEATURE_COLS, build_opening_range_table
from src.features.builder import FEATURE_COLS, build_feature_frame
from src.models.registry import save_bundle
from src.models.train import fit_bundle


def main(refresh: bool = False, no_opening_range: bool = False) -> None:
    print("Loading historical data...")
    daily = load_training_frame(refresh=refresh)

    use_or = (not no_opening_range) and OR_CACHE.exists()
    if use_or:
        ortab = build_opening_range_table()
        frame = build_feature_frame(daily, opening_range=ortab)
        feature_cols = FEATURE_COLS + OR_FEATURE_COLS
        print(f"  using opening-range features: {OR_FEATURE_COLS}")
    else:
        frame = build_feature_frame(daily)
        feature_cols = FEATURE_COLS
        print("  daily-only features (no opening-range cache found)")
    print(f"  feature frame: {len(frame)} sessions "
          f"({frame.index.min().date()} -> {frame.index.max().date()}), {len(feature_cols)} features")

    print("Running walk-forward backtest for baseline metrics...")
    _, card = run_backtest(frame=frame, feature_cols=feature_cols)
    print(f"  composite={card.composite}  direction={card.direction}%  "
          f"calibration={card.calibration}  range_hit={card.range_hit}  trade_pnl={card.trade_pnl}")

    print("Fitting champion bundle on full history...")
    bundle = fit_bundle(frame, feature_cols=feature_cols,
                        metadata={"backtest": dict(card.__dict__), "uses_opening_range": use_or})
    version = save_bundle(bundle, make_active=True)
    print(f"Saved champion model version: {version} (now active)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download historical data")
    ap.add_argument("--no-opening-range", action="store_true", help="train daily-only")
    main(**vars(ap.parse_args()))
