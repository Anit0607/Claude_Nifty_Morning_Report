"""Champion/challenger self-improvement — the auto-healing core.

When performance is unsatisfactory, search a small grid of tunable parameter sets,
evaluate each by walk-forward backtest, and **promote a challenger only if it beats the
current champion on backtest by a margin**. This is the safety brake: the system can only
change itself in a direction proven better on history — it cannot drift worse.

Promotion persists the winning parameters to ``config/learned.yaml`` (so live Agent 1
matches), retrains the champion bundle, and points the registry at the new version.

v1 tunes the range-model band quantile (range hit is the current weak spot and these
params don't touch the classifiers' labels). The grid is intentionally small to bound
nightly compute; it can be widened as more data accumulates.
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.backtest.harness import run_backtest
from src.config import load_learned, save_learned, set_overrides, _deep_merge
from src.data.historical import load_training_frame
from src.data.intraday_history import OR_FEATURE_COLS, build_opening_range_table
from src.features.builder import FEATURE_COLS, build_feature_frame
from src.models.registry import save_bundle
from src.models.train import fit_bundle

PROMOTION_MARGIN = 0.5            # composite points the challenger must beat champion by
_CHALLENGER_REFIT_EVERY = 63      # coarser refit for faster candidate evaluation


def _candidate_overrides() -> list[dict]:
    return [{"tunable": {"range": {"band_quantile": q}}} for q in (0.75, 0.80, 0.85, 0.90)]


def _evaluate(frame, feature_cols, overrides) -> float:
    set_overrides(overrides)
    try:
        _, card = run_backtest(frame=frame, feature_cols=feature_cols,
                               refit_every=_CHALLENGER_REFIT_EVERY)
    finally:
        set_overrides(None)
    return card.composite


def try_improve() -> dict:
    daily = load_training_frame()
    ortab = build_opening_range_table()
    frame = build_feature_frame(daily, opening_range=ortab)
    feature_cols = FEATURE_COLS + OR_FEATURE_COLS

    champion_score = _evaluate(frame, feature_cols, None)
    results = [(ov, _evaluate(frame, feature_cols, ov)) for ov in _candidate_overrides()]
    best_ov, best_score = max(results, key=lambda x: x[1])

    decision = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "champion_composite": round(champion_score, 2),
        "best_challenger_composite": round(best_score, 2),
        "best_override": best_ov,
        "candidates": [{"override": ov, "composite": round(s, 2)} for ov, s in results],
        "promoted": False,
    }

    if best_score > champion_score + PROMOTION_MARGIN:
        learned = _deep_merge(load_learned(), best_ov)
        save_learned(learned)                       # persist so live + retrain use it
        bundle = fit_bundle(frame, feature_cols=feature_cols,
                            metadata={"promoted_from_composite": round(champion_score, 2),
                                      "to_composite": round(best_score, 2),
                                      "learned": learned})
        version = save_bundle(bundle, make_active=True)
        decision.update(promoted=True, new_version=version, learned=learned)

    return decision
