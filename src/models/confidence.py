"""Confidence scoring (0-100) for each output component and overall.

Confidence blends two things:
  * model certainty — how far the predicted probability is from a no-information prior
    (0.5 for direction, 1/3 for the 3-class regime), and
  * signal agreement — whether the independent signals (gap, Markov tilt, model) point
    the same way. A confident call backed by disagreeing signals is down-weighted.

Range confidence reflects volatility-regime stability: bands are less trustworthy when
VIX is in the Extreme regime or lurching day-to-day. Everything here is deterministic;
the scorecard's "confidence reliability" term later checks these scores actually track
accuracy.
"""
from __future__ import annotations

from src.config import load_settings


def _edge_to_conf(edge: float) -> float:
    """Map an edge in [0,1] to a 50-100 confidence; clamp to [50,99]."""
    return float(max(50.0, min(99.0, 50.0 + 50.0 * edge)))


def direction_confidence(p_up: float, gap_pct: float, markov_p_up: float, markov_p_down: float) -> float:
    prob_edge = abs(p_up - 0.5) * 2.0
    prob_conf = _edge_to_conf(prob_edge)

    # Agreement: do model, gap, and Markov tilt share the model's direction?
    model_dir = 1 if p_up >= 0.5 else -1
    signals = [
        1 if gap_pct >= 0 else -1,
        1 if (markov_p_up - markov_p_down) >= 0 else -1,
    ]
    agree = sum(1 for s in signals if s == model_dir) / len(signals)  # 0, .5, 1

    w = float(load_settings()["tunable"]["confidence"]["agreement_weight"])
    return round((1 - w) * prob_conf + w * (50.0 + 50.0 * agree), 1)


def regime_confidence(p_down: float, p_sideways: float, p_up: float) -> float:
    p_max = max(p_down, p_sideways, p_up)
    edge = (p_max - 1 / 3) / (1 - 1 / 3)
    return round(_edge_to_conf(max(0.0, edge)), 1)


def range_confidence(vix_regime: float, vix_change: float) -> float:
    base = 80.0
    base -= 12.0 * (float(vix_regime) / 3.0)        # Extreme regime -> less reliable band
    base -= min(15.0, abs(float(vix_change)) * 2.0)  # volatile VIX -> less reliable
    return round(max(30.0, min(90.0, base)), 1)


def overall_confidence(direction: float, regime: float, range_: float) -> float:
    # Direction and the probability triple are the headline calls; weight them higher.
    return round(0.4 * direction + 0.3 * regime + 0.3 * range_, 1)
