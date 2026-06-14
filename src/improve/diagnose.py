"""Diagnose which component is dragging the scorecard down.

Deterministic heuristics comparing each sub-score to its target, producing human-readable
findings that go into the evening report and steer the champion/challenger search. (An
optional LLM layer could deepen this later; the rule-based version stands alone.)
"""
from __future__ import annotations

from src.config import load_settings
from src.scoring.scorecard import Scorecard


def diagnose(card: Scorecard) -> list[str]:
    sc = load_settings()["scoring"]
    floors = sc["satisfactory"]["floors"]
    findings: list[str] = []

    if card.direction < floors["direction"] * 100:
        findings.append(f"Direction accuracy {card.direction:.0f}% below floor "
                        f"{floors['direction']*100:.0f}% — directional edge weak.")
    if card.calibration < floors["calibration"] * 100:
        findings.append(f"Probability calibration {card.calibration:.0f} below floor "
                        f"{floors['calibration']*100:.0f} (Brier {card.detail.get('brier')}).")
    if card.range_hit < 50:
        findings.append(f"Range hit {card.range_hit:.0f}: containment "
                        f"{card.detail.get('range_contained')}%, efficiency "
                        f"{card.detail.get('range_efficiency')}% — band width/calibration is the lever.")
    if card.trade_pnl == card.trade_pnl and card.trade_pnl < 50:  # not NaN and weak
        findings.append(f"Trade P&L score {card.trade_pnl:.0f} (mean {card.detail.get('mean_trade_r')}R) "
                        f"— expectancy negative.")
    if not findings:
        findings.append("No single component below floor; composite shortfall is broad-based.")
    return findings
