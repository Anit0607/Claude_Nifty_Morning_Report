"""Decide whether recent performance is "satisfactory".

Wraps the scorecard over a rolling window of scored outcomes. Returns the scorecard plus
flags for (a) whether there is enough history to judge at all and (b) whether the
improvement loop should run.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import load_settings
from src.scoring import scorecard

# Minimum scored sessions before we trust a judgment / attempt model changes.
MIN_SESSIONS_TO_JUDGE = 10
MIN_SESSIONS_TO_IMPROVE = 40


@dataclass
class Judgment:
    enough_data: bool
    should_improve: bool
    card: scorecard.Scorecard | None
    window: int


def judge(outcomes: pd.DataFrame) -> Judgment:
    window = load_settings()["scoring"]["satisfactory"]["rolling_window"]
    n = len(outcomes)
    if n < MIN_SESSIONS_TO_JUDGE:
        return Judgment(enough_data=False, should_improve=False, card=None, window=window)

    recent = outcomes.sort_values("date").tail(window)
    card = scorecard.compute(recent)
    should_improve = (not card.satisfactory) and n >= MIN_SESSIONS_TO_IMPROVE
    return Judgment(enough_data=True, should_improve=should_improve, card=card, window=window)
