"""Append-only JSONL logs — the system's persistent memory (committed to git).

    predictions.jsonl  — one row per Agent 1 morning call (all outputs + model version)
    outcomes.jsonl     — one row per Agent 2 evening review (actuals + scored metrics)
    metrics.jsonl      — rolling scorecard snapshots / champion-challenger decisions

JSONL keeps the history append-only and diff-friendly in git. Agent 2 joins predictions to
outcomes on ``date`` to score the model and drive the self-improvement loop.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import project_path

PREDICTIONS = project_path("data", "logs", "predictions.jsonl")
OUTCOMES = project_path("data", "logs", "outcomes.jsonl")
METRICS = project_path("data", "logs", "metrics.jsonl")


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def log_prediction(record: dict) -> None:
    append_jsonl(PREDICTIONS, record)


def log_outcome(record: dict) -> None:
    append_jsonl(OUTCOMES, record)


def log_metrics(record: dict) -> None:
    append_jsonl(METRICS, record)
