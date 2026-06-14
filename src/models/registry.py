"""Versioned model registry.

A "model bundle" is everything Agent 1 needs to make a prediction: the fitted
direction/regime/range estimators, their calibration constants, the feature column
order, and metadata (training date, data range, backtest metrics). Bundles are written
to ``data/models/<version>/bundle.joblib`` and a ``data/models/active.txt`` pointer
names the live version.

Agent 2's champion/challenger loop writes a new version and only repoints ``active.txt``
if the challenger beats the champion on backtest.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from src.config import load_settings, project_path


@dataclass
class ModelBundle:
    direction: Any
    regime: Any
    range_model: Any
    feature_cols: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def _models_dir() -> Path:
    d = project_path(load_settings()["models"]["dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_version() -> str:
    return datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")


def save_bundle(bundle: ModelBundle, version: str | None = None, *, make_active: bool = True) -> str:
    version = version or new_version()
    vdir = _models_dir() / version
    vdir.mkdir(parents=True, exist_ok=True)

    bundle.metadata.setdefault("version", version)
    bundle.metadata.setdefault("created_utc", datetime.now(timezone.utc).isoformat())

    joblib.dump(bundle, vdir / "bundle.joblib")
    with open(vdir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(bundle.metadata, fh, indent=2, default=str)

    if make_active:
        set_active(version)
    return version


def set_active(version: str) -> None:
    with open(_models_dir() / "active.txt", "w", encoding="utf-8") as fh:
        fh.write(version)


def get_active_version() -> str | None:
    pointer = _models_dir() / "active.txt"
    if pointer.exists():
        return pointer.read_text(encoding="utf-8").strip() or None
    return None


def load_bundle(version: str | None = None) -> ModelBundle:
    version = version or get_active_version()
    if not version:
        raise RuntimeError("No active model version. Run scripts/train_initial.py first.")
    path = _models_dir() / version / "bundle.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}")
    return joblib.load(path)


def list_versions() -> list[str]:
    return sorted(p.name for p in _models_dir().iterdir() if p.is_dir())
