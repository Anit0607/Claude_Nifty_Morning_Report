"""Configuration + environment loading.

`load_settings()` reads config/settings.yaml (the single source of tunable + structural
config). `get_env()` reads secrets from the process environment, loading a local `.env`
if present (via python-dotenv) so the same code works locally and in GitHub Actions.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Project root = parent of the `src` package directory.
ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = ROOT / "config" / "settings.yaml"
# Parameters learned/promoted by Agent 2's champion-challenger loop. Kept separate from the
# hand-written base config so its comments survive and learned changes are clear in git.
LEARNED_PATH = ROOT / "config" / "learned.yaml"


@lru_cache(maxsize=1)
def _load_base() -> dict[str, Any]:
    with open(SETTINGS_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# Runtime overrides for the champion/challenger loop (Agent 2). When set, load_settings()
# returns a deep-merged copy so candidate parameter sets can be backtested without editing
# the YAML. Always cleared in a finally-block by the caller.
_OVERRIDES: dict[str, Any] = {}


def _deep_merge(base: dict, over: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def set_overrides(overrides: dict[str, Any] | None) -> None:
    """Set (or clear, with None/{}) the runtime config overrides."""
    global _OVERRIDES
    _OVERRIDES = overrides or {}


def load_learned() -> dict[str, Any]:
    """Persistent learned-parameter overlay written by Agent 2 (may be empty)."""
    if LEARNED_PATH.exists():
        with open(LEARNED_PATH, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


def save_learned(learned: dict[str, Any]) -> None:
    with open(LEARNED_PATH, "w", encoding="utf-8") as fh:
        yaml.safe_dump(learned, fh, sort_keys=False)


def load_settings() -> dict[str, Any]:
    """Return config merged as: base <- learned (Agent 2) <- runtime overrides."""
    merged = _deep_merge(_load_base(), load_learned())
    if _OVERRIDES:
        merged = _deep_merge(merged, _OVERRIDES)
    return merged


def _ensure_dotenv_loaded() -> None:
    """Load .env from the project root if it exists. Safe to call repeatedly."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # dotenv optional; env vars may already be set (CI)
        return
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def get_env(key: str, default: str | None = None, required: bool = False) -> str | None:
    """Fetch a secret from the environment (loading .env first).

    If `required` and the value is missing/empty, raise a clear error naming the key.
    """
    _ensure_dotenv_loaded()
    value = os.environ.get(key, default)
    if required and not value:
        raise RuntimeError(
            f"Missing required environment variable '{key}'. "
            f"Set it in your .env file or as a GitHub Actions secret."
        )
    return value


def project_path(*parts: str) -> Path:
    """Build an absolute path rooted at the project directory."""
    return ROOT.joinpath(*parts)
