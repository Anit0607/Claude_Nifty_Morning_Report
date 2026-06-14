"""Telegram delivery via the Bot API.

Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from the environment. ``is_configured()``
lets callers (e.g. a dry run) skip sending cleanly when creds are absent. Messages are
sent as plain text (no Markdown parse mode) so the report's box-drawing chars and symbols
render verbatim without escaping issues. Long messages are split under Telegram's 4096-char
limit.
"""
from __future__ import annotations

import requests

from src.config import get_env

_API = "https://api.telegram.org/bot{token}/sendMessage"
_LIMIT = 4000


def is_configured() -> bool:
    return bool(get_env("TELEGRAM_BOT_TOKEN") and get_env("TELEGRAM_CHAT_ID"))


def _chunks(text: str, size: int = _LIMIT) -> list[str]:
    lines, out, buf = text.split("\n"), [], ""
    for ln in lines:
        if len(buf) + len(ln) + 1 > size:
            out.append(buf)
            buf = ""
        buf += ln + "\n"
    if buf:
        out.append(buf)
    return out


def send_message(text: str) -> None:
    token = get_env("TELEGRAM_BOT_TOKEN", required=True)
    chat_id = get_env("TELEGRAM_CHAT_ID", required=True)
    url = _API.format(token=token)
    for chunk in _chunks(text):
        resp = requests.post(url, json={"chat_id": chat_id, "text": chunk}, timeout=20)
        if resp.status_code != 200:
            raise RuntimeError(f"Telegram send failed: HTTP {resp.status_code}: {resp.text[:300]}")
