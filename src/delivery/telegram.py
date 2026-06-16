"""Telegram delivery via the Bot API — resilient, best-effort.

Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from the environment. ``is_configured()``
lets callers skip cleanly when creds are absent.

Robustness: each chunk is retried with backoff on timeouts / 429 / 5xx, with a generous
read timeout. After exhausting retries it logs a warning and returns False rather than
raising — a flaky network must NOT turn into a crash + false "agent failed" alert (a read
timeout often means the message was actually delivered, just the response was slow).

Messages are plain text (no Markdown parse mode) so box-drawing chars and emoji render
verbatim. Long messages are split under Telegram's 4096-char limit.
"""
from __future__ import annotations

import time

import requests

from src.config import get_env

_API = "https://api.telegram.org/bot{token}/sendMessage"
_LIMIT = 4000
_TIMEOUT = (10, 45)        # (connect, read) seconds — generous read for slow responses
_RETRIES = 4


def is_configured() -> bool:
    return bool(get_env("TELEGRAM_BOT_TOKEN") and get_env("TELEGRAM_CHAT_ID"))


def _chunks(text: str, size: int = _LIMIT) -> list[str]:
    out, buf = [], ""
    for ln in text.split("\n"):
        if len(buf) + len(ln) + 1 > size:
            out.append(buf)
            buf = ""
        buf += ln + "\n"
    if buf:
        out.append(buf)
    return out


def _send_chunk(url: str, chat_id: str, chunk: str) -> bool:
    last = None
    for attempt in range(_RETRIES):
        try:
            r = requests.post(url, json={"chat_id": chat_id, "text": chunk}, timeout=_TIMEOUT)
            if r.status_code == 200:
                return True
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
                time.sleep(2 * (attempt + 1))
                continue
            print(f"[telegram] non-retryable HTTP {r.status_code}: {r.text[:200]}")
            return False
        except requests.exceptions.RequestException as exc:
            last = exc  # timeouts, connection errors -> retry (message may have gone through)
            time.sleep(2 * (attempt + 1))
    print(f"[telegram] gave up after {_RETRIES} attempts: {last}")
    return False


def send_message(text: str) -> bool:
    """Best-effort send (chunked + retried). Returns True if all chunks acknowledged."""
    token = get_env("TELEGRAM_BOT_TOKEN", required=True)
    chat_id = get_env("TELEGRAM_CHAT_ID", required=True)
    url = _API.format(token=token)
    ok = True
    for chunk in _chunks(text):
        ok = _send_chunk(url, chat_id, chunk) and ok
    return ok
