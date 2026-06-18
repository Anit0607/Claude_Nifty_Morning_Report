"""Pluggable notification dispatch — sends to every configured channel.

Each channel module exposes ``is_configured() -> bool`` and ``send(subject, body) -> bool``.
To add a channel (Discord, Slack, WhatsApp...), implement those two functions and append it
to ``_CHANNELS``. Delivery is best-effort: one channel failing never stops the others, and a
total failure is reported (the caller can decide what to do).
"""
from __future__ import annotations

from src.delivery import email_sender, telegram

# Order = priority for logging only; all configured channels receive the message.
_CHANNELS = [("email", email_sender), ("telegram", telegram)]


def is_configured() -> bool:
    return any(channel.is_configured() for _, channel in _CHANNELS)


def send(body: str, subject: str = "NIFTY Quant Report") -> bool:
    """Send to all configured channels. Returns True if at least one succeeded."""
    sent = False
    for name, channel in _CHANNELS:
        if not channel.is_configured():
            continue
        try:
            channel.send(subject, body)
            sent = True
            print(f"[delivery] sent via {name}")
        except Exception as exc:
            print(f"[delivery] {name} failed: {str(exc)[:150]}")
    if not sent:
        print("[delivery] no channel configured (or all failed)")
    return sent
