"""Email delivery via Gmail SMTP (SSL).

Reads EMAIL_USER (the Gmail address), EMAIL_APP_PASSWORD (a Google *App Password*, not the
account password — requires 2FA enabled), and EMAIL_TO (recipient; comma-separate for
several). Works reliably in India and has no message-length limit, so it carries the full
multi-section report. ``is_configured()`` lets the dispatcher skip it when unset.
"""
from __future__ import annotations

import smtplib
import ssl
from email.message import EmailMessage

from src.config import get_env


def is_configured() -> bool:
    return bool(get_env("EMAIL_USER") and get_env("EMAIL_APP_PASSWORD") and get_env("EMAIL_TO"))


def send(subject: str, body: str) -> bool:
    user = get_env("EMAIL_USER", required=True)
    password = get_env("EMAIL_APP_PASSWORD", required=True)
    to = get_env("EMAIL_TO", required=True)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to
    msg.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context, timeout=30) as server:
        server.login(user, password)
        server.send_message(msg)
    return True
