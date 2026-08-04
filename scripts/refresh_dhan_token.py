"""Keep the Dhan access token alive without a daily login.

Dhan caps access tokens at 24h (a SEBI requirement), but ``/v2/RenewToken`` mints a fresh
24h token from the *current* one — no browser, no 2FA. Running this on a schedule keeps the
chain going indefinitely; the only manual step is the initial bootstrap.

Flow: read DHAN_ACCESS_TOKEN -> call RenewToken -> write the new token back into the
repository secret (encrypted with the repo's public key, as the GitHub API requires) so the
agents pick it up on their next run.

The chain only breaks if renewal fails for a full 24h, so the workflow runs twice daily and
emails on failure.

Env:
    DHAN_ACCESS_TOKEN, DHAN_CLIENT_ID   current credentials
    GH_TOKEN                            PAT with "Secrets: read and write" on this repo
    GITHUB_REPOSITORY                   owner/repo (set automatically by Actions)

    python -m scripts.refresh_dhan_token            # renew + update the secret
    python -m scripts.refresh_dhan_token --check    # report expiry only, change nothing
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import sys

import requests

from src.config import get_env

_RENEW_URL = "https://api.dhan.co/v2/RenewToken"
_SECRET_NAME = "DHAN_ACCESS_TOKEN"


def token_expiry(token: str) -> dt.datetime | None:
    """Read the `exp` claim from the JWT payload (no signature check — informational)."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)          # restore base64url padding
        exp = json.loads(base64.urlsafe_b64decode(payload))["exp"]
        return dt.datetime.fromtimestamp(int(exp), tz=dt.timezone.utc)
    except Exception:
        return None


def renew(token: str, client_id: str) -> str:
    resp = requests.get(_RENEW_URL, timeout=30, headers={
        "access-token": token, "dhanClientId": client_id, "Accept": "application/json"})
    if resp.status_code != 200:
        raise RuntimeError(f"RenewToken failed: HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    new_token = data.get("accessToken") or data.get("access_token") or data.get("token")
    if not new_token:
        raise RuntimeError(f"RenewToken response had no token: {str(data)[:300]}")
    return new_token


def update_github_secret(name: str, value: str, repo: str, gh_token: str) -> None:
    """Store a repository secret (GitHub requires libsodium sealed-box encryption)."""
    from nacl import encoding, public

    headers = {"Authorization": f"Bearer {gh_token}",
               "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28"}

    key_resp = requests.get(f"https://api.github.com/repos/{repo}/actions/secrets/public-key",
                            headers=headers, timeout=30)
    if key_resp.status_code != 200:
        raise RuntimeError(f"public-key fetch failed: HTTP {key_resp.status_code}: "
                           f"{key_resp.text[:200]}")
    key_data = key_resp.json()

    sealed = public.SealedBox(
        public.PublicKey(key_data["key"].encode(), encoding.Base64Encoder())
    ).encrypt(value.encode())

    put = requests.put(
        f"https://api.github.com/repos/{repo}/actions/secrets/{name}", headers=headers, timeout=30,
        json={"encrypted_value": base64.b64encode(sealed).decode(), "key_id": key_data["key_id"]})
    if put.status_code not in (201, 204):
        raise RuntimeError(f"secret update failed: HTTP {put.status_code}: {put.text[:200]}")


def main(check: bool = False) -> None:
    token = get_env("DHAN_ACCESS_TOKEN", required=True)
    client_id = get_env("DHAN_CLIENT_ID", required=True)

    exp = token_expiry(token)
    if exp:
        left = exp - dt.datetime.now(dt.timezone.utc)
        hours = left.total_seconds() / 3600
        print(f"current token expires {exp:%Y-%m-%d %H:%M} UTC ({hours:.1f}h left)")
        if hours < 0:
            raise RuntimeError(
                "Token already EXPIRED — RenewToken only works on active tokens. "
                "Bootstrap manually: log in to web.dhan.co, generate a token, and update the "
                f"{_SECRET_NAME} secret once. The chain then self-sustains.")
    if check:
        return

    new_token = renew(token, client_id)
    new_exp = token_expiry(new_token)
    print(f"renewed OK — new token valid until {new_exp:%Y-%m-%d %H:%M} UTC"
          if new_exp else "renewed OK")

    repo = get_env("GITHUB_REPOSITORY")
    gh_token = get_env("GH_TOKEN")
    if repo and gh_token:
        update_github_secret(_SECRET_NAME, new_token, repo, gh_token)
        print(f"updated {_SECRET_NAME} secret in {repo}")
    elif repo:
        # In CI without a PAT this is NOT a success: RenewToken has already invalidated the
        # old token, so the stored secret is now dead. The new token cannot be rescued from
        # the log either — GitHub masks it — so the only fix is a manual re-bootstrap.
        raise RuntimeError(
            "Renewed the token but GH_SECRETS_TOKEN is missing, so the secret was NOT updated. "
            "RenewToken invalidates the old token on renewal, and GitHub masks the new one in "
            "logs, so it cannot be recovered here. FIX: (1) add a PAT with 'Secrets: read and "
            "write' as the GH_SECRETS_TOKEN secret, (2) generate a fresh token at web.dhan.co "
            f"and set {_SECRET_NAME}, (3) re-run this workflow.")
    else:
        # Local run: print so it can be pasted, rather than silently doing nothing.
        print("\n[local run — secret not updated; paste this into the "
              f"{_SECRET_NAME} secret]")
        print(f"new token:\n{new_token}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report expiry only")
    args = ap.parse_args()
    try:
        main(check=args.check)
    except Exception as exc:
        # Alert loudly — a silent failure here breaks tomorrow's whole pipeline.
        try:
            from src.delivery import dispatch
            dispatch.send(f"{type(exc).__name__}: {str(exc)[:400]}\n\n"
                          "If the token has fully expired, bootstrap once at web.dhan.co and "
                          "update the DHAN_ACCESS_TOKEN secret.",
                          subject="⚠️ Dhan token refresh FAILED")
        except Exception:
            pass
        raise
