"""Thin client for the Meta (Facebook/Instagram) Graph API."""

from __future__ import annotations

import hashlib
import hmac
import logging

import requests

log = logging.getLogger("messenger")

GRAPH = "https://graph.facebook.com/v21.0"


class MetaClient:
    def __init__(self, page_access_token: str, app_secret: str = ""):
        self.token = page_access_token
        self.app_secret = app_secret

    # ------------------------------------------------------------- sending
    def send_text(self, recipient_id: str, text: str) -> None:
        resp = requests.post(
            f"{GRAPH}/me/messages",
            params={"access_token": self.token},
            json={
                "recipient": {"id": recipient_id},
                "messaging_type": "RESPONSE",
                "message": {"text": text},
            },
            timeout=15,
        )
        if not resp.ok:
            log.error("send_text failed: %s %s", resp.status_code, resp.text)
        resp.raise_for_status()

    def send_action(self, recipient_id: str, action: str = "typing_on") -> None:
        """Show the 'typing…' bubble — makes the bot feel human."""
        try:
            requests.post(
                f"{GRAPH}/me/messages",
                params={"access_token": self.token},
                json={"recipient": {"id": recipient_id}, "sender_action": action},
                timeout=10,
            )
        except requests.RequestException as exc:
            log.debug("send_action failed: %s", exc)

    # ---------------------------------------------------------- validation
    def verify_signature(self, body: bytes, signature_header: str | None) -> bool:
        """Verify X-Hub-Signature-256 so only Meta can hit our webhook.
        If no app secret is configured we're in local dev mode → allow."""
        if not self.app_secret:
            return True
        if not signature_header or not signature_header.startswith("sha256="):
            return False
        expected = hmac.new(self.app_secret.encode(), body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature_header[len("sha256="):])
