"""Webhook server that receives Instagram & Facebook DM events from Meta
and replies with the bot.

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000

Meta will call:
    GET  /webhook  -> verification handshake (one-time setup)
    POST /webhook  -> every incoming customer message
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response

from bot.brain import Brain
from bot.config import load_config
from bot.memory import Memory
from bot.messenger import MetaClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("server")

# ------------------------------------------------------------ configuration
cfg = load_config(os.getenv("BUSINESS_CONFIG", "config/business.yaml"))
memory = Memory(os.getenv("MEMORY_DB", "conversations.db"))
brain = Brain(cfg, memory)
meta = MetaClient(
    page_access_token=os.getenv("PAGE_ACCESS_TOKEN", ""),
    app_secret=os.getenv("META_APP_SECRET", ""),
)
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "dev-verify-token")
RESPOND_TO = os.getenv("RESPOND_TO", "both").lower()
MIN_DELAY = float(os.getenv("MIN_DELAY_SECONDS", "1.5"))
MAX_DELAY = float(os.getenv("MAX_DELAY_SECONDS", "4.0"))
DEBOUNCE = float(os.getenv("DEBOUNCE_SECONDS", "6"))
HANDOFF_COOLDOWN = float(os.getenv("HANDOFF_COOLDOWN_MINUTES", "120"))
HANDOFF_WEBHOOK_URL = os.getenv("HANDOFF_WEBHOOK_URL", "")

app = FastAPI(title=f"CS Bot — {cfg.business_name}")

# Per-customer message buffers + pending reply tasks (message batching).
_buffers: dict[str, list[str]] = {}
_tasks: dict[str, asyncio.Task] = {}


# ---------------------------------------------------------------- endpoints
@app.get("/")
def root() -> dict:
    return {
        "status": "running",
        "business": cfg.business_name,
        "llm_provider": brain.provider,
        "respond_to": RESPOND_TO,
    }


@app.get("/webhook")
def verify_webhook(request: Request) -> Response:
    """Meta's one-time verification handshake."""
    q = request.query_params
    if q.get("hub.mode") == "subscribe" and q.get("hub.verify_token") == VERIFY_TOKEN:
        log.info("Webhook verified by Meta ✅")
        return Response(content=q.get("hub.challenge", ""), media_type="text/plain")
    log.warning("Webhook verification failed (wrong verify token?)")
    return Response(status_code=403)


@app.post("/webhook", response_model=None)
async def receive_webhook(request: Request):
    body = await request.body()
    if not meta.verify_signature(body, request.headers.get("X-Hub-Signature-256")):
        log.warning("Rejected POST with invalid signature")
        return Response(status_code=403)

    data = json.loads(body)
    platform = "instagram" if data.get("object") == "instagram" else "facebook"
    if RESPOND_TO not in ("both", platform):
        return {"status": f"ignored ({platform})"}

    for entry in data.get("entry", []):
        for event in entry.get("messaging", []):
            _dispatch_event(event, platform)

    # Always 200 fast — Meta retries deliveries on timeouts.
    return {"status": "ok"}


# ------------------------------------------------------------------ logic
def _dispatch_event(event: dict, platform: str) -> None:
    sender_id = (event.get("sender") or {}).get("id")
    message = event.get("message")
    if not sender_id or not message:
        return  # delivery/read receipts etc.

    if message.get("is_echo"):
        return  # ignore our own messages / page replies

    mid = message.get("mid")
    if memory.seen_mid(mid):
        log.info("Duplicate delivery of %s — skipping", mid)
        return

    text = (message.get("text") or "").strip()
    if text:
        # --- THE ESCAPE HATCH ---
        if text.lower() == "/resume":
            memory.clear_pause(sender_id)
            meta.send_text(sender_id, "🤖 AI auto-replies resumed. I am back!")
            log.info("Chat %s manually resumed via DM command", sender_id)
            return
        # ------------------------

        log.info("[%s] %s: %s", platform, sender_id, text)
        if memory.is_paused(sender_id):
            memory.add_message(sender_id, "user", text)
            log.info("Chat %s is paused for human — message stored only", sender_id)
            return
        
        _buffers.setdefault(sender_id, []).append(text)
        task = _tasks.get(sender_id)
        if task is None or task.done():
            _tasks[sender_id] = asyncio.create_task(_reply_loop(sender_id))
            
    elif message.get("attachments"):
        # Photo / voice / video: we can't read it — acknowledge naturally.
        if memory.is_paused(sender_id):
            return
        asyncio.create_task(_ack_attachment(sender_id))


async def _reply_loop(sender_id: str) -> None:
    """Waits a few seconds to collect rapid-fire messages — a human reads the
    whole burst before answering, so the bot should too."""
    try:
        while True:
            await asyncio.sleep(DEBOUNCE)
            texts = _buffers.pop(sender_id, [])
            if not texts:
                break
            combined = " ".join(texts)
            meta.send_action(sender_id, "typing_on")
            await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

            result = await asyncio.to_thread(brain.generate, sender_id, combined)

            if result.needs_human:
                # Flag BEFORE sending so a send failure never skips the alert.
                # (Pausing affects future messages, not the reply going out now.)
                _flag_handoff(sender_id, combined)

            # Split on blank lines so longer answers arrive as separate bubbles.
            bubbles = [b.strip() for b in result.text.split("\n\n") if b.strip()]
            for bubble in bubbles[:3]:
                try:
                    meta.send_text(sender_id, bubble)
                except Exception:
                    log.exception("Could not send reply to %s", sender_id)
                    break
                if len(bubbles) > 1:
                    await asyncio.sleep(0.8)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        log.exception("Reply loop crashed for %s", sender_id)
    finally:
        _tasks.pop(sender_id, None)


async def _ack_attachment(sender_id: str) -> None:
    try:
        meta.send_action(sender_id, "typing_on")
        await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
        result = await asyncio.to_thread(brain.acknowledge_attachment, sender_id)
        meta.send_text(sender_id, result.text)
        _flag_handoff(sender_id, "[attachment]")  # a human should look at it
    except Exception:  # noqa: BLE001
        log.exception("Attachment ack failed for %s", sender_id)


def _flag_handoff(sender_id: str, last_message: str) -> None:
    """Alert the human team and pause auto-replies for this customer."""
    memory.set_handoff_pause(sender_id, HANDOFF_COOLDOWN)
    log.warning("👤 HUMAN NEEDED for %s (last msg: %s)", sender_id, last_message)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {sender_id} | {last_message}\n"
    with open("handoffs.log", "a", encoding="utf-8") as fh:
        fh.write(line)
    if HANDOFF_WEBHOOK_URL:
        try:
            requests.post(
                HANDOFF_WEBHOOK_URL,
                json={
                    "text": f"👤 *Human needed* in {cfg.business_name} DMs\n"
                            f"Customer: `{sender_id}`\nLast message: {last_message}"
                },
                timeout=10,
            )
        except requests.RequestException as exc:
            log.warning("Handoff webhook failed: %s", exc)


@app.post("/resume/{sender_id}")
def resume_bot(sender_id: str) -> dict:
    """Call this after your team has handled the customer to unpause the bot."""
    memory.clear_pause(sender_id)
    log.info("Auto-replies resumed for %s", sender_id)
    return {"status": "resumed", "sender_id": sender_id}
