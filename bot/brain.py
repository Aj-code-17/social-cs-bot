"""The reply engine.

Picks an LLM provider (OpenAI / Anthropic / Gemini) based on env vars,
falls back to a lightweight offline rule-based brain when no API key is
configured (great for testing the pipeline end-to-end without spending
a cent). Also detects when a conversation needs a human teammate.
"""

from __future__ import annotations

import logging
import os
import random
import re
from dataclasses import dataclass

from .config import BusinessConfig
from .memory import Memory
from .prompt import HANDOFF_TOKEN, build_messages

log = logging.getLogger("brain")

MAX_HISTORY = 12  # last N messages sent to the LLM for context


@dataclass
class BrainResult:
    text: str
    needs_human: bool = False


class Brain:
    def __init__(self, cfg: BusinessConfig, memory: Memory):
        self.cfg = cfg
        self.memory = memory
        self.provider = os.getenv("LLM_PROVIDER", "auto").lower()
        if self.provider == "auto":
            self.provider = self._detect_provider()
    

    # ------------------------------------------------------------------ API
    def generate(self, sender_id: str, user_text: str) -> BrainResult:
        """Generate a reply. Also stores both sides of the conversation."""
        self.memory.add_message(sender_id, "user", user_text)

        text: str | None = None
        if self.provider != "offline":
            try:
                text = self._call_llm(sender_id)
            except Exception as exc:  # noqa: BLE001 - never crash on LLM hiccups
                print(f"CRITICAL ERROR: {exc}")
        needs_human = False
        if text:
            if HANDOFF_TOKEN in text:
                needs_human = True
                text = text.replace(HANDOFF_TOKEN, "").strip()
        if not text:
            text, offline_flag = _offline_reply(self.cfg, user_text)
            needs_human = needs_human or offline_flag

        text = _tidy(text)
        self.memory.add_message(sender_id, "assistant", text)
        return BrainResult(text=text, needs_human=needs_human)

    def acknowledge_attachment(self, sender_id: str) -> BrainResult:
        """Natural reply for photos/voices/videos (we can't read those)."""
        self.memory.add_message(sender_id, "user", "[sent a photo/voice message]")
        text = random.choice(
            [
                "Thanks for sending that over! Give me a moment to have a look 😊",
                "Got it, thanks! Let me take a quick look at this.",
                "Thanks! Having a look at this now 😊",
            ]
        )
        self.memory.add_message(sender_id, "assistant", text)
        return BrainResult(text=text, needs_human=False)

    # ------------------------------------------------------------- providers
    @staticmethod
    def _detect_provider() -> str:
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.getenv("GOOGLE_API_KEY"):
            return "gemini"
        log.info("No LLM API key found — running with offline demo replies.")
        return "offline"

    def _call_llm(self, sender_id: str) -> str | None:
        history = self.memory.last_messages(sender_id, limit=MAX_HISTORY)
        messages = build_messages(self.cfg, history)

        if self.provider == "openai":
            from openai import OpenAI

            client = OpenAI()  # reads OPENAI_API_KEY
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=messages,
                
            )
           
            return resp.choices[0].message.content

        if self.provider == "anthropic":
            import anthropic

            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
            system = messages[0]["content"]
            chat = messages[1:]
            resp = client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
                system=system,
                messages=chat,
                temperature=0.8,
                max_tokens=120,
            )
            return "".join(b.text for b in resp.content if b.type == "text")

        if self.provider == "gemini":
            from google import genai
            from google.genai import types

            # The new SDK uses a Client object instead of genai.configure
            client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
            
            # Convert your chat history to the new SDK's strict data types
            gemini_history = [
                types.Content(
                    role="user" if m["role"] == "user" else "model",
                    parts=[types.Part.from_text(text=m["content"])]
                )
                for m in messages[1:-1]
            ]
            
            # Start the chat session using the updated model and system prompt
            chat = client.chats.create(
                model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
                history=gemini_history,
                config=types.GenerateContentConfig(
                    system_instruction=messages[0]["content"]
                )
            )
            
            # Send the latest user message
            resp = chat.send_message(messages[-1]["content"])
            return resp.text

        return None


# ====================================================================== utils
def _tidy(text: str) -> str:
    """Clean up small LLM artifacts."""
    text = text.strip().strip('"')
    text = re.sub(r"\s+\n", "\n", text)
    return text[:500]  # hard safety cap


_WORD = re.compile(r"[a-zA-Z]{3,}")
_STOP = {"you", "your", "yours", "the", "and", "for", "are", "can", "does",
         "have", "has", "any", "with", "that", "this", "what", "when", "where"}


def _faq_score(message: str, question: str) -> int:
    msg_words = set(_WORD.findall(message.lower())) - _STOP
    q_words = set(_WORD.findall(question.lower())) - _STOP
    return len(msg_words & q_words)


def _offline_reply(cfg: BusinessConfig, user_text: str) -> tuple[str, bool]:
    """Simple rule-based replies so the whole pipeline can be tested without
    an API key. Realistic conversations come from the LLM provider.
    Returns (text, needs_human)."""
    msg = user_text.lower()

    angry = ["angry", "terrible", "worst", "disgusting", "ridiculous",
             "unacceptable", "scam", "furious", "pathetic"]
    if any(w in msg for w in angry):
        return (random.choice([
            "Oh no, I'm really sorry to hear that 😟 Let me get this sorted with the team right away.",
            "I'm so sorry! Let me check this with the team and fix it for you ASAP.",
        ]), True)

    if any(w in msg for w in ["human", "real person", "manager", "agent", "staff member"]):
        return (random.choice([
            "Of course! Let me grab a teammate for you — one moment 😊",
            "Sure thing, getting someone from the team to jump in now!",
        ]), True)

    matched: list[str] = []

    best_faq, best_score = None, 0
    for f in cfg.faqs:
        s = _faq_score(msg, f.get("q", ""))
        if s > best_score:
            best_faq, best_score = f, s
    if best_faq and best_score >= 2:
        matched.append(random.choice(["Yep — {a}", "Great question! {a}", "{a}"])
                       .format(a=best_faq["a"]))

    rules = [
        (["hour", "open", "close", "timing", "when do you"],
         [f"We're open {cfg.hours} 😊", f"Sure! Our hours are {cfg.hours}."]),
        (["where", "location", "address", "located", "find you", "directions"],
         [f"You'll find us at {cfg.location} 😊", f"We're at {cfg.location} — hope to see you soon!"]),
        (["price", "cost", "how much", "charge", "rate", "fee", "pricelist", "price list"],
         [f"Here's what we charge: {cfg.pricing_inline}.", f"Our prices: {cfg.pricing_inline} 😊"]),
        (["book", "appointment", "schedule", "reserve", "slot"],
         [f"You can grab a slot here: {cfg.booking_link} 😊",
          f"Here's our booking link: {cfg.booking_link} — pick whatever time suits you!"]),
        (["cancel", "refund", "return", "exchange", "reschedule"],
         [cfg.policies_inline or "Let me check our policy on that for you!"]),
        (["service", "offer", "do you do", "what do you have", "menu"],
         [f"We offer {cfg.services_inline} 😊", f"Sure! We've got {cfg.services_inline}."]),
        (["thank", "shukriya", "shukria"],
         ["Anytime! 😊", "You're so welcome! 😊", "No problem at all!"]),
    ]
    for keywords, replies in rules:
        if any(k in msg for k in keywords):
            matched.append(random.choice(replies))
        if len(matched) >= 2:
            break

    if not matched:
        greet = any(w in msg.split() for w in ["hi", "hello", "hey", "salam", "aoa", "aoa!"])
        if greet:
            return (random.choice([
                "Hey there! 😊 How can I help you today?",
                "Hi! What can I do for you? 😊",
                "Hello hello! How can I help?",
            ]), False)
        return (random.choice([
            "Good question — let me check on that for you and get right back to you!",
            "Hmm, let me double-check that with the team and I'll get back to you ASAP!",
            "Let me find out for you — back in a jiffy! 😊",
        ]), True)

    return " ".join(matched), False
