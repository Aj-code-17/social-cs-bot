"""Local test chat — talk to the bot from your terminal, no Meta needed.

    python chat.py                 # chat as a test customer
    python chat.py --sender sara   # pretend to be customer 'sara'

Commands inside the chat:
    /reset     wipe this customer's history
    /new <id>  switch to a different customer
    /quit      exit
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

from dotenv import load_dotenv

load_dotenv(override=True)

from bot.brain import Brain  # noqa: E402
from bot.config import load_config  # noqa: E402
from bot.memory import Memory  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Test the CS bot locally")
    ap.add_argument("--config", default=os.getenv("BUSINESS_CONFIG", "config/business.yaml"))
    ap.add_argument("--sender", default="test-customer")
    ap.add_argument("--db", default=":memory:", help="SQLite path (default: in-memory)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    memory = Memory(args.db)
    brain = Brain(cfg, memory)
    sender = args.sender

    print("=" * 60)
    print(f"  Testing bot for:  {cfg.business_name}")
    print(f"  LLM provider:     {brain.provider}")
    print(f"  Chatting as:      {sender}")
    print("  Commands: /reset · /new <id> · /quit")
    print("=" * 60)

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue

        if user_text == "/quit":
            break
        if user_text == "/reset":
            memory.reset(sender)
            print(f"🧹 History cleared for {sender}")
            continue
        if user_text.startswith("/new "):
            sender = user_text.split(maxsplit=1)[1]
            print(f"🔄 Now chatting as: {sender}")
            continue

        if memory.is_paused(sender):
            memory.add_message(sender, "user", user_text)
            print(f"{cfg.business_name}: (💤 auto-replies paused — a human is handling this chat)")
            continue

        time.sleep(random.uniform(0.4, 1.0))  # pretend typing
        result = brain.generate(sender, user_text)
        print(f"{cfg.business_name}: {result.text}")
        if result.needs_human:
            memory.set_handoff_pause(sender, 5)
            print("   [👤 flagged for a human teammate — bot will stay quiet in this chat. "
                  "Use /reset to unpause.]")

    print("Bye! 👋")
    sys.exit(0)


if __name__ == "__main__":
    main()
