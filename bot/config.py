"""Loads the business profile from YAML — the single source of truth
the bot is allowed to answer from."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class BusinessConfig:
    business_name: str = "Our Business"
    brand_tone: str = "Friendly & casual"
    hours: str = ""
    location: str = ""
    pricing: str = ""
    services: list[str] = field(default_factory=list)
    policies: str = ""
    booking_link: str = ""
    faqs: list[dict] = field(default_factory=list)

    @property
    def services_inline(self) -> str:
        return ", ".join(self.services)

    @property
    def pricing_inline(self) -> str:
        """Pricing as a single line, e.g. 'Haircut: Rs 1,500, Manicure: Rs 1,200'."""
        lines = [ln.strip() for ln in self.pricing.strip().splitlines() if ln.strip()]
        return ", ".join(lines)

    @property
    def policies_inline(self) -> str:
        lines = [ln.strip() for ln in self.policies.strip().splitlines() if ln.strip()]
        return " ".join(lines)


def load_config(path: str | Path) -> BusinessConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Business config not found at {p.resolve()}\n"
            "Copy config/business.yaml and fill in the client's details."
        )
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    services = raw.get("services", []) or []
    if isinstance(services, str):
        services = [services]
    return BusinessConfig(
        business_name=raw.get("business_name", "Our Business"),
        brand_tone=raw.get("brand_tone", "Friendly & casual"),
        hours=raw.get("hours", ""),
        location=raw.get("location", ""),
        pricing=raw.get("pricing", ""),
        services=services,
        policies=raw.get("policies", ""),
        booking_link=raw.get("booking_link", ""),
        faqs=raw.get("faqs", []) or [],
    )
