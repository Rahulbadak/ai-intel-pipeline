"""Deterministic pricing-model classifier. Returns the enum plus the exact page snippet that
justifies it, so every PRODUCT row is traceable to its source text. The LLM is only a fallback."""
from __future__ import annotations

import re

_FREE_TIER = re.compile(
    r"(free (plan|tier|forever)|freemium|start(ed)? (for )?free|free to (use|start|try)|get started free|always free)", re.I)
_TRIAL = re.compile(r"free trial|try (it )?free for", re.I)
_PRICE = re.compile(r"[\$€£₹]\s?\d")
_ENT = re.compile(r"(contact (sales|us for pricing)|book a demo|request (a )?demo|enterprise (plan|pricing|grade)|custom pricing|talk to sales)", re.I)
_OSS = re.compile(r"(open[- ]source|apache 2|mit license|100% free|completely free)", re.I)

ENUM = ("FREE", "FREEMIUM", "PAID", "ENTERPRISE")


def _snippet(text: str, m: re.Match) -> str:
    return re.sub(r"\s+", " ", text[max(0, m.start() - 60): m.end() + 60]).strip()


def infer_pricing(text: str) -> tuple[str | None, str | None]:
    if not text:
        return None, None
    free, trial, price = _FREE_TIER.search(text), _TRIAL.search(text), _PRICE.search(text)
    ent, oss = _ENT.search(text), _OSS.search(text)
    if free and (price or "pricing" in text.lower()):
        return "FREEMIUM", _snippet(text, free)
    if price:
        return "PAID", _snippet(text, price)
    if trial:
        return "PAID", _snippet(text, trial)
    if ent:
        return "ENTERPRISE", _snippet(text, ent)
    if free or oss:
        return "FREE", _snippet(text, free or oss)
    return None, None
