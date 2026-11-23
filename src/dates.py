"""Publication-date extraction and normalisation (Phase II).

Handles ISO / RFC-822 strings, epoch seconds/ms, relative dates ("2 hours ago", "3h ago",
"yesterday", "just now") and missing metadata (JSON-LD, meta tags, <time>, visible text).
Naive timestamps are assumed UTC. Date-only values cannot prove a 24h window, so they are
only accepted as fresh when the date is *today* (UTC).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from bs4 import BeautifulSoup
from dateutil import parser as dtparser

_UNITS = {
    "second": 1, "sec": 1, "s": 1,
    "minute": 60, "min": 60, "m": 60,
    "hour": 3600, "hr": 3600, "h": 3600,
    "day": 86400, "d": 86400,
    "week": 604800, "wk": 604800, "w": 604800,
    "month": 2_592_000, "mo": 2_592_000,
    "year": 31_536_000, "yr": 31_536_000, "y": 31_536_000,
}
_UNIT_KEYS = sorted(_UNITS, key=len, reverse=True)
_REL = re.compile(
    r"(?<![\w])(\d+|an?|one)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|wks?|"
    r"months?|mos?|years?|yrs?|[smhdwy])\s+ago\b|(?<![\w])(\d+)([smhdwy])\s*ago\b",
    re.I,
)
_NOW = re.compile(r"\b(just now|moments? ago|a moment ago|right now)\b", re.I)
_TZINFOS = {"EST": -18000, "EDT": -14400, "CST": -21600, "CDT": -18000, "PST": -28800,
            "PDT": -25200, "IST": 19800, "UTC": 0, "GMT": 0, "BST": 3600, "CET": 3600}


@dataclass
class ParsedDate:
    dt: datetime            # timezone-aware, UTC
    date_only: bool = False
    method: str = "absolute"


def _unit_seconds(unit: str) -> int:
    u = unit.lower()
    for key in _UNIT_KEYS:
        if u.startswith(key):
            return _UNITS[key]
    raise ValueError(unit)


def _qty(token: str) -> int:
    return 1 if token.lower() in ("a", "an", "one") else int(token)


def parse_date(value, now: datetime | None = None) -> ParsedDate | None:
    now = now or datetime.now(timezone.utc)
    if value is None:
        return None
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip().isdigit()):
        ts = float(value)
        if ts > 1e11:  # milliseconds
            ts /= 1000
        if ts < 1e8:
            return None
        return ParsedDate(datetime.fromtimestamp(ts, timezone.utc), method="epoch")
    s = str(value).strip()
    if not s:
        return None
    if _NOW.search(s):
        return ParsedDate(now, method="relative")
    m = _REL.search(s)
    if m:
        qty, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        return ParsedDate(now - timedelta(seconds=_qty(qty) * _unit_seconds(unit)), method="relative")
    low = s.lower()
    if re.search(r"\byesterday\b", low):
        return ParsedDate(now - timedelta(days=1), date_only=True, method="relative")
    if re.fullmatch(r"(posted\s+)?today", low):
        return ParsedDate(now, date_only=True, method="relative")
    try:
        dt = dtparser.parse(s, tzinfos=_TZINFOS)
    except (ValueError, OverflowError, TypeError):
        return None
    date_only = not re.search(r"\d{1,2}:\d{2}", s) and "T" not in s
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return ParsedDate(dt.astimezone(timezone.utc), date_only=date_only, method="absolute")


def is_fresh(pd: ParsedDate | None, now: datetime | None = None, hours: int = 24) -> bool:
    if pd is None:
        return False
    now = now or datetime.now(timezone.utc)
    if pd.date_only:
        return pd.dt.date() == now.date()
    delta = now - pd.dt
    return timedelta(minutes=-5) <= delta <= timedelta(hours=hours)  # 5 min clock-skew tolerance


_META_KEYS = [
    ("property", "article:published_time"), ("property", "og:article:published_time"),
    ("name", "article:published_time"), ("name", "pubdate"), ("name", "publishdate"),
    ("name", "publish-date"), ("name", "date"), ("name", "dc.date.issued"), ("name", "dc.date"),
    ("name", "parsely-pub-date"), ("name", "sailthru.date"), ("itemprop", "datePublished"),
]


def _walk_jsonld(node):
    if isinstance(node, dict):
        if "datePublished" in node:
            yield node["datePublished"]
        for v in node.values():
            yield from _walk_jsonld(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk_jsonld(v)


def extract_date_from_html(html: str, now: datetime | None = None) -> ParsedDate | None:
    """Priority: JSON-LD datePublished > meta tags > <time datetime> > visible 'x ago' text.
    dateModified / article:modified_time are intentionally ignored (not publication time)."""
    now = now or datetime.now(timezone.utc)
    soup = BeautifulSoup(html or "", "lxml")
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except (ValueError, TypeError):
            continue
        for val in _walk_jsonld(data):
            pd = parse_date(val, now)
            if pd:
                pd.method = "json-ld"
                return pd
    for attr, key in _META_KEYS:
        tag = soup.find("meta", attrs={attr: re.compile(f"^{re.escape(key)}$", re.I)})
        if tag and tag.get("content"):
            pd = parse_date(tag["content"], now)
            if pd:
                pd.method = f"meta:{key}"
                return pd
    for t in soup.find_all("time"):
        val = t.get("datetime") or t.get_text(strip=True)
        pd = parse_date(val, now)
        if pd:
            pd.method = "time-tag"
            return pd
    text = soup.get_text(" ", strip=True)[:6000]
    m = re.search(r"(?:published|posted)[^.\n]{0,40}?((?:\d+|an?)\s*\w+\s+ago)", text, re.I)
    if m:
        pd = parse_date(m.group(1), now)
        if pd:
            pd.method = "visible-relative"
            return pd
    return None
