"""Startups + Products vertical (Phase I).

Source: the community-maintained, openly published Y Combinator company dataset
(https://yc-oss.github.io/api — static JSON mirror of YC's public directory). One HTTP call
returns thousands of companies, so no scraping/anti-bot issues for the startup list. Every
record's source.url is the company's YC page. Products are derived from each startup's own
website: pricing is classified from page text and stored with the exact evidence snippet.
"""
from __future__ import annotations

import asyncio
import logging
import re

from bs4 import BeautifulSoup

from ..config import SCHEMA_VERSION
from ..net import Blocked, HttpError
from ..pricing import ENUM, infer_pricing
from ..util import iso

log = logging.getLogger("yc")

YC_URL = "https://yc-oss.github.io/api/companies/all.json"
AI_RE = re.compile(r"\b(ai|a\.i\.|artificial intelligence|machine learning|llms?|generative|gpt|agents?|"
                   r"copilot|computer vision|nlp|deep learning|foundation model)\b", re.I)


def page_text(html: str, cap: int = 60000) -> str:
    soup = BeautifulSoup(html or "", "lxml")
    for t in soup(["script", "style", "noscript", "svg"]):
        t.decompose()
    return soup.get_text(" ", strip=True)[:cap]


def _is_ai(c: dict) -> bool:
    blob = " ".join([c.get("one_liner") or "", " ".join(c.get("tags") or []), c.get("industry") or "",
                     (c.get("long_description") or "")[:300]])
    return bool(AI_RE.search(blob))


async def _inspect_site(fetcher, llm, url: str):
    """Returns (pricingModel, evidence, evidence_url) or None if undeterminable / blocked."""
    base = url.rstrip("/")
    texts: dict[str, str] = {}
    for path in ("", "/pricing", "/plans"):
        try:
            html = await fetcher.get(base + path, check_robots=True, retries=1)
        except (Blocked, HttpError):
            if path == "":
                return None
            continue
        except Exception:
            continue
        texts[base + path] = page_text(html)
        model, ev = infer_pricing(texts[base + path])
        if model:
            return model, ev, base + path
    joined = "\n".join(texts.values())
    if llm is not None and llm.enabled and joined:
        try:
            data, _ = await llm.extract(
                "Classify the product's pricing model from the page text. Allowed values: "
                "FREE, FREEMIUM, PAID, ENTERPRISE.", joined, ["pricingModel"], evidence=("pricingModel",))
            val = (data.get("pricingModel") or "").upper()
            if val in ENUM:
                return val, data.get("pricingModel_evidence"), next(iter(texts))
        except Exception as exc:
            log.debug("LLM pricing fallback failed for %s: %s", url, exc)
    return None


async def collect_yc(fetcher, resolver, llm, now, startup_target=1000, product_target=1000):
    data = await fetcher.get(YC_URL, kind="json")
    companies = [c for c in data if c.get("name") and (c.get("status") or "Active") != "Inactive"]
    ai = [c for c in companies if _is_ai(c)]
    log.info("YC dataset: %d companies, %d AI-related", len(companies), len(ai))
    pool = ai if len(ai) >= startup_target else ai + [c for c in companies if c not in ai]

    startups, seen = [], set()
    for c in pool:
        url = c.get("url") or (f"https://www.ycombinator.com/companies/{c['slug']}" if c.get("slug") else None)
        if not url:
            continue
        canon = resolver.resolve(c["name"], "STARTUP", url).canonical
        if canon.lower() in seen:
            continue
        seen.add(canon.lower())
        team = c.get("team_size")
        startups.append({
            "schemaVersion": SCHEMA_VERSION,
            "recordType": "STARTUP",
            "source": {"name": "Y Combinator", "url": url},
            "content": {
                "entityName": canon,
                "data": {"employeeCount": int(team) if isinstance(team, (int, float)) else None,
                         "website": c.get("website"), "batch": c.get("batch"),
                         "description": c.get("one_liner")},
            },
            "collectedAt": iso(now),
            "_website": c.get("website"),
            "_name": c["name"],
        })
        if len(startups) >= int(startup_target * 1.6):
            break

    # products: inspect each startup's own site (bounded concurrency, robots.txt respected)
    sem = asyncio.Semaphore(30)
    blocked = 0

    async def product(s):
        nonlocal blocked
        if not s["_website"]:
            return None
        async with sem:
            res = await _inspect_site(fetcher, llm, s["_website"])
        if res is None:
            blocked += 1
            return None
        model, ev, ev_url = res
        return {
            "schemaVersion": SCHEMA_VERSION,
            "recordType": "PRODUCT",
            "source": {"name": "Company website", "url": ev_url},
            "content": {"startupName": s["content"]["entityName"], "productName": s["_name"],
                        "pricingModel": model, "pricingEvidence": ev},
            "collectedAt": iso(now),
        }

    products = []
    for i in range(0, len(startups), 150):
        batch = await asyncio.gather(*(product(s) for s in startups[i:i + 150]), return_exceptions=True)
        products += [p for p in batch if isinstance(p, dict)]
        log.info("products: %d verified (%d sites undeterminable/blocked)", len(products), blocked)
        if len(products) >= product_target:
            break

    keep = {id(s) for s in startups[:startup_target + 50]}
    out_startups = [{k: v for k, v in s.items() if not k.startswith("_")} for s in startups if id(s) in keep]
    return out_startups, products[:product_target + 50]
