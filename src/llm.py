"""Multi-tier LLM extraction engine (Phase III).

* Fallback chain: Gemini Flash -> Groq Llama 3 -> DeepSeek (tiers without a key are skipped).
* 429: exponential backoff with full jitter; honours Retry-After; if the server asks for a
  long wait, the tier is put on cooldown and we fall through to the next tier immediately.
* 413 / context-length 400: payload is re-chunked at half size (semantic truncation keeps the
  lead paragraphs and the densest remaining ones) and retried on the same tier.
* Anti-hallucination: extracted strings must appear in the source text; enum values need an
  exact supporting quote. Anything ungrounded is rejected (record dropped, never invented).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import Counter

try:
    import aiohttp
except ImportError:
    aiohttp = None

from .config import TIERS
from .net import backoff_delay, parse_retry_after

log = logging.getLogger("llm")


class LLMError(Exception): ...
class AllTiersFailed(LLMError): ...
class Ungrounded(LLMError): ...


# ---------------------------------------------------------------- chunking / truncation
def _score(par: str) -> float:
    words = par.split()
    n = len(words)
    if n < 4:
        return 0.0
    uniq = len({w.lower() for w in words}) / n
    nums = sum(any(c.isdigit() for c in w) for w in words) / n
    caps = sum(w[:1].isupper() for w in words) / n
    return uniq + 2 * nums + caps + min(n, 60) / 120


def smart_truncate(text: str, max_chars: int) -> str:
    """Shrink text to <= max_chars while keeping the lead and the most information-dense
    paragraphs, de-duplicated and in original order."""
    text = re.sub(r"[ \t]+", " ", text or "")
    if len(text) <= max_chars:
        return text
    paras, seen = [], set()
    for i, p in enumerate(x.strip() for x in text.split("\n")):
        if not p or p.lower() in seen:
            continue
        seen.add(p.lower())
        paras.append((i, p))
    head, rest = paras[:2], paras[2:]
    keep = list(head)
    budget = max_chars - sum(len(p) + 1 for _, p in head)
    for i, p in sorted(rest, key=lambda t: _score(t[1]), reverse=True):
        if len(p) + 1 <= budget:
            keep.append((i, p))
            budget -= len(p) + 1
    keep.sort(key=lambda t: t[0])
    return "\n".join(p for _, p in keep)[:max_chars]


def est_tokens(text: str) -> int:
    return len(text) // 4 + 1


# ---------------------------------------------------------------- grounding
def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", str(s).lower()).strip()


def is_grounded(value, source_text: str) -> bool:
    v = _norm(value)
    return bool(v) and v in _norm(source_text)


def _is_context_error(body) -> bool:
    s = json.dumps(body).lower() if not isinstance(body, str) else body.lower()
    return any(k in s for k in ("context length", "context_length", "maximum context", "too large",
                                "too many tokens", "reduce the length", "payload too large", "request too large"))


# ---------------------------------------------------------------- engine
class LLMEngine:
    def __init__(self, tiers=None, transport=None, max_retries=5, max_wait=30.0, sleep=asyncio.sleep, concurrency=8):
        self.all_tiers = tiers if tiers is not None else TIERS
        self.tiers = [t for t in self.all_tiers if self._key(t)]
        self.transport = transport
        self.max_retries = max_retries
        self.max_wait = max_wait
        self._sleep = sleep
        self._sem = asyncio.Semaphore(concurrency)
        self._tier_sems = {t["name"]: asyncio.Semaphore(t.get("concurrency", 4)) for t in self.all_tiers}
        self._cool: dict[str, float] = {}
        self.stats: Counter = Counter()
        self._session = None

    @staticmethod
    def _key(tier) -> str | None:
        return tier.get("api_key") or os.getenv(tier.get("key_env", ""), "") or None

    @property
    def enabled(self) -> bool:
        return bool(self.tiers)

    async def __aenter__(self):
        if self.transport is None and self.tiers:
            if aiohttp is None:
                raise RuntimeError("aiohttp is not installed: pip install -r requirements.txt")
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=90))
            self.transport = self._http_transport
        return self

    async def __aexit__(self, *exc):
        if self._session:
            await self._session.close()
        if self.stats:
            log.info("LLM stats: %s", dict(self.stats))

    async def _http_transport(self, tier, payload):
        headers = {"Authorization": f"Bearer {self._key(tier)}", "Content-Type": "application/json"}
        try:
            async with self._session.post(tier["url"], json=payload, headers=headers) as r:
                try:
                    body = await r.json(content_type=None)
                except Exception:
                    body = {"raw": await r.text()}
                return r.status, {k.lower(): v for k, v in r.headers.items()}, body
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            return 0, {}, {"error": str(exc)}

    @staticmethod
    def _payload(tier, system, user_text):
        return {
            "model": tier["model"],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_text}],
            "temperature": 0,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
        }

    @staticmethod
    def _parse(body) -> dict:
        content = body["choices"][0]["message"]["content"].strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("model did not return a JSON object")
        return data

    async def complete_json(self, system: str, text: str, min_chars: int = 1500):
        """Returns (parsed_json, tier_name). Raises AllTiersFailed."""
        if not self.tiers:
            raise AllTiersFailed("no LLM API keys configured")
        errors = []
        async with self._sem:
            for tier in self.tiers:
                name = tier["name"]
                if time.monotonic() < self._cool.get(name, 0):
                    errors.append(f"{name}: cooling down")
                    continue
                if errors:
                    self.stats["fallbacks"] += 1
                budget, attempt, shrinks = tier["max_chars"], 0, 0
                while attempt <= self.max_retries:
                    payload = self._payload(tier, system, smart_truncate(text, budget))
                    async with self._tier_sems[name]:
                        status, headers, body = await self.transport(tier, payload)
                    if status == 200:
                        try:
                            data = self._parse(body)
                            self.stats[f"ok:{name}"] += 1
                            return data, name
                        except (ValueError, KeyError, IndexError, TypeError) as exc:
                            self.stats["bad_json"] += 1
                            attempt += 1
                            log.warning("%s returned unparseable output (%s)", name, exc)
                            continue
                    if status == 429:
                        self.stats["429"] += 1
                        ra = parse_retry_after(headers.get("retry-after"))
                        if ra is not None and ra > self.max_wait:
                            self._cool[name] = time.monotonic() + ra
                            errors.append(f"{name}: rate-limited {ra:.0f}s")
                            break
                        await self._sleep(backoff_delay(attempt, retry_after=ra))
                        attempt += 1
                        continue
                    if status == 413 or (status == 400 and _is_context_error(body)):
                        self.stats["413"] += 1
                        shrinks += 1
                        if budget <= min_chars or shrinks > 4:
                            errors.append(f"{name}: payload too large")
                            break
                        budget = max(min_chars, budget // 2)
                        log.info("%s: payload too large, re-chunking to %d chars", name, budget)
                        continue
                    if status in (0, 500, 502, 503, 504):
                        await self._sleep(backoff_delay(attempt))
                        attempt += 1
                        continue
                    if status in (401, 403):
                        self._cool[name] = time.monotonic() + 3600
                    errors.append(f"{name}: HTTP {status}")
                    break
                else:
                    errors.append(f"{name}: retries exhausted")
        raise AllTiersFailed("; ".join(errors))

    async def extract(self, instruction: str, text: str, fields: list[str],
                      grounded=(), evidence=()) -> tuple[dict, str]:
        """Structured extraction with hallucination guards.
        grounded: fields whose value(s) must appear verbatim in `text`.
        evidence: fields that must come with `<field>_evidence`, an exact quote from `text`."""
        ev_note = ""
        if evidence:
            ev_note = " For each of [%s] also return '<field>_evidence' with an EXACT quote copied from the text." % ", ".join(evidence)
        system = (
            f"{instruction}\nReturn ONLY a JSON object with keys: {', '.join(fields)}."
            " Use null when the text does not explicitly state a value. Never guess, infer or add"
            f" information that is not in the text.{ev_note}"
        )
        data, tier = await self.complete_json(system, text)
        for f in grounded:
            v = data.get(f)
            vals = v if isinstance(v, list) else [v]
            if any(x is not None and not is_grounded(x, text) for x in vals):
                self.stats["ungrounded"] += 1
                raise Ungrounded(f"field {f!r} not found in source text")
        for f in evidence:
            if data.get(f) is not None:
                quote = data.get(f"{f}_evidence")
                if not quote or not is_grounded(quote, text):
                    self.stats["ungrounded"] += 1
                    raise Ungrounded(f"field {f!r} lacks a verifiable quote")
        return data, tier
