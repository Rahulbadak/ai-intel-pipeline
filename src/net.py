"""Async HTTP layer: bounded concurrency, per-host limits, robots.txt, retries with
exponential backoff + full jitter, and Retry-After support.

Deliberately does NOT try to defeat anti-bot systems: a 401/403/451 is surfaced as
`Blocked` and the caller skips the URL (see docs in architecture.pdf, Phase V).
"""
from __future__ import annotations

import asyncio
import logging
import random
from collections import defaultdict
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

try:
    import aiohttp
except ImportError:  # allows importing pure-logic modules without aiohttp installed
    aiohttp = None

from .config import USER_AGENT

log = logging.getLogger("net")


class Blocked(Exception):
    """Site refused us (robots.txt, 401/403/451). We skip instead of evading."""


class HttpError(Exception):
    def __init__(self, status: int, url: str):
        super().__init__(f"HTTP {status} for {url}")
        self.status = status
        self.url = url


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 60.0, retry_after: float | None = None) -> float:
    """Exponential backoff with FULL jitter; honours Retry-After when the server sends it."""
    if retry_after is not None:
        return min(cap, max(0.0, retry_after)) + random.uniform(0, 0.5)
    return random.uniform(0, min(cap, base * (2 ** attempt)))


def parse_retry_after(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class Fetcher:
    def __init__(self, concurrency: int = 40, per_host: int = 4, timeout: int = 30):
        self.timeout = timeout
        self._sem = asyncio.Semaphore(concurrency)
        self._per_host = per_host
        self._host_sems: dict[str, asyncio.Semaphore] = defaultdict(lambda: asyncio.Semaphore(per_host))
        self._robots: dict[str, RobotFileParser] = {}
        self.session = None

    async def __aenter__(self):
        if aiohttp is None:
            raise RuntimeError("aiohttp is not installed: pip install -r requirements.txt")
        self.session = aiohttp.ClientSession(
            headers={"User-Agent": USER_AGENT, "Accept-Language": "en"},
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            connector=aiohttp.TCPConnector(limit=0, ttl_dns_cache=300),
        )
        return self

    async def __aexit__(self, *exc):
        await self.session.close()

    async def request(self, url, *, params=None, headers=None, kind="text", retries=3):
        """Returns (status, lowercase-headers, body). Retries 429/5xx/network errors only."""
        host = urlparse(url).netloc
        status, hdrs, body = 0, {}, None
        for attempt in range(retries + 1):
            delay = None
            async with self._sem, self._host_sems[host]:
                try:
                    async with self.session.get(url, params=params, headers=headers) as r:
                        status = r.status
                        hdrs = {k.lower(): v for k, v in r.headers.items()}
                        if status == 200:
                            if kind == "json":
                                try:
                                    body = await r.json(content_type=None)
                                except Exception:
                                    body = None
                            elif kind == "bytes":
                                body = await r.read()
                            else:
                                body = await r.text(errors="replace")
                            return status, hdrs, body
                        body = await r.text(errors="replace")
                        if status == 429 or status >= 500:
                            ra = parse_retry_after(hdrs.get("retry-after")) if status == 429 else None
                            delay = backoff_delay(attempt, retry_after=ra)
                            log.warning("HTTP %s on %s (attempt %d), sleeping %.1fs", status, url, attempt + 1, delay)
                        else:
                            return status, hdrs, body
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    status = 0
                    delay = backoff_delay(attempt)
                    log.warning("Network error on %s: %s (attempt %d)", url, exc, attempt + 1)
            if attempt < retries and delay is not None:
                await asyncio.sleep(delay)  # sleep OUTSIDE the semaphores
        return status, hdrs, body

    async def get(self, url, *, params=None, headers=None, kind="text", retries=3, check_robots=False):
        if check_robots and not await self.allowed(url):
            raise Blocked(f"robots.txt disallows {url}")
        status, _, body = await self.request(url, params=params, headers=headers, kind=kind, retries=retries)
        if status == 200 and body is not None:
            return body
        if status in (401, 403, 451):
            raise Blocked(f"HTTP {status} for {url}")
        raise HttpError(status, url)

    async def allowed(self, url: str) -> bool:
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        rp = self._robots.get(base)
        if rp is None:
            rp = RobotFileParser()
            try:
                status, _, body = await self.request(base + "/robots.txt", retries=1)
                rp.parse(body.splitlines() if status == 200 and body else [])
            except Exception:
                rp.parse([])
            rp.modified()  # mark as checked, otherwise can_fetch() returns False
            self._robots[base] = rp
        return rp.can_fetch(USER_AGENT, url)
