"""Research-paper vertical (Phase I): arXiv API for discovery + GitHub API for live star counts.

Why this design: arXiv's API and GitHub's REST API are official, stable and need no anti-bot
evasion. Papers whose abstract/comment field links a GitHub repo are discovered via the query
`abs:github`; stars are then verified live. A paper is only emitted if its repo resolves
(HTTP 200) so github_url / github_stars are never guessed.
"""
from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET

from ..config import SCHEMA_VERSION
from ..util import iso, utcnow

log = logging.getLogger("papers")

ARXIV_API = "https://export.arxiv.org/api/query"
QUERY = "abs:github AND (cat:cs.LG OR cat:cs.CL OR cat:cs.CV OR cat:cs.AI)"
NS = {"a": "http://www.w3.org/2005/Atom", "ax": "http://arxiv.org/schemas/atom"}
_GH = re.compile(r"github\.com/([A-Za-z0-9][A-Za-z0-9-]{0,38})/([A-Za-z0-9_.-]+)", re.I)
_NOT_USERS = {"sponsors", "orgs", "features", "topics", "settings", "marketplace", "about", "apps"}


def find_github(*texts: str) -> tuple[str, str] | None:
    for t in texts:
        for m in _GH.finditer(t or ""):
            owner, repo = m.group(1), m.group(2).rstrip(".,;:)]}>'\"")
            repo = re.sub(r"\.git$", "", repo)
            if owner.lower() in _NOT_USERS or not repo:
                continue
            return owner, repo
    return None


def parse_arxiv(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    out = []
    for e in root.findall("a:entry", NS):
        abs_url = (e.findtext("a:id", "", NS) or "").strip()
        if not abs_url or "/api/errors" in abs_url:
            continue
        title = re.sub(r"\s+", " ", e.findtext("a:title", "", NS) or "").strip()
        summary = e.findtext("a:summary", "", NS) or ""
        comment = e.findtext("ax:comment", "", NS) or ""
        authors = [(a.findtext("a:name", "", NS) or "").strip() for a in e.findall("a:author", NS)]
        out.append({
            "arxiv_id": re.sub(r"v\d+$", "", abs_url.rsplit("/abs/", 1)[-1]),
            "abs_url": abs_url,
            "title": title,
            "authors": [a for a in authors if a],
            "published": (e.findtext("a:published", "", NS) or "").strip(),
            "github": find_github(comment, summary),
        })
    return out


_fail_count = 0


def _note_failure(owner, repo, status, body):
    global _fail_count
    _fail_count += 1
    if _fail_count <= 5:  # only the first few, to keep the log readable
        msg = body if isinstance(body, str) else ""
        log.warning("GitHub lookup failed for %s/%s: HTTP %s %s", owner, repo, status, msg[:150])


async def preflight_github(fetcher, token: str) -> None:
    """Fail fast (before the 2-minute arXiv crawl) if the token is bad. /rate_limit is free."""
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    status, _, body = await fetcher.request("https://api.github.com/rate_limit", headers=headers, kind="json")
    if status == 401:
        raise RuntimeError("GitHub rejected GITHUB_TOKEN (401 Bad credentials). Create a new classic token at "
                           "github.com/settings/tokens (no scopes needed) and paste it without spaces/quotes.")
    if status == 200 and isinstance(body, dict):
        core = body.get("resources", {}).get("core", {})
        log.info("GitHub rate limit: %s/%s remaining", core.get("remaining"), core.get("limit"))
        if core.get("remaining", 1) == 0:
            raise RuntimeError("GitHub rate limit already exhausted; wait for reset or use a token.")
    else:
        log.warning("GitHub /rate_limit check returned HTTP %s: %s", status, str(body)[:150])


async def github_stars(fetcher, owner: str, repo: str, token: str = ""):
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for _ in range(2):
        status, h, body = await fetcher.request(f"https://api.github.com/repos/{owner}/{repo}",
                                                headers=headers, kind="json")
        if status == 200 and isinstance(body, dict):
            return body.get("stargazers_count"), body.get("html_url")
        if status in (403, 429) and h.get("x-ratelimit-remaining") == "0":
            import time
            wait = max(1, int(h.get("x-ratelimit-reset", "0")) - int(time.time())) + 2
            if wait > 900:
                log.error("GitHub rate limit exhausted (resets in %ds). Set GITHUB_TOKEN.", wait)
                return None
            log.warning("GitHub rate limit hit, sleeping %ds", wait)
            await asyncio.sleep(wait)
            continue
        _note_failure(owner, repo, status, body)
        return None
    return None


async def collect_papers(fetcher, target: int = 1000, token: str = "", max_pages: int = 40) -> list[dict]:
    if not token:
        log.warning("GITHUB_TOKEN not set: GitHub allows only 60 requests/hour unauthenticated. "
                    "Set it to reach %d papers.", target)
    token = (token or "").strip()
    await preflight_github(fetcher, token)
    candidates: dict[str, dict] = {}
    for page in range(max_pages):
        params = {"search_query": QUERY, "start": page * 100, "max_results": 100,
                  "sortBy": "submittedDate", "sortOrder": "descending"}
        try:
            xml_text = await fetcher.get(ARXIV_API, params=params)
        except Exception as exc:
            log.error("arXiv page %d failed: %s", page, exc)
            break
        entries = parse_arxiv(xml_text)
        if not entries:
            break
        for e in entries:
            if e["github"] and e["arxiv_id"] not in candidates:
                candidates[e["arxiv_id"]] = e
        log.info("arXiv page %d: %d candidates with GitHub links", page + 1, len(candidates))
        if len(candidates) >= int(target * 1.4):
            break
        await asyncio.sleep(3.1)  # arXiv API etiquette: one request per 3 seconds

    sem = asyncio.Semaphore(10)
    now = utcnow()

    async def verify(e):
        async with sem:
            owner, repo = e["github"]
            res = await github_stars(fetcher, owner, repo, token)
        if not res or res[0] is None:
            return None
        stars, html_url = res
        return {
            "schemaVersion": SCHEMA_VERSION,
            "recordType": "RESEARCH_PAPER",
            "source": {"name": "arXiv", "url": e["abs_url"]},
            "content": {
                "title": e["title"],
                "authors": e["authors"],
                "paper_url": e["abs_url"],
                "github_url": html_url or f"https://github.com/{owner}/{repo}",
                "github_stars": int(stars),
                "github_stars_checked_at": iso(utcnow()),
                "published_date": e["published"],
            },
            "collectedAt": iso(now),
        }

    results, ordered = [], list(candidates.values())
    for i in range(0, len(ordered), 100):
        batch = await asyncio.gather(*(verify(e) for e in ordered[i:i + 100]), return_exceptions=True)
        for r in batch:
            if isinstance(r, Exception) and _fail_count < 5:
                _note_failure("?", "?", "exception", repr(r))
        results += [r for r in batch if isinstance(r, dict)]
        log.info("verified %d/%d repos", len(results), min(i + 100, len(ordered)))
        if len(results) >= target:
            break
    return results[:target + 100]
