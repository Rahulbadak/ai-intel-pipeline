"""High-fidelity signal ingestion (Phase II): AI news (RSS + full-text) and AI jobs (public APIs/RSS).

Freshness rules (never relaxed):
  1. Use the feed/API date if present; else extract from the article page (JSON-LD, meta, <time>).
  2. Date-only values count only if the date is today (UTC). Relative dates are resolved vs. now.
  3. No date anywhere -> heuristic: emit only if the item is NEW since the last run for that
     source (seen-store) AND the source was already baselined; the first run just baselines.
  4. Exactly-once: canonical-URL hash is claimed atomically in the seen-store before emitting.
"""
from __future__ import annotations

import asyncio
import logging
import re

from bs4 import BeautifulSoup

from ..config import FRESH_HOURS, SCHEMA_VERSION
from ..dates import extract_date_from_html, is_fresh, parse_date
from ..net import Blocked, HttpError
from ..util import html_to_text, iso, url_key

log = logging.getLogger("signals")

NEWS_FEEDS = [
    ("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/"),
    ("The Verge AI", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"),
    ("VentureBeat AI", "https://venturebeat.com/category/ai/feed/"),
    ("MIT Technology Review AI", "https://www.technologyreview.com/topic/artificial-intelligence/feed"),
    ("Hugging Face Blog", "https://huggingface.co/blog/feed.xml"),
    ("Ars Technica AI", "https://arstechnica.com/ai/feed/"),
]

AI_JOB_RE = re.compile(r"\b(ai|a\.i\.|ml|machine learning|deep learning|llms?|nlp|generative|genai|"
                       r"computer vision|data scien\w+|prompt|rag|mlops|artificial intelligence)\b", re.I)


# ------------------------------------------------------------------ parsing helpers
def parse_feed(xml_text: str) -> list[dict]:
    soup = BeautifulSoup(xml_text, "xml")
    items = []
    for it in soup.find_all(["item", "entry"]):
        title_tag = it.find("title")
        link = ""
        l = it.find("link")
        if l is not None:
            link = l.get("href") or l.get_text(strip=True)
        if not link:
            g = it.find("guid")
            if g is not None and g.get_text(strip=True).startswith("http"):
                link = g.get_text(strip=True)
        raw_date = None
        for tag in ("pubDate", "published", "dc:date", "date"):
            t = it.find(tag)
            if t is not None and t.get_text(strip=True):
                raw_date = t.get_text(strip=True)
                break
        body = ""
        for tag in ("content:encoded", "content", "description", "summary"):
            t = it.find(tag)
            if t is not None and t.get_text(strip=True):
                body = html_to_text(t.get_text())
                break
        items.append({"title": title_tag.get_text(" ", strip=True) if title_tag else "",
                      "url": link, "date_raw": raw_date, "body": body})
    return [i for i in items if i["url"] and i["title"]]


def extract_article_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "lxml")
    for t in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
        t.decompose()
    node = soup.find("article") or soup.find("main") or soup.body or soup
    paras = [p.get_text(" ", strip=True) for p in node.find_all(["p", "li", "h2", "h3"])]
    return "\n".join(p for p in paras if len(p) > 40)


ROLE_RULES = [
    ("Product", r"product (manager|owner|lead)|\bpm\b|head of product"),
    ("Design", r"designer|\bux\b|\bui\b|creative"),
    ("Sales", r"\bsales\b|account (executive|manager)|business development|\bsdr\b|\bbdr\b"),
    ("Marketing", r"marketing|growth|content|community|seo|brand"),
    ("Operations", r"operations|support|customer success|recruit|talent|people|finance|legal|\bhr\b|admin"),
    ("Data & ML", r"data scien|research scien|machine learning (researcher|scientist)|analyst|\bresearcher\b"),
    ("Engineering", r"engineer|developer|devops|\bsre\b|backend|frontend|full[- ]?stack|software|architect|programmer|\bml\b|\bai\b"),
]


def role_family(title: str) -> str:
    t = (title or "").lower()
    for fam, pat in ROLE_RULES:
        if re.search(pat, t):
            return fam
    return "Other"


def _heuristic_new(store, source: str, url: str) -> bool:
    """Undated item: fresh only if unseen AND the source was already baselined by a previous run."""
    known = store.source_known(source)
    claimed = store.claim(url_key(url), source)
    return known and claimed


# ------------------------------------------------------------------ news
async def _process_news_item(fetcher, store, llm, resolver, src, it, now):
    pd, method, html = None, "feed", None
    if it["date_raw"]:
        pd = parse_date(it["date_raw"], now)
    if pd is None:
        try:
            html = await fetcher.get(it["url"], check_robots=True, retries=1)
            pd = extract_date_from_html(html, now)
            method = pd.method if pd else "none"
        except (Blocked, HttpError):
            pass
    heuristic = False
    if pd is None:
        if not _heuristic_new(store, src, it["url"]):
            return None
        heuristic, method = True, "first_seen_heuristic"
    else:
        if method == "feed":
            method = f"feed:{pd.method}"
        if not is_fresh(pd, now, FRESH_HOURS):
            return None
        if not store.claim(url_key(it["url"]), src):
            return None

    text, text_source = "", "feed"
    try:
        html = html or await fetcher.get(it["url"], check_robots=True, retries=1)
        text = extract_article_text(html)
        text_source = "article_page"
    except (Blocked, HttpError):
        pass
    if len(text) < 400 and len(it["body"]) > len(text):
        text, text_source = it["body"], "feed"
    if not text:
        return None

    content = {"title": it["title"], "text": text[:30000], "date_method": method, "text_source": text_source,
               "published_date": iso(pd.dt) if pd else None}
    if heuristic:
        content["published_date"] = None
        content["first_seen"] = iso(now)
    if llm is not None and llm.enabled:
        try:
            data, _ = await llm.extract("List the AI companies/startups explicitly named in this article.",
                                        text, ["companies"], grounded=("companies",))
            comps = data.get("companies") or []
            content["companies"] = sorted({resolver.resolve(c, "NEWS", it["url"]).canonical for c in comps if c})
        except Exception as exc:
            log.debug("news enrichment skipped: %s", exc)
    return {"schemaVersion": SCHEMA_VERSION, "recordType": "NEWS",
            "source": {"name": src, "url": it["url"]}, "content": content, "collectedAt": iso(now)}


async def collect_news(fetcher, store, llm, resolver, now, max_per_feed=60):
    async def one(src, feed_url):
        try:
            xml_text = await fetcher.get(feed_url)
        except Exception as exc:
            log.error("feed %s unavailable: %s", src, exc)
            return []
        items = parse_feed(xml_text)[:max_per_feed]
        res = await asyncio.gather(*(_process_news_item(fetcher, store, llm, resolver, src, i, now) for i in items),
                                   return_exceptions=True)
        ok = [r for r in res if isinstance(r, dict)]
        log.info("news %-26s %3d items -> %3d fresh", src, len(items), len(ok))
        return ok

    out = await asyncio.gather(*(one(n, u) for n, u in NEWS_FEEDS))
    return [r for sub in out for r in sub]


# ------------------------------------------------------------------ jobs
def _tags(x):
    return " ".join(x) if isinstance(x, list) else (x or "")


async def _remoteok(f):
    data = await f.get("https://remoteok.com/api", kind="json")
    return [{"title": j.get("position"), "company": j.get("company"), "url": j.get("url"),
             "date_raw": j.get("date") or j.get("epoch"), "remote": True,
             "text": _tags(j.get("tags")) + " " + html_to_text(j.get("description", ""))[:600]}
            for j in data if isinstance(j, dict) and j.get("position")]


async def _remotive(f):
    data = await f.get("https://remotive.com/api/remote-jobs", params={"search": "AI"}, kind="json")
    return [{"title": j.get("title"), "company": j.get("company_name"), "url": j.get("url"),
             "date_raw": j.get("publication_date"), "remote": True,
             "text": (j.get("category") or "") + " " + _tags(j.get("tags")) + " " + html_to_text(j.get("description", ""))[:600]}
            for j in data.get("jobs", [])]


async def _arbeitnow(f):
    data = await f.get("https://www.arbeitnow.com/api/job-board-api", kind="json")
    return [{"title": j.get("title"), "company": j.get("company_name"), "url": j.get("url"),
             "date_raw": j.get("created_at"), "remote": bool(j.get("remote")),
             "text": _tags(j.get("tags")) + " " + html_to_text(j.get("description", ""))[:600]}
            for j in data.get("data", [])]


async def _himalayas(f):
    data = await f.get("https://himalayas.app/jobs/api", params={"limit": 100}, kind="json")
    return [{"title": j.get("title"), "company": j.get("companyName"), "url": j.get("applicationLink") or j.get("guid"),
             "date_raw": j.get("pubDate"), "remote": True,
             "text": _tags(j.get("categories")) + " " + html_to_text(j.get("description", ""))[:600]}
            for j in data.get("jobs", [])]


async def _wwr(f):
    xml_text = await f.get("https://weworkremotely.com/categories/remote-programming-jobs.rss")
    out = []
    for it in parse_feed(xml_text):
        company, _, role = it["title"].partition(":")
        out.append({"title": (role or it["title"]).strip(), "company": company.strip() if role else None,
                    "url": it["url"], "date_raw": it["date_raw"], "remote": True, "text": it["body"][:600]})
    return out


JOB_BOARDS = [("RemoteOK", _remoteok), ("Remotive", _remotive), ("Arbeitnow", _arbeitnow),
              ("Himalayas", _himalayas), ("We Work Remotely", _wwr)]


async def collect_jobs(fetcher, store, resolver, now):
    out = []
    for name, fn in JOB_BOARDS:
        try:
            raw = await fn(fetcher)
        except Exception as exc:
            log.error("job board %s failed: %s", name, exc)
            continue
        kept = 0
        for j in raw:
            if not (j.get("title") and j.get("url")):
                continue
            if not AI_JOB_RE.search(f"{j['title']} {j.get('text', '')}"):
                continue
            pd = parse_date(j.get("date_raw"), now)
            if pd is not None:
                if not is_fresh(pd, now, FRESH_HOURS) or not store.claim(url_key(j["url"]), name):
                    continue
                date_val, method = iso(pd.dt), f"api:{pd.method}"
            else:
                if not _heuristic_new(store, name, j["url"]):
                    continue
                date_val, method = None, "first_seen_heuristic"
            company = j.get("company") or ""
            canon = resolver.resolve(company, "JOB", j["url"]).canonical if company else None
            remote = j.get("remote")
            if remote is None:
                remote = bool(re.search(r"\bremote\b", f"{j['title']} {j.get('text', '')}", re.I))
            out.append({"schemaVersion": SCHEMA_VERSION, "recordType": "JOB",
                        "source": {"name": name, "url": j["url"]},
                        "content": {"title": j["title"], "company": canon, "date": date_val,
                                    "is_remote": remote, "role_family": role_family(j["title"]),
                                    "date_method": method},
                        "collectedAt": iso(now)})
            kept += 1
        log.info("jobs %-18s %3d raw -> %3d fresh AI jobs", name, len(raw), kept)
    return out
