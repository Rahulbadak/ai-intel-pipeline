"""CLI:  python -m src.main <papers|yc|signals|all|export|validate> [options]"""
from __future__ import annotations

import argparse
import asyncio
import logging

from .config import DATA_DIR, GITHUB_TOKEN
from .entities import EntityResolver
from .export import export_workbook, validate
from .llm import LLMEngine
from .net import Fetcher
from .store import SeenStore
from .util import utcnow, write_jsonl

log = logging.getLogger("main")


async def run(args) -> None:
    now = utcnow()
    resolver = EntityResolver()
    async with Fetcher(concurrency=args.concurrency) as fetcher, LLMEngine() as llm:
        if not llm.enabled:
            log.warning("No LLM keys set (GEMINI_API_KEY / GROQ_API_KEY / DEEPSEEK_API_KEY): "
                        "running deterministic paths only.")
        cmd = args.command
        if cmd in ("papers", "all"):
            from .sources.papers import collect_papers
            recs = await collect_papers(fetcher, target=args.papers, token=GITHUB_TOKEN)
            log.info("papers: wrote %d", write_jsonl(DATA_DIR / "papers.jsonl", recs))
        if cmd in ("yc", "all"):
            from .sources.yc import collect_yc
            startups, products = await collect_yc(fetcher, resolver, llm, now, args.startups, args.products)
            log.info("startups: wrote %d", write_jsonl(DATA_DIR / "startups.jsonl", startups))
            log.info("products: wrote %d", write_jsonl(DATA_DIR / "products.jsonl", products))
        if cmd in ("signals", "all"):
            from .sources.signals import collect_jobs, collect_news
            store = SeenStore(DATA_DIR / "seen.db")
            if args.reset_seen:
                store.reset()
            jobs = await collect_jobs(fetcher, store, resolver, now)
            news = await collect_news(fetcher, store, llm if args.enrich_news else None, resolver, now)
            log.info("jobs: wrote %d", write_jsonl(DATA_DIR / "jobs.jsonl", jobs))
            log.info("news: wrote %d", write_jsonl(DATA_DIR / "news.jsonl", news))
        if resolver.log:
            write_jsonl(DATA_DIR / "entity_log.jsonl", resolver.log, append=True)


def main() -> None:
    ap = argparse.ArgumentParser(prog="src.main")
    ap.add_argument("command", choices=["papers", "yc", "signals", "all", "export", "validate"])
    ap.add_argument("--papers", type=int, default=1000, help="target research-paper rows")
    ap.add_argument("--startups", type=int, default=1000)
    ap.add_argument("--products", type=int, default=1000)
    ap.add_argument("--concurrency", type=int, default=40)
    ap.add_argument("--reset-seen", action="store_true", help="clear the seen-store (use for the first/only run)")
    ap.add_argument("--enrich-news", action="store_true", help="LLM-extract (grounded) company mentions for news")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(DATA_DIR / "run.log")])
    if args.command == "export":
        print(export_workbook(DATA_DIR))
    elif args.command == "validate":
        raise SystemExit(0 if validate(DATA_DIR) else 1)
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
