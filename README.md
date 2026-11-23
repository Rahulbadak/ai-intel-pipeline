# AI Intelligence Graph: Ingestion Pipeline (Demo Task)

Async, fault-tolerant pipeline that collects startups, products, research papers (with live GitHub
stars), 24h-fresh AI news and jobs, resolves entity names to canonical forms, and exports the
six-tab sheet required by the brief. See `architecture.pdf` for the production design.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10+
pip install -r requirements.txt
cp .env.example .env && export $(grep -v '^#' .env | xargs)   # or export the vars manually

pytest -q                                   # 24 offline tests (no network needed)
python -m src.main all --reset-seen         # full run (~20-30 min)
python -m src.main validate                 # row counts, source URLs, freshness, duplicates
python -m src.main export                   # data/export/ai_intelligence_graph.xlsx + CSVs
```

Individual stages: `papers`, `yc` (startups + products), `signals` (jobs + news). Useful flags:
`--papers/--startups/--products N`, `--enrich-news` (LLM company mentions), `--reset-seen`, `-v`.

**Keys:** `GITHUB_TOKEN` is effectively required for papers (60 req/h unauthenticated vs 5,000/h).
`GEMINI_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY` are optional; tiers without a key are skipped.
**Run signals close to submission time**: news/jobs must be published within the last 24 hours.

## Layout

| Path | Purpose |
|---|---|
| `src/net.py` | aiohttp client: global + per-host concurrency, robots.txt, retries, exp. backoff + full jitter, `Retry-After` |
| `src/sources/papers.py` | arXiv API discovery -> GitHub API star verification (only verified repos are emitted) |
| `src/sources/yc.py` | Startups from the open YC directory dataset; products from each startup's own site |
| `src/sources/signals.py` | 6 news RSS feeds (+ full-text) and 5 job boards; freshness logic |
| `src/dates.py` | Date normalisation: ISO/RFC822/epoch/relative ("2 hours ago"), JSON-LD/meta/`<time>` extraction |
| `src/store.py` | Seen-store: atomic claim => never process an item twice (SQLite; Redis/Postgres in prod) |
| `src/llm.py` | Gemini -> Groq -> DeepSeek fallback, 429 backoff, 413 re-chunking, hallucination guards |
| `src/entities.py` | Deterministic entity resolution + mapping log, seeded with 50 AI startups |
| `src/pricing.py` | Rule-based pricing classifier that returns the supporting page snippet |
| `src/export.py` | JSONL -> 6-tab workbook + validator |
| `tests/` | Dates, entities, chunking, fallback/429/413 behaviour, grounding, parsers, store |

## Design decisions

**No hallucinated data.** Every record carries `source.url`. Facts come from APIs or deterministic
parsers wherever possible. Where an LLM is used (pricing fallback, news company mentions) its
output is accepted only if the extracted strings appear verbatim in the source text or it supplies
an exact supporting quote; otherwise the value is rejected (`Ungrounded`) and the record dropped.
Pricing rows include `pricingEvidence`, the page snippet that justified the label.

**Freshness (24h).** Date priority: feed/API date -> JSON-LD `datePublished` -> meta tags ->
`<time>` -> visible "x ago" text. `dateModified` is never used. Relative dates resolve against run
time; naive timestamps are assumed UTC. Date-only values are accepted only if the date is today
(they cannot prove a 24h window). Undated items use a heuristic: emitted only if new since the last
run of that source (the first run just baselines the source), tagged `first_seen_heuristic`.

**LLM resilience.** 429 -> exponential backoff with full jitter, honouring `Retry-After`; a long
`Retry-After` puts the tier on cooldown and falls through to the next tier immediately.
413 (or a context-length 400) -> the input is re-chunked at half size by a semantic truncator (keeps the
lead + the densest de-duplicated paragraphs) and retried on the same tier.

**Entity resolution.** Normalise (case, punctuation, `&`, legal suffixes) -> exact/alias match on a
compact key -> dynamic de-dupe of unseen names -> guarded fuzzy match (>= 0.93, same first letter,
length within 2: "Coherent" does not collapse into "Cohere"). `OpenAI`, `OpenAI, Inc.`, `Open AI`
all resolve to `OpenAI`. Every decision is logged (Entity Mapping Log tab).

**Anti-bot stance.** The pipeline does not bypass Cloudflare/Datadome. It prefers official
APIs/RSS, respects robots.txt, identifies itself, rate-limits per host, and treats 401/403/451 as
"blocked, skip". The strategy for protected/JS-heavy domains is documented in `architecture.pdf`.

## Data sources and honest limitations

* **Startups/Products:** `yc-oss.github.io/api` (open mirror of YC's public directory). Products are
  the startups' own sites; sites that are JS-only or block bots yield no pricing evidence and are
  skipped (counted in logs) rather than guessed. A Playwright-async fallback is designed but not
  enabled in this trial build.
* **Papers:** arXiv API (`abs:github`) + GitHub REST API. Papers with Code was retired, so it is not used.
* **Jobs/News:** RemoteOK, Remotive, Arbeitnow, Himalayas, We Work Remotely; TechCrunch, The Verge,
  VentureBeat, MIT Tech Review, Hugging Face, Ars Technica. Feeds/APIs change; `validate` and the run
  log show which ones returned data on your run. Job role family is rule-based (no LLM).
* Row counts depend on live availability at run time. Anything under the 1,000-row minimums is reported
  by `validate`; nothing is padded or synthesised.
