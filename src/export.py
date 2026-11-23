"""JSONL -> 6-tab workbook (+ CSVs) matching the submission sheet, and a validator."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .dates import is_fresh, parse_date
from .util import read_jsonl, utcnow

log = logging.getLogger("export")

TABS = [  # (sheet name, jsonl file, minimum rows)
    ("Startups", "startups.jsonl", 1000),
    ("Products", "products.jsonl", 1000),
    ("Research Papers", "papers.jsonl", 1000),
    ("Jobs", "jobs.jsonl", 0),
    ("News", "news.jsonl", 0),
]


def flat(rec: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in rec.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        elif isinstance(v, list):
            out[key] = "; ".join(map(str, v))
        else:
            out[key] = v
    return out


def _entity_log(data_dir: Path) -> pd.DataFrame:
    rows = read_jsonl(data_dir / "entity_log.jsonl")
    seen, uniq = set(), []
    for r in rows:
        k = (r["raw_name"], r["record_type"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    cols = ["raw_name", "canonical_name", "method", "score", "record_type", "source_url"]
    return pd.DataFrame(uniq, columns=cols)


def export_workbook(data_dir: Path) -> Path:
    out_dir = data_dir / "export"
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx = out_dir / "ai_intelligence_graph.xlsx"
    frames = {name: pd.DataFrame([flat(r) for r in read_jsonl(data_dir / f)]) for name, f, _ in TABS}
    frames["Entity Mapping Log"] = _entity_log(data_dir)
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        for name, df in frames.items():
            df.to_excel(xw, sheet_name=name, index=False)
            df.to_csv(out_dir / f"{name.replace(' ', '_').lower()}.csv", index=False)
            log.info("%-20s %5d rows", name, len(df))
    return xlsx


def validate(data_dir: Path) -> bool:
    """Checks counts vs. the brief, source-URL traceability, uniqueness and 24h freshness."""
    ok, now = True, utcnow()
    for name, fname, minimum in TABS:
        recs = read_jsonl(data_dir / fname)
        problems = []
        if len(recs) < minimum:
            problems.append(f"only {len(recs)} rows (brief asks >= {minimum})")
        no_src = sum(1 for r in recs if not str(r.get("source", {}).get("url", "")).startswith("http"))
        if no_src:
            problems.append(f"{no_src} rows without a valid source URL")
        urls = [r["source"]["url"] for r in recs if r.get("source", {}).get("url")]
        if len(urls) != len(set(urls)) and name in ("Startups", "Research Papers", "Jobs", "News"):
            problems.append(f"{len(urls) - len(set(urls))} duplicate source URLs")
        if name in ("Jobs", "News"):
            key = "date" if name == "Jobs" else "published_date"
            stale = 0
            for r in recs:
                v = r["content"].get(key)
                if v and not is_fresh(parse_date(v, now), now, 24 * 2):  # generous: re-check after hours pass
                    stale += 1
            if stale:
                problems.append(f"{stale} rows older than the freshness window (re-run closer to submission)")
        if name == "Research Papers":
            missing = sum(1 for r in recs if r["content"].get("github_stars") is None)
            if missing:
                problems.append(f"{missing} rows without github_stars")
        status = "OK  " if not problems else "WARN"
        ok &= not problems
        print(f"[{status}] {name:<16} {len(recs):>5} rows" + (" | " + "; ".join(problems) if problems else ""))
    return ok
