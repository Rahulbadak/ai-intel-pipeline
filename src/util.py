from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from bs4 import BeautifulSoup

_TRACKING = re.compile(r"^(utm_|fbclid|gclid|mc_|ref$|ref_src)", re.I)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_url(url: str) -> str:
    p = urlsplit(url.strip())
    query = [(k, v) for k, v in parse_qsl(p.query) if not _TRACKING.match(k)]
    host = p.netloc.lower().removeprefix("www.")
    path = p.path.rstrip("/") or "/"
    return urlunsplit(((p.scheme or "https").lower(), host, path, urlencode(query), ""))


def url_key(url: str) -> str:
    return hashlib.sha1(canonical_url(url).encode()).hexdigest()


def html_to_text(html: str) -> str:
    return BeautifulSoup(html or "", "lxml").get_text(" ", strip=True)


def write_jsonl(path: Path, records, append: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a" if append else "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
