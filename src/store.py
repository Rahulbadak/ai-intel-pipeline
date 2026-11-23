"""Seen-store used for freshness tracking and exactly-once processing.

Single-node: SQLite with a PRIMARY KEY on the canonical-URL hash, so `claim()` is an atomic
insert-if-absent. Distributed: swap this class for Redis `SET key NX` or a Postgres
`INSERT ... ON CONFLICT DO NOTHING` (same interface) — see architecture.pdf.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class SeenStore:
    def __init__(self, path: str | Path = "data/seen.db"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS seen (key TEXT PRIMARY KEY, source TEXT, first_seen TEXT)"
        )
        self.db.commit()

    def claim(self, key: str, source: str) -> bool:
        """True only for the first caller ever to claim this key."""
        cur = self.db.execute(
            "INSERT OR IGNORE INTO seen (key, source, first_seen) VALUES (?, ?, ?)",
            (key, source, datetime.now(timezone.utc).isoformat()),
        )
        self.db.commit()
        return cur.rowcount == 1

    def source_known(self, source: str) -> bool:
        return self.db.execute("SELECT 1 FROM seen WHERE source=? LIMIT 1", (source,)).fetchone() is not None

    def reset(self) -> None:
        self.db.execute("DELETE FROM seen")
        self.db.commit()
