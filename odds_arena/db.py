"""Storage for markets, predictions and resolutions.

Uses Postgres when DATABASE_URL (or POSTGRES_URL) is set, e.g. Supabase via Vercel;
otherwise a local SQLite file. Queries use `?` placeholders and SQL both databases accept.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .config import DATABASE_URL, DB_PATH

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    end_date TEXT,
    outcome INTEGER,            -- 1 = first outcome won, 0 = lost, NULL = unresolved
    resolved_at TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL REFERENCES markets(id),
    model TEXT NOT NULL,
    probability REAL NOT NULL,  -- predicted P(first outcome)
    market_price REAL,          -- market price at prediction time
    rationale TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (market_id, model)
);
"""

POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    end_date TEXT,
    outcome INTEGER,
    resolved_at TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    market_id TEXT NOT NULL REFERENCES markets(id),
    model TEXT NOT NULL,
    probability DOUBLE PRECISION NOT NULL,
    market_price DOUBLE PRECISION,
    rationale TEXT,
    created_at TEXT NOT NULL DEFAULT to_char(now() AT TIME ZONE 'utc', 'YYYY-MM-DD HH24:MI:SS'),
    UNIQUE (market_id, model)
);
"""


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class _Postgres:
    """Wraps a psycopg connection so callers can keep sqlite-style `?` placeholders."""

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql: str, params=()):
        return self._conn.execute(sql.replace("?", "%s"), params)

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()


_pg_schema_ready = False

# Vercel's Supabase integration appends params libpq rejects (e.g. `supa=base-pooler.x`).
_NON_LIBPQ_PARAMS = {"supa", "pgbouncer"}


def _clean_url(url: str) -> str:
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query) if k not in _NON_LIBPQ_PARAMS]
    return urlunsplit(parts._replace(query=urlencode(query)))


def _open():
    global _pg_schema_ready
    if DATABASE_URL:
        import psycopg
        from psycopg.rows import dict_row

        # prepare_threshold=None: Supabase's transaction pooler (port 6543) can't keep prepared statements.
        conn = _Postgres(psycopg.connect(_clean_url(DATABASE_URL), row_factory=dict_row, prepare_threshold=None))
        if not _pg_schema_ready:
            conn._conn.execute(POSTGRES_SCHEMA)
            conn.commit()
            _pg_schema_ready = True
        return conn
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SQLITE_SCHEMA)
    return conn


@contextmanager
def connect():
    conn = _open()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
