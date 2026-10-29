"""SQLite storage for markets, predictions and resolutions."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager

from .config import DB_PATH

SCHEMA = """
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


@contextmanager
def connect(path: str | None = None):
    conn = sqlite3.connect(path or DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
