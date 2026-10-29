"""predict / resolve jobs, shared by the CLI and the Vercel cron endpoints.

Each job stops starting new work once `time_budget` seconds have passed, so it fits inside a
serverless function's time limit. Work is committed as it goes; the next run picks up the rest.
"""
from __future__ import annotations

import time
from typing import Callable

from . import db, polymarket
from .config import ENABLED_MODELS
from .models import load_models

Log = Callable[[str], None]


def run_predict(limit: int = 20, model_names: list[str] | None = None,
                time_budget: float | None = None, log: Log = print) -> dict:
    deadline = time.monotonic() + time_budget if time_budget else None
    models = load_models(model_names or ENABLED_MODELS)
    markets = polymarket.fetch_open_markets(limit=limit)
    log(f"Fetched {len(markets)} open binary markets")
    made = failed = 0
    timed_out = False
    with db.connect() as conn:
        for m in markets:
            conn.execute(
                "INSERT INTO markets (id, question, end_date) VALUES (?, ?, ?) ON CONFLICT (id) DO NOTHING",
                (m.id, m.question, m.end_date),
            )
            for model in models:
                if deadline and time.monotonic() > deadline:
                    timed_out = True
                    break
                if conn.execute("SELECT 1 FROM predictions WHERE market_id=? AND model=?", (m.id, model.name)).fetchone():
                    continue
                try:
                    pred = model.predict(m)
                except Exception as e:  # one failing model shouldn't stop the others
                    failed += 1
                    log(f"  [{model.name}] failed on {m.id}: {e}")
                    continue
                conn.execute(
                    "INSERT INTO predictions (market_id, model, probability, market_price, rationale) VALUES (?, ?, ?, ?, ?)",
                    (m.id, model.name, pred.probability, m.yes_price, pred.rationale),
                )
                conn.commit()
                made += 1
                log(f"  [{model.name}] {pred.probability:.2f}  {m.question[:70]}")
            if timed_out:
                log("Time budget reached; the next run continues from here")
                break
    return {"markets": len(markets), "predictions": made, "failed": failed, "timed_out": timed_out}


def run_resolve(time_budget: float | None = None, log: Log = print) -> dict:
    deadline = time.monotonic() + time_budget if time_budget else None
    checked = resolved = failed = 0
    timed_out = False
    with db.connect() as conn:
        # Random order so a time-limited run doesn't keep re-checking the same markets.
        pending = conn.execute("SELECT id FROM markets WHERE outcome IS NULL ORDER BY RANDOM()").fetchall()
        log(f"Checking {len(pending)} unresolved markets")
        for row in pending:
            if deadline and time.monotonic() > deadline:
                timed_out = True
                log("Time budget reached; the next run continues from here")
                break
            checked += 1
            try:
                m = polymarket.fetch_market(row["id"])
            except Exception as e:  # one bad lookup shouldn't stop the rest
                failed += 1
                log(f"  failed to fetch {row['id']}: {e}")
                continue
            if m.resolved_outcome is not None:
                conn.execute(
                    "UPDATE markets SET outcome=?, resolved_at=? WHERE id=?",
                    (m.resolved_outcome, db.now(), m.id),
                )
                conn.commit()
                resolved += 1
                log(f"  resolved {'YES' if m.resolved_outcome else 'NO '}  {m.question[:70]}")
    return {"pending": len(pending), "checked": checked, "resolved": resolved, "failed": failed, "timed_out": timed_out}
