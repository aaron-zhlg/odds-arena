"""predict / resolve jobs, shared by the CLI and the Vercel cron endpoints.

Each job stops starting new work once `time_budget` seconds have passed, so it fits inside a
serverless function's time limit. Work is committed as it goes; the next run picks up the rest.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from . import db, polymarket, search
from .config import ENABLED_MODELS
from .models import load_models

Log = Callable[[str], None]


def run_predict(limit: int = 10, model_names: list[str] | None = None, within_days: float | None = 30,
                time_budget: float | None = None, log: Log = print) -> dict:
    """Keep the hottest Polymarket questions forecast: walk the top HOT_EVENTS events (by 24-hour volume),
    skipping ones every model already answered, and forecast up to `limit` events this run. Each model is
    called once per event, whatever its type."""
    deadline = time.monotonic() + time_budget if time_budget else None
    models = load_models(model_names or ENABLED_MODELS)
    made = failed = worked = 0
    timed_out = False
    with db.connect() as conn, ThreadPoolExecutor(max_workers=max(len(models), 1)) as pool:
        for event in polymarket.iter_hot_events(within_days=within_days):
            if worked >= limit:
                break
            if deadline and time.monotonic() > deadline:
                timed_out = True
                log("Time budget reached; the next run continues from here")
                break
            n_made, n_failed, did = _predict_event(conn, pool, models, event, log)
            made, failed, worked = made + n_made, failed + n_failed, worked + did
    return {"markets": worked, "predictions": made, "failed": failed, "timed_out": timed_out}


def _save_market(conn, m) -> None:
    fields = {
        "question": m.question, "end_date": m.end_date, "description": m.description, "slug": m.slug,
        "outcomes": json.dumps(m.outcomes, ensure_ascii=False), "event_id": m.event_id, "event_title": m.event_title,
        "event_slug": m.event_slug, "group_title": m.group_title, "neg_risk": int(m.neg_risk),
        "category": m.category,
    }
    cols = ", ".join(fields)
    updates = ", ".join(f"{c} = excluded.{c}" for c in fields if c != "question")
    conn.execute(
        f"INSERT INTO markets (id, {cols}) VALUES (?, {', '.join('?' * len(fields))}) ON CONFLICT (id) DO UPDATE SET {updates}",
        (m.id, *fields.values()),
    )


def _predict_event(conn, pool, models, event, log: Log) -> tuple[int, int, int]:
    """Forecast one event as a unit, one call per model:
      one_winner          -> predict_event: a distribution over all open options (adds up to 1)
      several options     -> predict_options: each option's own chance (they can all happen)
      a single market     -> predict
    Returns (made, failed, worked)."""
    if event.options[0].kind == polymarket.ONE_WINNER:
        event = polymarket.fetch_event(event.id)  # every open option, long shots included, for the distribution
        call = lambda model, ctx: model.predict_event(event, ctx)
    elif len(event.options) > 1:
        call = lambda model, ctx: model.predict_options(event, ctx)
    else:
        only = event.options[0]
        call = lambda model, ctx: {only.id: model.predict(only, ctx)}
    opts = event.options
    if not opts:
        return 0, 0, 0
    marks = ",".join("?" * len(opts))
    have = {(r["market_id"], r["model"]) for r in conn.execute(
        f"SELECT market_id, model FROM predictions WHERE market_id IN ({marks})", [o.id for o in opts]).fetchall()}
    todo = [model for model in models if any((o.id, model.name) not in have for o in opts)]
    if not todo:
        return 0, 0, 0
    for o in opts:
        _save_market(conn, o)
    made = failed = 0
    context = None
    if any(model.uses_search for model in todo):
        query = event.title if len(opts) > 1 else opts[0].question
        try:  # one search per event, shared by every option and every model
            context = _search_context(conn, [o.id for o in opts], query, log)
        except Exception as e:  # no search context: skip the search models rather than predict without it
            log(f"  [search] failed on event {event.id}: {e}")
            failed += sum(1 for model in todo if model.uses_search)
            todo = [model for model in todo if not model.uses_search]
    # Models are independent services, so call them at the same time; writes stay on this thread.
    futures = {model: pool.submit(call, model, context if model.uses_search else None) for model in todo}
    for model, fut in futures.items():
        try:
            preds = fut.result()
        except Exception as e:  # one failing model shouldn't stop the others
            failed += 1
            log(f"  [{model.name}] failed on event {event.id}: {e}")
            continue
        rows = [(o.id, model.name, preds[o.id].probability, o.yes_price, preds[o.id].rationale)
                for o in opts if o.id in preds and (o.id, model.name) not in have]
        conn.executemany(
            "INSERT INTO predictions (market_id, model, probability, market_price, rationale) VALUES (?, ?, ?, ?, ?)", rows)
        made += len(rows)
        log(f"  [{model.name}] {len(rows)} option(s)  {event.title[:60]}")
    conn.commit()
    return made, failed, 1


def _search_context(conn, market_ids: list[str], query: str, log: Log) -> list[dict]:
    """Stored search results for these markets, or one new search stored for all of them before use.
    Stored results are reused, so a +search model added later sees what the others saw."""
    row = conn.execute("SELECT results FROM search_contexts WHERE market_id = ?", (market_ids[0],)).fetchone()
    if row:
        return json.loads(row["results"])
    context = search.search_context(query)
    blob = json.dumps(context, ensure_ascii=False)
    conn.executemany(
        "INSERT INTO search_contexts (market_id, query, results) VALUES (?, ?, ?) ON CONFLICT (market_id) DO NOTHING",
        [(mid, search.query_for(query), blob) for mid in market_ids],
    )
    conn.commit()
    log(f"  [search] {len(context)} sources  {query[:60]}")
    return context


def run_resolve(time_budget: float | None = None, log: Log = print) -> dict:
    """For every unresolved market: record the outcome once Polymarket settles it, and refresh its prices."""
    deadline = time.monotonic() + time_budget if time_budget else None
    checked = resolved = failed = prices = 0
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
            conn.execute("UPDATE markets SET neg_risk = ? WHERE id = ?", (int(m.neg_risk), m.id))
            try:
                prices += _sync_prices(conn, m)
            except Exception as e:  # price history is nice to have; never block resolution on it
                log(f"  price history failed for {m.id}: {e}")
            if m.resolved_outcome is not None:
                conn.execute(
                    "UPDATE markets SET outcome=?, resolved_at=? WHERE id=?",
                    (m.resolved_outcome, db.now(), m.id),
                )
                resolved += 1
                log(f"  resolved {m.outcomes[0 if m.resolved_outcome else 1]}  {m.question[:70]}")
            conn.commit()
    return {"pending": len(pending), "checked": checked, "resolved": resolved, "prices": prices,
            "failed": failed, "timed_out": timed_out}


def _sync_prices(conn, m) -> int:
    """Copy the market's price history into price_points so the site only ever reads our database.
    First sight: the whole history daily plus the last week hourly. After that, incremental: only points
    newer than the latest one stored."""
    if not m.first_token:
        return 0
    latest = conn.execute("SELECT MAX(t) AS t FROM price_points WHERE market_id = ?", (m.id,)).fetchone()["t"]
    if latest is None:
        points = polymarket.fetch_price_history(m.first_token, "all") + polymarket.fetch_price_history(m.first_token, "1w")
    else:
        points = polymarket.fetch_price_history(m.first_token, since=latest + 1)
    conn.executemany("INSERT INTO price_points (market_id, t, p) VALUES (?, ?, ?) ON CONFLICT (market_id, t) DO NOTHING",
                     [(m.id, t, p) for t, p in points])
    return len(points)
