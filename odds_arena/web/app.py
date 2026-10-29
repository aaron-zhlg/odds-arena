import hmac
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse

from .. import db, jobs, polymarket
from ..scoring import brier, leaderboard

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Odds Arena")

# Cron runs must finish inside the serverless time limit; jobs stop starting new work after this.
CRON_TIME_BUDGET = float(os.getenv("CRON_TIME_BUDGET", "45"))


@app.get("/")
@app.get("/markets")  # full market list; the same page renders it from the path
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/leaderboard")
def api_leaderboard():
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT p.model, p.probability, m.outcome
               FROM predictions p JOIN markets m ON m.id = p.market_id
               WHERE m.outcome IS NOT NULL"""
        ).fetchall()
        pending = conn.execute("SELECT COUNT(*) AS n FROM markets WHERE outcome IS NULL").fetchone()["n"]
    return {"leaderboard": leaderboard((r["model"], r["probability"], r["outcome"]) for r in rows), "pending_markets": pending}


@app.get("/api/markets")
def api_markets(limit: int = 50):
    with db.connect() as conn:
        markets = conn.execute(
            """SELECT id, question, end_date, outcome, outcomes, event_id, event_title, event_slug, group_title
               FROM markets ORDER BY COALESCE(resolved_at, end_date) DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        preds = conn.execute("SELECT market_id, model, probability, rationale FROM predictions").fetchall()
    by_market: dict[str, dict] = {}
    for p in preds:
        by_market.setdefault(p["market_id"], {})[p["model"]] = {"p": p["probability"], "rationale": p["rationale"]}
    return [_market_json(m) | {"predictions": by_market.get(m["id"], {})} for m in markets]


def _market_json(row) -> dict:
    d = dict(row)
    # Older rows predate the outcomes column; they were all Yes/No markets.
    d["outcomes"] = json.loads(d["outcomes"]) if d.get("outcomes") else ["Yes", "No"]
    return d


@app.get("/api/markets/{market_id}")
def api_market(market_id: str):
    with db.connect() as conn:
        m = conn.execute(
            """SELECT id, question, end_date, outcome, resolved_at, description, slug, outcomes,
                      event_id, event_title, event_slug, group_title FROM markets WHERE id = ?""",
            (market_id,),
        ).fetchone()
        if not m:
            raise HTTPException(status_code=404, detail="Market not found")
        ctx = conn.execute("SELECT query, results, created_at FROM search_contexts WHERE market_id = ?", (market_id,)).fetchone()
        preds = conn.execute(
            "SELECT model, probability, market_price, rationale, created_at FROM predictions WHERE market_id = ? ORDER BY model",
            (market_id,),
        ).fetchall()
        # The other options of the same multi-outcome event, e.g. the other parties in "Which party wins the House?"
        siblings = conn.execute(
            "SELECT id, question, group_title, outcome FROM markets WHERE event_id = ? AND id <> ? ORDER BY group_title",
            (m["event_id"], market_id),
        ).fetchall() if m["event_id"] else []
    outcome = m["outcome"]
    return _market_json(m) | {
        "siblings": [dict(s) for s in siblings],
        "search": None if not ctx else {"query": ctx["query"], "created_at": ctx["created_at"], "results": json.loads(ctx["results"])},
        "predictions": [
            dict(p) | {"brier": None if outcome is None else brier(p["probability"], outcome)} for p in preds
        ],
    }


# Price history is public but rate limited; cache per (market, range) so repeat views don't refetch.
HISTORY_TTL = 300
_history_cache: dict[tuple[str, str], tuple[float, list]] = {}
_token_cache: dict[str, str] = {}


@app.get("/api/markets/{market_id}/history")
def api_market_history(market_id: str, range: str = "1w"):
    if range not in polymarket.HISTORY_RANGES:
        raise HTTPException(status_code=400, detail=f"range must be one of {list(polymarket.HISTORY_RANGES)}")
    hit = _history_cache.get((market_id, range))
    if hit and time.time() - hit[0] < HISTORY_TTL:
        return {"range": range, "points": hit[1]}
    try:
        token = _token_cache.get(market_id) or polymarket.fetch_yes_token(market_id)
        _token_cache[market_id] = token
        points = polymarket.fetch_price_history(token, range)
    except Exception as e:  # unknown market, or Polymarket unavailable
        raise HTTPException(status_code=502, detail=f"Price history unavailable: {e}")
    _history_cache[(market_id, range)] = (time.time(), points)
    return {"range": range, "points": points}


def _check_cron_auth(authorization: str | None):
    # Vercel Cron sends `Authorization: Bearer $CRON_SECRET`. Without a secret configured, refuse.
    secret = os.getenv("CRON_SECRET")
    if not secret or not hmac.compare_digest(authorization or "", f"Bearer {secret}"):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/api/cron/predict")
def cron_predict(authorization: str | None = Header(default=None)):
    _check_cron_auth(authorization)
    return jobs.run_predict(
        limit=int(os.getenv("CRON_PREDICT_LIMIT", "10")),
        within_days=float(os.getenv("CRON_WITHIN_DAYS", "30")) or None,
        time_budget=CRON_TIME_BUDGET,
    )


@app.get("/api/cron/resolve")
def cron_resolve(authorization: str | None = Header(default=None)):
    _check_cron_auth(authorization)
    return jobs.run_resolve(time_budget=CRON_TIME_BUDGET)
