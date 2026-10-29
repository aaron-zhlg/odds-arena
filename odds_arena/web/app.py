import hmac
import json
import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from .. import db, jobs
from ..polymarket import question_kind
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
            """SELECT id, question, end_date, outcome, outcomes, event_id, event_title, event_slug, group_title, neg_risk,
                      category FROM markets ORDER BY COALESCE(resolved_at, end_date) DESC LIMIT ?""",
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
    d["kind"] = question_kind(d["outcomes"], bool(d.get("neg_risk")), d.get("event_id"))  # the one rule, server-side
    return d


@app.get("/api/markets/{market_id}")
def api_market(market_id: str):
    with db.connect() as conn:
        m = conn.execute(
            """SELECT id, question, end_date, outcome, resolved_at, description, slug, outcomes,
                      event_id, event_title, event_slug, group_title, neg_risk, category FROM markets WHERE id = ?""",
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
        sibling_preds: dict[str, dict] = {}
        for s in siblings:
            for p in conn.execute("SELECT model, probability FROM predictions WHERE market_id = ?", (s["id"],)).fetchall():
                sibling_preds.setdefault(s["id"], {})[p["model"]] = p["probability"]
    outcome = m["outcome"]
    return _market_json(m) | {
        "siblings": [dict(s) | {"predictions": sibling_preds.get(s["id"], {})} for s in siblings],
        "search": None if not ctx else {"query": ctx["query"], "created_at": ctx["created_at"], "results": json.loads(ctx["results"])},
        "predictions": [
            dict(p) | {"brier": None if outcome is None else brier(p["probability"], outcome)} for p in preds
        ],
    }


# Seconds of history per chart range; "all" has no cutoff. Points come from price_points, filled by the
# resolve job, so the site never calls Polymarket itself.
HISTORY_SPANS = {"1d": 86400, "1w": 7 * 86400, "1m": 30 * 86400, "all": None}


@app.get("/api/markets/{market_id}/history")
def api_market_history(market_id: str, range: str = "1w"):
    if range not in HISTORY_SPANS:
        raise HTTPException(status_code=400, detail=f"range must be one of {list(HISTORY_SPANS)}")
    with db.connect() as conn:
        latest = conn.execute("SELECT MAX(t) AS t FROM price_points WHERE market_id = ?", (market_id,)).fetchone()["t"]
        if latest is None:
            return {"range": range, "points": []}
        since = latest - HISTORY_SPANS[range] if HISTORY_SPANS[range] else 0
        rows = conn.execute("SELECT t, p FROM price_points WHERE market_id = ? AND t >= ? ORDER BY t",
                            (market_id, since)).fetchall()
    return {"range": range, "points": [[r["t"], r["p"]] for r in rows]}


def _check_cron_auth(authorization: str | None):
    # Vercel Cron sends `Authorization: Bearer $CRON_SECRET`. Without a secret configured, refuse.
    secret = os.getenv("CRON_SECRET")
    if not secret or not hmac.compare_digest(authorization or "", f"Bearer {secret}"):
        raise HTTPException(status_code=401, detail="Unauthorized")


def _run_job(fn, **kwargs):
    # Cron callers can't see server logs, so say what went wrong. Messages name settings (model names,
    # env var names), never their values, and only callers holding CRON_SECRET get this far.
    try:
        return fn(**kwargs)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"{type(e).__name__}: {e}"})


@app.get("/api/cron/predict")
def cron_predict(authorization: str | None = Header(default=None)):
    _check_cron_auth(authorization)
    return _run_job(
        jobs.run_predict,
        limit=int(os.getenv("CRON_PREDICT_LIMIT", "10")),
        within_days=float(os.getenv("CRON_WITHIN_DAYS", "30")) or None,
        time_budget=CRON_TIME_BUDGET,
    )


@app.get("/api/cron/resolve")
def cron_resolve(authorization: str | None = Header(default=None)):
    _check_cron_auth(authorization)
    return _run_job(jobs.run_resolve, time_budget=CRON_TIME_BUDGET)
