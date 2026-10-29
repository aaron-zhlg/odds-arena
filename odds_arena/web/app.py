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
# Vercel functions run up to 300 s by default; stop starting new work well before that. Each event is
# committed as soon as it's done, so a run cut short loses nothing.
CRON_TIME_BUDGET = float(os.getenv("CRON_TIME_BUDGET", "240"))


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


MARKET_COLS = """id, question, end_date, outcome, outcomes, event_id, event_title, event_slug, group_title, neg_risk,
                 category"""


def _in(ids) -> str:
    return ",".join("?" * len(ids)) or "NULL"


def _with_predictions(conn, markets, rationale: bool = True) -> list[dict]:
    """Markets as JSON with their predictions and latest synced price, querying only these markets.
    Cards don't show rationales (the detail does), so list pages can leave them out."""
    ids = [m["id"] for m in markets]
    if not ids:
        return []
    by_market: dict[str, dict] = {}
    for p in conn.execute(f"SELECT market_id, model, probability, rationale, created_at FROM predictions WHERE market_id IN ({_in(ids)})", ids).fetchall():
        by_market.setdefault(p["market_id"], {})[p["model"]] = {"p": p["probability"], "at": p["created_at"]} | (
            {"rationale": p["rationale"]} if rationale else {})
    # The crowd's latest price, kept current by the resolve job (every few minutes), vs. the snapshot
    # stored with the predictions, which is what models are scored against.
    latest = {r["market_id"]: (r["p"], r["t"]) for r in conn.execute(
        f"""SELECT pp.market_id, pp.p, pp.t FROM price_points pp
            JOIN (SELECT market_id, MAX(t) AS t FROM price_points WHERE market_id IN ({_in(ids)}) GROUP BY market_id) l
              ON l.market_id = pp.market_id AND l.t = pp.t""", ids).fetchall()}
    return [_market_json(m) | {"predictions": by_market.get(m["id"], {}),
                                "price_now": latest.get(m["id"], (None, None))[0],
                                "price_at": latest.get(m["id"], (None, None))[1]} for m in markets]


@app.get("/api/markets")
def api_markets(limit: int = 50, resolved: bool = False):
    """Markets with predictions; `resolved=1` for settled ones only (what the calibration chart needs)."""
    with db.connect() as conn:
        markets = conn.execute(
            f"""SELECT {MARKET_COLS} FROM markets {"WHERE outcome IS NOT NULL" if resolved else ""}
                ORDER BY COALESCE(resolved_at, end_date) DESC LIMIT ?""",
            (min(limit, 5000),),
        ).fetchall()
        return _with_predictions(conn, markets)


# A question is one card: a Polymarket event (its options grouped), or a market saved before events were.
QUESTION_KEY = "COALESCE(event_id, id)"


@app.get("/api/questions")
def api_questions(category: str | None = None, status: str = "all", q: str = "", offset: int = 0, limit: int = 30):
    """One page of questions, filtered and searched server-side. `status`: all | open | resolved (any option
    settled). Order: open before settled, then categories interleaved (soonest-closing first within each) so
    one topic doesn't fill a page."""
    limit = max(1, min(limit, 100))
    where, params = [], []
    if category:
        where.append("COALESCE(category, 'Other') = ?"); params.append(category)
    if q.strip():
        like = f"%{q.strip().lower()}%"
        where.append("(LOWER(question) LIKE ? OR LOWER(COALESCE(group_title, '')) LIKE ? OR LOWER(COALESCE(event_title, '')) LIKE ?)")
        params += [like, like, like]
    having = {"open": "HAVING SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END) = 0",
              "resolved": "HAVING SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END) > 0"}.get(status, "")
    with db.connect() as conn:
        groups = conn.execute(
            f"""SELECT {QUESTION_KEY} AS qk, COALESCE(MAX(category), 'Other') AS category, MIN(end_date) AS end_date,
                       SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END) AS settled
                FROM markets {"WHERE " + " AND ".join(where) if where else ""}
                GROUP BY {QUESTION_KEY} {having}""", params).fetchall()
        # Interleave categories so the first page is a mix; settled questions after open ones.
        buckets: dict[tuple, list] = {}
        for g in sorted(groups, key=lambda g: g["end_date"] or "9"):
            buckets.setdefault((g["settled"] > 0, g["category"]), []).append(g["qk"])
        ordered = []
        for done in (False, True):
            lanes = [v for (d, _), v in buckets.items() if d == done]
            for i in range(max((len(v) for v in lanes), default=0)):
                ordered += [v[i] for v in lanes if i < len(v)]
        page = ordered[offset:offset + limit]
        markets = conn.execute(
            f"SELECT {MARKET_COLS} FROM markets WHERE {QUESTION_KEY} IN ({_in(page)})", page).fetchall() if page else []
        items = _with_predictions(conn, markets, rationale=False)
    by_q: dict[str, list] = {}
    for m in items:
        by_q.setdefault(m["event_id"] or m["id"], []).append(m)
    questions = [{"key": k, "title": by_q[k][0]["event_title"] or by_q[k][0]["question"], "items": by_q[k]}
                 for k in page if k in by_q]
    return {"total": len(ordered), "offset": offset, "limit": limit, "questions": questions}


@app.get("/api/meta")
def api_meta():
    """Small facts for the page chrome: question counts per category, resolved count, and the models."""
    with db.connect() as conn:
        cats = conn.execute(
            f"SELECT COALESCE(category, 'Other') AS c, COUNT(DISTINCT {QUESTION_KEY}) AS n FROM markets GROUP BY COALESCE(category, 'Other')"
        ).fetchall()
        resolved = conn.execute(f"SELECT COUNT(DISTINCT {QUESTION_KEY}) AS n FROM markets WHERE outcome IS NOT NULL").fetchone()["n"]
        models = [r["model"] for r in conn.execute("SELECT DISTINCT model FROM predictions").fetchall()]
        next_close = conn.execute("SELECT MIN(end_date) AS d FROM markets WHERE outcome IS NULL AND end_date >= ?",
                                  (db.now()[:10],)).fetchone()["d"]
    return {"categories": {r["c"]: r["n"] for r in cats}, "resolved": resolved, "models": sorted(models),
            "next_close": next_close}


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
        limit=int(os.getenv("CRON_PREDICT_LIMIT", "50")),  # questions per run: the whole hot list if time allows
        within_days=float(os.getenv("CRON_WITHIN_DAYS", "30")) or None,
        time_budget=CRON_TIME_BUDGET,
    )


@app.get("/api/cron/resolve")
def cron_resolve(authorization: str | None = Header(default=None)):
    _check_cron_auth(authorization)
    return _run_job(jobs.run_resolve, time_budget=CRON_TIME_BUDGET)
