import hmac
import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse

from .. import db, jobs
from ..scoring import leaderboard

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Odds Arena")

# Cron runs must finish inside the serverless time limit; jobs stop starting new work after this.
CRON_TIME_BUDGET = float(os.getenv("CRON_TIME_BUDGET", "45"))


@app.get("/")
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
            "SELECT id, question, end_date, outcome FROM markets ORDER BY COALESCE(resolved_at, end_date) DESC LIMIT ?",
            (limit,),
        ).fetchall()
        preds = conn.execute("SELECT market_id, model, probability, rationale FROM predictions").fetchall()
    by_market: dict[str, dict] = {}
    for p in preds:
        by_market.setdefault(p["market_id"], {})[p["model"]] = {"p": p["probability"], "rationale": p["rationale"]}
    return [dict(m) | {"predictions": by_market.get(m["id"], {})} for m in markets]


def _check_cron_auth(authorization: str | None):
    # Vercel Cron sends `Authorization: Bearer $CRON_SECRET`. Without a secret configured, refuse.
    secret = os.getenv("CRON_SECRET")
    if not secret or not hmac.compare_digest(authorization or "", f"Bearer {secret}"):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/api/cron/predict")
def cron_predict(authorization: str | None = Header(default=None)):
    _check_cron_auth(authorization)
    return jobs.run_predict(limit=int(os.getenv("CRON_PREDICT_LIMIT", "10")), time_budget=CRON_TIME_BUDGET)


@app.get("/api/cron/resolve")
def cron_resolve(authorization: str | None = Header(default=None)):
    _check_cron_auth(authorization)
    return jobs.run_resolve(time_budget=CRON_TIME_BUDGET)
