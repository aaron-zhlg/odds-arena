from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from .. import db
from ..scoring import leaderboard

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Odds Arena")


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
