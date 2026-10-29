"""odds-arena predict | resolve | serve | demo"""
from __future__ import annotations

import argparse
import random
from datetime import date, timedelta

from . import db, jobs


def cmd_predict(args):
    jobs.run_predict(limit=args.limit, model_names=args.models)


def cmd_resolve(args):
    jobs.run_resolve()


def cmd_demo(args):
    """Fill the database with fake resolved markets so the dashboard has something to show."""
    rng = random.Random(42)
    with db.connect() as conn:
        for i in range(60):
            truth = rng.random()
            outcome = int(rng.random() < truth)
            price = min(max(truth + rng.gauss(0, 0.12), 0.02), 0.98)
            mid = f"demo-{i}"
            conn.execute(
                """INSERT INTO markets (id, question, end_date, outcome, resolved_at) VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT (id) DO UPDATE SET question = excluded.question, end_date = excluded.end_date,
                   outcome = excluded.outcome, resolved_at = excluded.resolved_at""",
                (mid, f"Demo question #{i + 1}?", (date.today() - timedelta(days=i)).isoformat(), outcome, db.now()),
            )
            preds = {
                "market": price,
                "coinflip": 0.5,
                "openai": min(max(truth + rng.gauss(0, 0.2), 0.01), 0.99),
                "typesafe": min(max(truth + rng.gauss(0, 0.16), 0.01), 0.99),
            }
            for model, p in preds.items():
                conn.execute(
                    """INSERT INTO predictions (market_id, model, probability, market_price, rationale) VALUES (?, ?, ?, ?, 'demo')
                       ON CONFLICT (market_id, model) DO UPDATE SET probability = excluded.probability,
                       market_price = excluded.market_price, rationale = excluded.rationale""",
                    (mid, model, p, price),
                )
    print("Demo data written. Run `odds-arena serve` to view it.")


def cmd_check_keys(args):
    """Make one signed read-only request to confirm the Polymarket US key works."""
    from .polymarket_us import PolymarketUSClient

    data = PolymarketUSClient().positions()
    count = len(data) if isinstance(data, list) else len(data.get("positions", data)) if isinstance(data, dict) else "?"
    print(f"Polymarket US key OK. Signed request succeeded ({count} position entries).")


def cmd_serve(args):
    import uvicorn
    uvicorn.run("odds_arena.web.app:app", host=args.host, port=args.port, reload=False)


def main():
    p = argparse.ArgumentParser(prog="odds-arena")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("predict", help="Fetch open markets and record each model's prediction")
    sp.add_argument("--limit", type=int, default=20)
    sp.add_argument("--models", nargs="*", help="Override ODDS_ARENA_MODELS")
    sp.set_defaults(func=cmd_predict)

    sub.add_parser("resolve", help="Check Polymarket for resolved outcomes").set_defaults(func=cmd_resolve)
    sub.add_parser("check-keys", help="Verify your Polymarket US API key with a signed request").set_defaults(func=cmd_check_keys)
    sub.add_parser("demo", help="Seed fake data for the dashboard").set_defaults(func=cmd_demo)

    sp = sub.add_parser("serve", help="Run the leaderboard website")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.set_defaults(func=cmd_serve)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
