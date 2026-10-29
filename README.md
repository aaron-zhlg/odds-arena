# Odds Arena

Decision models forecast [Polymarket](https://polymarket.com) questions. Once markets resolve, each model is scored and compared on a leaderboard webpage.

## How it works

1. **`odds-arena predict`** pulls the highest-volume open Yes/No markets ending within 30 days (`--within-days`) from Polymarket's public Gamma API and asks every enabled model for P(Yes). Predictions are stored in SQLite, one per model per market (made once, never overwritten).
2. **`odds-arena resolve`** re-checks open markets and records the outcome once Polymarket's oracle marks them `resolved` (closed alone isn't final).
3. **`odds-arena serve`** runs the website: a leaderboard (Brier score, log loss, accuracy, skill vs. coin flip) plus a per-market table of every model's call.

## Models

| Name | What it does |
|------|--------------|
| `market` | Baseline: echoes the current Polymarket price (the crowd) |
| `coinflip` | Baseline: always 50% |
| `openai` | Asks an OpenAI model (`OPENAI_MODEL`) for a probability, without showing it the price |
| `typesafe` | Asks TypeSafe's Jev one Noul question (P resolves Yes), without the price; set `TYPESAFE_API_KEY` |
| `perplexity` | Same question via Perplexity's Decisions API (`pplx-decider-latest`); set `PERPLEXITY_API_KEY` |

Add a model: subclass `DecisionModel` in `odds_arena/models/`, implement `predict(market) -> Prediction`, and register it in `odds_arena/models/__init__.py`.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env   # add your API keys, choose ODDS_ARENA_MODELS

odds-arena check-keys  # optional: verify your Polymarket US API key
odds-arena demo        # optional: seed fake resolved markets to preview the page
odds-arena predict --limit 20
odds-arena resolve     # run periodically (e.g. daily cron)
odds-arena serve       # http://127.0.0.1:8000
```

## Polymarket US API key

Market data comes from the public Gamma API and needs no key. For authenticated Polymarket US endpoints,
put `POLYMARKET_KEY_ID` and `POLYMARKET_SECRET_KEY` in `.env` and use `odds_arena.polymarket_us.PolymarketUSClient`,
which signs each request with Ed25519 as described in the [Polymarket US auth docs](https://docs.polymarket.us/api-reference/authentication).

## Deploying on Vercel with Supabase

1. Import the GitHub repo into Vercel (the FastAPI entrypoint is set in `pyproject.toml`).
2. In the Vercel project, **Storage → Create → Supabase** and connect it to the project. This sets `POSTGRES_URL`.
3. Copy `POSTGRES_URL` into your local `.env`, then run `odds-arena predict` / `odds-arena resolve` from your machine.
   They write to Supabase, and the deployed site reads from it.

### Automatic daily runs (Vercel Cron)

`vercel.json` schedules `/api/cron/predict` (14:00 UTC) and `/api/cron/resolve` (15:00 UTC) once a day.
Add these in Vercel → Settings → Environment Variables, then redeploy:

- `CRON_SECRET`: random string (`openssl rand -hex 32`); the endpoints refuse every call without it
- `OPENAI_API_KEY`, `ODDS_ARENA_MODELS` (e.g. `market,coinflip,openai`), optionally `OPENAI_MODEL`
- optional: `CRON_PREDICT_LIMIT` (default 10 markets), `CRON_WITHIN_DAYS` (default 30, 0 = any end date), `CRON_TIME_BUDGET` (default 45 s)

Each run stops starting new work once the time budget is used and commits as it goes, so the next run continues.

Without `POSTGRES_URL` everything uses a local SQLite file. On Vercel that file lives in `/tmp` and is wiped often,
so a hosted database is needed for real data.

## Scoring

- **Brier score**: mean of (p − outcome)². Lower is better; always guessing 50% scores 0.25.
- **Log loss**: punishes confident wrong calls much harder.
- **Accuracy**: share of markets where p ≥ 0.5 matched the outcome.

Markets resolve over weeks or months, so leave `predict` and `resolve` running on a schedule to build up a meaningful sample.
