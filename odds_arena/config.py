import os

from dotenv import load_dotenv

load_dotenv()

# Postgres connection string (Vercel's Supabase integration sets POSTGRES_URL). Unset = local SQLite.
DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")

# Vercel's filesystem is read-only except /tmp (which is also wiped between cold starts).
DB_PATH = os.getenv("ODDS_ARENA_DB") or ("/tmp/odds_arena.db" if os.getenv("VERCEL") else "odds_arena.db")
GAMMA_API = os.getenv("POLYMARKET_GAMMA_API", "https://gamma-api.polymarket.com")
# The market baseline plus every AI model; ODDS_ARENA_MODELS only needs setting to run a subset.
ENABLED_MODELS = [m.strip() for m in os.getenv("ODDS_ARENA_MODELS", "market,openai,typesafe,perplexity").split(",") if m.strip()]
