import os

from dotenv import load_dotenv

load_dotenv()

# Vercel's filesystem is read-only except /tmp (which is also wiped between cold starts).
DB_PATH = os.getenv("ODDS_ARENA_DB") or ("/tmp/odds_arena.db" if os.getenv("VERCEL") else "odds_arena.db")
GAMMA_API = os.getenv("POLYMARKET_GAMMA_API", "https://gamma-api.polymarket.com")
ENABLED_MODELS = [m.strip() for m in os.getenv("ODDS_ARENA_MODELS", "market,coinflip").split(",") if m.strip()]
