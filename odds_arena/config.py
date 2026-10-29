import os

from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("ODDS_ARENA_DB", "odds_arena.db")
GAMMA_API = os.getenv("POLYMARKET_GAMMA_API", "https://gamma-api.polymarket.com")
ENABLED_MODELS = [m.strip() for m in os.getenv("ODDS_ARENA_MODELS", "market,coinflip").split(",") if m.strip()]
