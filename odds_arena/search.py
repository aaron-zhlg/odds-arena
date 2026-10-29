"""Web context for forecasts, from Brave's LLM Context API.

Each market is searched once; every search-enabled model gets the exact same context, so the
comparison stays fair. Prediction-market sites, and any snippet that mentions one, are dropped:
their odds are the crowd's answer, which the models are not supposed to see.
Docs: https://api-dashboard.search.brave.com/documentation/services/llm-context
"""
from __future__ import annotations

import os
import re
import time
from urllib.parse import urlsplit

import httpx

from . import config  # noqa: F401  (loads .env)

API_URL = "https://api.search.brave.com/res/v1/llm/context"

# Brave's budget is approximate. Jev's default per-question limit is 32k tokens (state + question),
# so cap well below it; Perplexity's decider takes 256k.
DEFAULT_MAX_TOKENS = 8192
MAX_TOKENS_CAP = 16384

PREDICTION_MARKET_DOMAINS = (
    "polymarket.com", "kalshi.com", "predictit.org", "manifold.markets",
    "metaculus.com", "insightprediction.com", "betfair.com", "smarkets.com",
)
PREDICTION_MARKET_WORDS = re.compile(
    r"polymarket|kalshi|predictit|manifold markets|metaculus|betfair|smarkets|prediction market",
    re.IGNORECASE,
)


def _is_prediction_market(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in PREDICTION_MARKET_DOMAINS)


def query_for(question: str) -> str:
    # Brave accepts at most 75 words / 600 characters.
    return " ".join(question.split()[:75])[:600]


def clean(data: dict) -> list[dict]:
    """Turn a Brave response into the context we send: one entry per source, filtered."""
    sources = data.get("sources") or {}
    out = []
    for item in (data.get("grounding") or {}).get("generic") or []:
        url = item.get("url", "")
        # An article whose title is about prediction markets is reporting their odds: drop it whole.
        if _is_prediction_market(url) or PREDICTION_MARKET_WORDS.search(item.get("title", "")):
            continue
        snippets = [s for s in item.get("snippets") or [] if not PREDICTION_MARKET_WORDS.search(s)]
        if not snippets:
            continue
        age = (sources.get(url) or {}).get("age") or []
        out.append({
            "source": urlsplit(url).hostname,
            "title": item.get("title", ""),
            "published": age[1] if len(age) > 1 and age[1] else "unknown",
            "snippets": snippets,
        })
    return out


def search_context(question: str, max_tokens: int | None = None, api_key: str | None = None) -> list[dict]:
    max_tokens = min(max_tokens or int(os.getenv("SEARCH_MAX_TOKENS", DEFAULT_MAX_TOKENS)), MAX_TOKENS_CAP)
    params = {"q": query_for(question), "maximum_number_of_tokens": max_tokens}
    headers = {"X-Subscription-Token": api_key or os.environ["BRAVE_API_KEY"], "Accept": "application/json"}
    for attempt in range(3):
        r = httpx.get(API_URL, params=params, headers=headers, timeout=30)
        if r.status_code == 429 and attempt < 2:
            time.sleep(float(r.headers.get("retry-after", 1 + attempt)))
            continue
        r.raise_for_status()
        return clean(r.json())
    raise RuntimeError("unreachable")
