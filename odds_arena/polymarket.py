"""Thin client for Polymarket's public Gamma API (no auth needed for reads)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx

from .config import GAMMA_API


@dataclass
class Market:
    id: str
    question: str
    description: str
    outcomes: list[str]
    prices: list[float]  # market-implied probability per outcome
    end_date: str | None
    closed: bool
    volume: float
    resolution_status: str | None = None  # UMA oracle status, "resolved" once final
    slug: str | None = None
    # Multi-outcome events (e.g. "Which party wins the House?") are a group of binary markets, one per option.
    event_id: str | None = None
    event_title: str | None = None
    event_slug: str | None = None
    group_title: str | None = None  # this market's option within the event, e.g. "Democratic Party"
    first_token: str | None = None  # CLOB token of the first outcome, for its price history
    # True when the event's options are mutually exclusive (exactly one wins, e.g. "Which party wins?").
    # Other events can have several Yes options (e.g. "ceasefire holds through Oct 31 / Nov 15").
    neg_risk: bool = False

    @property
    def is_binary(self) -> bool:
        return len(self.outcomes) == 2

    @property
    def is_yes_no(self) -> bool:
        return [o.lower() for o in self.outcomes] == ["yes", "no"]

    @property
    def yes_price(self) -> float:
        return self.prices[0]

    @property
    def resolved_outcome(self) -> int | None:
        """1 if the first outcome (usually "Yes") won, 0 if it lost, None if unresolved."""
        # Closed only means trading stopped; wait for the oracle to finalize (it can be disputed).
        if not self.closed or not self.prices or self.resolution_status != "resolved":
            return None
        if self.prices[0] >= 0.99:
            return 1
        if self.prices[0] <= 0.01:
            return 0
        return None


def _parse_list(value) -> list:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return value or []


def _to_market(raw: dict) -> Market:
    event = (raw.get("events") or [{}])[0]
    return Market(
        id=str(raw["id"]),
        question=raw.get("question", ""),
        description=raw.get("description", "") or "",
        outcomes=_parse_list(raw.get("outcomes")),
        prices=[float(p) for p in _parse_list(raw.get("outcomePrices"))],
        end_date=raw.get("endDate"),
        closed=bool(raw.get("closed")),
        volume=float(raw.get("volume") or 0),
        resolution_status=raw.get("umaResolutionStatus"),
        slug=raw.get("slug"),
        event_id=str(event["id"]) if event.get("id") else None,
        event_title=event.get("title"),
        event_slug=event.get("slug"),
        group_title=raw.get("groupItemTitle") or None,
        first_token=(_parse_list(raw.get("clobTokenIds")) or [None])[0],
        neg_risk=bool(raw.get("negRisk")),
    )


PAGE_SIZE = 100


def _open_params(within_days: float | None, offset: int) -> dict:
    params = {"active": "true", "closed": "false", "limit": PAGE_SIZE, "offset": offset,
              "order": "volumeNum", "ascending": "false"}
    if within_days:
        now = datetime.now(timezone.utc)
        params["end_date_min"] = now.isoformat()
        params["end_date_max"] = (now + timedelta(days=within_days)).isoformat()
    return params


def _forecastable(m: Market, min_volume: float = 10_000) -> bool:
    # Every Polymarket market has two outcomes: Yes/No, or two named options ("Lakers" / "Celtics", "Over" / "Under").
    # Skip markets the crowd already treats as settled (e.g. a game that has ended but isn't resolved yet):
    # there is nothing left to forecast.
    return m.is_binary and bool(m.prices) and m.volume >= min_volume and 0.02 <= m.yes_price <= 0.98


def fetch_open_markets(limit: int = 20, within_days: float | None = 30) -> list[Market]:
    """The first page of highest-volume open markets. `within_days` keeps only markets ending that soon,
    so results (and scores) arrive in days rather than years; None means any end date."""
    r = httpx.get(f"{GAMMA_API}/markets", params=_open_params(within_days, 0), timeout=30)
    r.raise_for_status()
    return [m for m in map(_to_market, r.json()) if _forecastable(m)][:limit]


def iter_open_markets(within_days: float | None = 30, max_pages: int = 10):
    """Open markets page by page (highest volume first), so callers can stop as soon as they have enough.
    `max_pages` bounds the requests a single run can make."""
    for page in range(max_pages):
        r = httpx.get(f"{GAMMA_API}/markets", params=_open_params(within_days, page * PAGE_SIZE), timeout=30)
        r.raise_for_status()
        raw = r.json()
        yield from filter(_forecastable, map(_to_market, raw))
        if len(raw) < PAGE_SIZE:  # last page
            return


CLOB_API = "https://clob.polymarket.com"

# range -> (CLOB interval, minutes per point)
HISTORY_RANGES = {"1d": ("1d", 10), "1w": ("1w", 60), "1m": ("1m", 360), "all": ("max", 1440)}


def fetch_yes_token(market_id: str) -> str:
    r = httpx.get(f"{GAMMA_API}/markets/{market_id}", timeout=30)
    r.raise_for_status()
    return _parse_list(r.json().get("clobTokenIds"))[0]


def fetch_price_history(token_id: str, range_: str = "1w", since: int | None = None) -> list[tuple[int, float]]:
    """[(unix seconds, price of the token's outcome)] from Polymarket's public CLOB API.
    With `since`, only points after that time, hourly (for incremental syncs)."""
    if since is not None:
        params = {"market": token_id, "startTs": since, "fidelity": 60}
    else:
        interval, fidelity = HISTORY_RANGES[range_]
        params = {"market": token_id, "interval": interval, "fidelity": fidelity}
    r = httpx.get(f"{CLOB_API}/prices-history", params=params, timeout=30)
    r.raise_for_status()
    return [(int(p["t"]), float(p["p"])) for p in r.json().get("history", [])]


def fetch_market(market_id: str) -> Market:
    r = httpx.get(f"{GAMMA_API}/markets/{market_id}", timeout=30)
    r.raise_for_status()
    return _to_market(r.json())
