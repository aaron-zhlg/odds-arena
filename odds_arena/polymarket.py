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
    category: str | None = None  # from the event's Polymarket tags, see CATEGORIES

    @property
    def is_binary(self) -> bool:
        return len(self.outcomes) == 2

    @property
    def is_yes_no(self) -> bool:
        return [o.lower() for o in self.outcomes] == ["yes", "no"]

    @property
    def kind(self) -> str:
        return question_kind(self.outcomes, self.neg_risk, self.event_id)

    @property
    def option_label(self) -> str:
        """This market's option within its event ("Democratic Party"), or its question."""
        return self.group_title or self.question

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


# Question types, decided only from Polymarket's own fields:
#   one_winner: an option of an event Polymarket trades as mutually exclusive (negRisk), e.g. "Which party
#               wins the House?". Exactly one option resolves Yes, so the event is forecast as one Choice.
#   matchup:    two named outcomes ("Lakers" / "Celtics", "Over" / "Under"), forecast as a Choice of two.
#   yes_no:     everything else: a lone Yes/No question, or one option of an event whose options resolve
#               independently (e.g. "ceasefire holds through Oct 31 / Nov 15": both can be Yes).
YES_NO, MATCHUP, ONE_WINNER = "yes_no", "matchup", "one_winner"


def question_kind(outcomes: list[str], neg_risk: bool, event_id: str | None) -> str:
    if neg_risk and event_id:
        return ONE_WINNER
    if [o.lower() for o in outcomes] != ["yes", "no"]:
        return MATCHUP
    return YES_NO


# Display category -> the Polymarket tag slugs that mean it. An event usually carries several tags (a
# geopolitics event is tagged Politics too), so the first matching category in this order wins.
CATEGORIES = {
    "Sports": ["sports"],
    "Crypto": ["crypto"],
    "Weather": ["weather"],
    "Tech": ["tech", "ai"],
    "Culture": ["pop-culture"],
    "Finance": ["finance", "economy"],
    "Geopolitics": ["geopolitics"],
    "Politics": ["politics"],
}


def category_of(tags: list) -> str | None:
    slugs = {(t.get("slug") or "").lower() for t in tags if isinstance(t, dict)}
    return next((name for name, wanted in CATEGORIES.items() if slugs & set(wanted)), None)


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
        category=category_of(raw.get("tags") or event.get("tags") or []),
    )


PAGE_SIZE = 100


def _window(within_days: float | None) -> dict:
    if not within_days:
        return {}
    now = datetime.now(timezone.utc)
    return {"end_date_min": now.isoformat(), "end_date_max": (now + timedelta(days=within_days)).isoformat()}


def _forecastable(m: Market, min_volume: float = 10_000) -> bool:
    # Every Polymarket market has two outcomes: Yes/No, or two named options ("Lakers" / "Celtics", "Over" / "Under").
    # Skip markets the crowd already treats as settled (e.g. a game that has ended but isn't resolved yet):
    # there is nothing left to forecast.
    return m.is_binary and bool(m.prices) and m.volume >= min_volume and 0.02 <= m.yes_price <= 0.98


def _event_markets(e: dict) -> list[Market]:
    """An event's markets, with the event (and its tags, for the category) attached to each."""
    ev = {"id": e["id"], "title": e.get("title"), "slug": e.get("slug"), "tags": e.get("tags") or []}
    return [_to_market(raw | {"events": [ev], "negRisk": e.get("negRisk"), "tags": e.get("tags") or []})
            for raw in e.get("markets", []) if not raw.get("closed")]


HOT_EVENTS = 50


def iter_hot_events(n: int = HOT_EVENTS, within_days: float | None = 30, max_pages: int = 3):
    """The `n` hottest open events (by 24-hour trading volume) that still have something to forecast,
    hottest first. Yields Event objects with their forecastable options only."""
    found = 0
    for page in range(max_pages):
        r = httpx.get(f"{GAMMA_API}/events", timeout=30, params={
            "active": "true", "closed": "false", "limit": PAGE_SIZE, "offset": page * PAGE_SIZE,
            "order": "volume24hr", "ascending": "false", **_window(within_days)})
        r.raise_for_status()
        events = r.json()
        for e in events:
            options = [m for m in _event_markets(e) if _forecastable(m)]
            if not options:
                continue
            yield Event(id=str(e["id"]), title=e.get("title", ""), description=e.get("description") or "",
                        end_date=e.get("endDate"), options=options)
            found += 1
            if found >= n:
                return
        if len(events) < PAGE_SIZE:
            return


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


@dataclass
class Event:
    id: str
    title: str
    description: str
    end_date: str | None
    options: list[Market]  # open options only


def fetch_event(event_id: str) -> Event:
    """A one-winner event with its open options (closed ones, e.g. a candidate who dropped out, are left out)."""
    r = httpx.get(f"{GAMMA_API}/events/{event_id}", timeout=30)
    r.raise_for_status()
    e = r.json()
    options = []
    for raw in e.get("markets", []):
        if raw.get("closed") or not raw.get("active", True):
            continue
        # Markets listed inside an event don't repeat the event, so attach it.
        raw = raw | {"events": [{"id": e["id"], "title": e.get("title"), "slug": e.get("slug")}], "negRisk": e.get("negRisk"),
                     "tags": e.get("tags") or []}
        options.append(_to_market(raw))
    return Event(id=str(e["id"]), title=e.get("title", ""), description=e.get("description") or "",
                 end_date=e.get("endDate"), options=options)


def fetch_market(market_id: str) -> Market:
    r = httpx.get(f"{GAMMA_API}/markets/{market_id}", timeout=30)
    r.raise_for_status()
    return _to_market(r.json())
