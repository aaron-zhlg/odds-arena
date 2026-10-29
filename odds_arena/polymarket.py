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
    )


def fetch_open_markets(limit: int = 20, min_volume: float = 10_000, within_days: float | None = 30) -> list[Market]:
    """Highest-volume open Yes/No markets. `within_days` keeps only markets ending that soon,
    so results (and scores) arrive in days rather than years; None means any end date."""
    params = {
        "active": "true",
        "closed": "false",
        "limit": 100,  # fetch a page, then filter down to `limit`
        "order": "volumeNum",
        "ascending": "false",
    }
    if within_days:
        now = datetime.now(timezone.utc)
        params["end_date_min"] = now.isoformat()
        params["end_date_max"] = (now + timedelta(days=within_days)).isoformat()
    r = httpx.get(f"{GAMMA_API}/markets", params=params, timeout=30)
    r.raise_for_status()
    markets = [_to_market(m) for m in r.json()]
    return [m for m in markets if m.is_yes_no and m.prices and m.volume >= min_volume][:limit]


def fetch_market(market_id: str) -> Market:
    r = httpx.get(f"{GAMMA_API}/markets/{market_id}", timeout=30)
    r.raise_for_status()
    return _to_market(r.json())
