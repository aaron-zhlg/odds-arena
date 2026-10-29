"""Thin client for Polymarket's public Gamma API (no auth needed for reads)."""
from __future__ import annotations

import json
from dataclasses import dataclass

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

    @property
    def is_binary(self) -> bool:
        return len(self.outcomes) == 2

    @property
    def yes_price(self) -> float:
        return self.prices[0]

    @property
    def resolved_outcome(self) -> int | None:
        """1 if the first outcome (usually "Yes") won, 0 if it lost, None if unresolved."""
        if not self.closed or not self.prices:
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
    )


def fetch_open_markets(limit: int = 20, min_volume: float = 10_000) -> list[Market]:
    params = {
        "active": "true",
        "closed": "false",
        "limit": limit,
        "order": "volumeNum",
        "ascending": "false",
    }
    r = httpx.get(f"{GAMMA_API}/markets", params=params, timeout=30)
    r.raise_for_status()
    markets = [_to_market(m) for m in r.json()]
    return [m for m in markets if m.is_binary and m.prices and m.volume >= min_volume]


def fetch_market(market_id: str) -> Market:
    r = httpx.get(f"{GAMMA_API}/markets/{market_id}", timeout=30)
    r.raise_for_status()
    return _to_market(r.json())
