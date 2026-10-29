from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..polymarket import Market


@dataclass
class Prediction:
    probability: float  # P(first outcome, usually "Yes"), in [0, 1]
    rationale: str = ""


class DecisionModel(ABC):
    """Implement `predict` to add a new contender to the arena."""

    name: str = "base"
    # Models that peek at the live market price are flagged so the leaderboard can tell them apart.
    uses_market_price: bool = False
    # Search-enabled models get `context`: the same filtered web results for every such model.
    uses_search: bool = False

    @abstractmethod
    def predict(self, market: Market, context: list[dict] | None = None) -> Prediction: ...
