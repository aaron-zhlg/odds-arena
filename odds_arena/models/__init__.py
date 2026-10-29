"""Registry of decision models. Add yours here to enter it into the arena."""
from .base import DecisionModel, Prediction
from .baselines import CoinFlipModel, MarketPriceModel


def _openai():
    from .openai_model import OpenAIModel
    return OpenAIModel()


def _typesafe():
    from .typesafe_model import TypesafeModel
    return TypesafeModel()


def _perplexity():
    from .perplexity_model import PerplexityModel
    return PerplexityModel()


REGISTRY = {
    "market": MarketPriceModel,
    "coinflip": CoinFlipModel,
    "openai": _openai,
    "typesafe": _typesafe,
    "perplexity": _perplexity,
}


def load_models(names: list[str]) -> list[DecisionModel]:
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Available: {list(REGISTRY)}")
    return [REGISTRY[n]() for n in names]


__all__ = ["DecisionModel", "Prediction", "REGISTRY", "load_models"]
