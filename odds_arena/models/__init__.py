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


SEARCH_SUFFIX = "+search"


def load_models(names: list[str]) -> list[DecisionModel]:
    """`<name>+search` enters the same model again, this time given web search context."""
    unknown = [n for n in names if n.removesuffix(SEARCH_SUFFIX) not in REGISTRY]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Available: {list(REGISTRY)} (each also as <name>{SEARCH_SUFFIX})")
    models = []
    for n in names:
        model = REGISTRY[n.removesuffix(SEARCH_SUFFIX)]()
        if n.endswith(SEARCH_SUFFIX):
            model.name, model.uses_search = n, True
        models.append(model)
    return models


__all__ = ["DecisionModel", "Prediction", "REGISTRY", "load_models"]
