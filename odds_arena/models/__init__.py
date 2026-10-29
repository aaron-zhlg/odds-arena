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


# Reference baselines that don't forecast, so they never get web results.
BASELINES = {"market", "coinflip"}
# Older configs named search variants "<model>+search"; every AI model searches now, so it's the same model.
SEARCH_SUFFIX = "+search"


def load_models(names: list[str]) -> list[DecisionModel]:
    """Every AI model forecasts with the question's web search results; baselines don't."""
    names = list(dict.fromkeys(n.removesuffix(SEARCH_SUFFIX) for n in names))
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Available: {list(REGISTRY)}")
    models = []
    for n in names:
        model = REGISTRY[n]()
        model.uses_search = n not in BASELINES
        models.append(model)
    return models


__all__ = ["DecisionModel", "Prediction", "REGISTRY", "load_models"]
