"""Perplexity Decisions API. Docs: https://docs.perplexity.ai/docs/decisions/quickstart"""
from .decision_api import DecisionAPIModel


class PerplexityModel(DecisionAPIModel):
    name = "perplexity"
    label = "Perplexity"
    api_url = "https://api.perplexity.ai/v1/decisions"
    api_key_env = "PERPLEXITY_API_KEY"
    model_env = "PERPLEXITY_MODEL"
    default_model = "pplx-decider-latest"
