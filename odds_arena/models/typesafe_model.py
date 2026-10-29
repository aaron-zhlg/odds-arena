"""TypeSafe Jev. Docs: https://docs.typesafe.ai/go/api.md"""
from .decision_api import DecisionAPIModel


class TypesafeModel(DecisionAPIModel):
    name = "typesafe"
    label = "Jev"
    api_url = "https://api.typesafe.ai/v1/systemone"
    api_key_env = "TYPESAFE_API_KEY"
    model_env = "TYPESAFE_MODEL"
    default_model = "jev-latest"
