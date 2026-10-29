from __future__ import annotations

import os

import httpx

from .base import DecisionModel, Prediction


class TypesafeModel(DecisionModel):
    """Placeholder for the Typesafe decision model.

    TODO: replace the request/response shape below with Typesafe's real API.
    """

    name = "typesafe"

    def __init__(self):
        self.url = os.environ["TYPESAFE_API_URL"]
        self.api_key = os.getenv("TYPESAFE_API_KEY", "")

    def predict(self, market):
        r = httpx.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"question": market.question, "description": market.description, "outcomes": market.outcomes},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        return Prediction(float(data["probability"]), data.get("rationale", ""))
