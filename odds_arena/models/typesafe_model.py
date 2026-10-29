"""TypeSafe Jev (System One model) as a forecaster.

One Noul per market: the probability that the market resolves Yes. Jev doesn't see the market
price, same as the OpenAI model, so the comparison stays fair. Question wording lives here, in
QUESTION, so it can be audited and edited in one place.
Docs: https://docs.typesafe.ai/go/api.md, https://docs.typesafe.ai/primitives/noul.md
"""
from __future__ import annotations

import os
import time
from datetime import date

import httpx

from .base import DecisionModel, Prediction

API_URL = "https://api.typesafe.ai/v1/systemone"

QUESTION = {
    "type": "noul",
    "instructions": "Will `question` resolve Yes under `resolution_rules`?",
    "criteria": {
        "true": "The event in `question` happens by `closes`, so the market resolves Yes",
        "false": "The event in `question` has not happened by `closes`, so the market resolves No",
    },
}


def build_request(market, model: str, today: date | None = None) -> dict:
    return {
        "model": model,
        "state": {
            "question": market.question,
            "resolution_rules": market.description,
            "closes": (market.end_date or "")[:10] or "unknown",
            "today": (today or date.today()).isoformat(),
        },
        "questions": {"resolves_yes": QUESTION},
    }


class TypesafeModel(DecisionModel):
    name = "typesafe"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("TYPESAFE_MODEL", "jev-latest")
        self.api_key = api_key or os.environ["TYPESAFE_API_KEY"]
        self.last_request: dict | None = None  # exact payload of the latest call, for debugging/rerun

    def _post(self, payload: dict, attempts: int = 3) -> dict:
        for attempt in range(attempts):
            r = httpx.post(API_URL, json=payload, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=30)
            if r.status_code == 429 and attempt < attempts - 1:
                time.sleep(float(r.headers.get("retry-after", 2 ** attempt)))
                continue
            r.raise_for_status()  # a failed call is an error, never a "no"
            return r.json()
        raise RuntimeError("unreachable")

    def predict(self, market):
        self.last_request = build_request(market, self.model)
        data = self._post(self.last_request)
        p = float(data["answers"]["resolves_yes"]["noul"])
        return Prediction(p, f"Jev {data.get('model', self.model)} noul")
