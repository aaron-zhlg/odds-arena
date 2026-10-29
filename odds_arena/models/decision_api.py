"""Decision models (TypeSafe Jev, Perplexity decider) as forecasters.

Both APIs take a `state` plus named typed questions and return a probability per question.
We ask one Noul per market: the probability that it resolves Yes. They don't see the market
price, same as the OpenAI model. The question wording lives in QUESTION, shared by every
decision model so their scores are comparable.
"""
from __future__ import annotations

import os
import time
from datetime import date

import httpx

from .base import DecisionModel, Prediction

QUESTION = {
    "type": "noul",
    "instructions": "Will `question` resolve Yes under `resolution_rules`?",
    "criteria": {
        "true": "The event in `question` happens by `closes`, so the market resolves Yes",
        "false": "The event in `question` has not happened by `closes`, so the market resolves No",
    },
}


def build_request(market, model: str, today: date | None = None, context: list[dict] | None = None) -> dict:
    req = {
        "model": model,
        "state": {
            "question": market.question,
            "resolution_rules": market.description,
            "closes": (market.end_date or "")[:10] or "unknown",
            "today": (today or date.today()).isoformat(),
        },
        "questions": {"resolves_yes": QUESTION},
    }
    if context is not None:
        req["state"]["search_results"] = context
    return req


class DecisionAPIModel(DecisionModel):
    """Subclasses set the endpoint, key and default model."""

    api_url: str
    api_key_env: str
    model_env: str
    default_model: str
    label: str

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv(self.model_env, self.default_model)
        self.api_key = api_key or os.environ[self.api_key_env]
        self.last_request: dict | None = None  # exact payload of the latest call, for debugging/rerun

    def _post(self, payload: dict, attempts: int = 3) -> dict:
        for attempt in range(attempts):
            r = httpx.post(self.api_url, json=payload, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=30)
            if r.status_code == 429 and attempt < attempts - 1:
                time.sleep(float(r.headers.get("retry-after", 2 ** attempt)))
                continue
            r.raise_for_status()  # a failed call is an error, never a "no"
            return r.json()
        raise RuntimeError("unreachable")

    def predict(self, market, context=None):
        self.last_request = build_request(market, self.model, context=context)
        data = self._post(self.last_request)
        p = float(data["answers"]["resolves_yes"]["noul"])
        return Prediction(p, f"{self.label} {data.get('model', self.model)} noul")
