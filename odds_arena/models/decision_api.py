"""Decision models (TypeSafe Jev, Perplexity decider) as forecasters.

Both APIs take a `state` plus named typed questions and return a probability per question.
A Yes/No market gets one Noul (P resolves Yes); a market with two named outcomes ("Lakers" /
"Celtics") gets one Choice between them, and we keep P(first outcome). They don't see the market
price, same as the OpenAI model. The question wording lives here, shared by every decision model
so their scores are comparable.
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


def choice_question(outcomes: list[str]) -> dict:
    return {
        "type": "choice",
        "instructions": "Which of `outcomes` will `question` resolve to under `resolution_rules`?",
        "criteria": {o: f'The market resolves to "{o}"' for o in outcomes},
    }


def build_request(market, model: str, today: date | None = None, context: list[dict] | None = None) -> dict:
    yes_no = market.is_yes_no
    req = {
        "model": model,
        "state": {
            "question": market.question,
            "resolution_rules": market.description,
            "closes": (market.end_date or "")[:10] or "unknown",
            "today": (today or date.today()).isoformat(),
        },
        "questions": {"resolves_yes": QUESTION} if yes_no else {"outcome": choice_question(market.outcomes)},
    }
    if not yes_no:
        req["state"]["outcomes"] = market.outcomes
    if context is not None:
        req["state"]["search_results"] = context
    return req


def build_event_request(event, model: str, today: date | None = None, context: list[dict] | None = None) -> dict:
    """One Choice over every open option of a one-winner event, so the chances are weighed against each
    other and add up to 1 (asking about each option alone gives answers that don't)."""
    labels = [o.option_label for o in event.options]
    req = {
        "model": model,
        "state": {
            "event": event.title,
            "options": labels,
            "resolution_rules": event.description,
            "closes": (event.end_date or "")[:10] or "unknown",
            "today": (today or date.today()).isoformat(),
        },
        "questions": {"winner": {
            "type": "choice",
            "instructions": "Which of `options` will win `event` under `resolution_rules`?",
            "criteria": {label: f'"{label}" wins' for label in labels},
        }},
    }
    if context is not None:
        req["state"]["search_results"] = context
    return req


def build_options_request(event, model: str, today: date | None = None, context: list[dict] | None = None) -> dict:
    """One request, one Noul per option of an event whose options resolve independently (e.g. price
    ladders, date ladders). The state is sent once and the questions are answered in parallel."""
    req = {
        "model": model,
        "state": {
            "event": event.title,
            "resolution_rules": event.description,
            "closes": (event.end_date or "")[:10] or "unknown",
            "today": (today or date.today()).isoformat(),
        },
        "questions": {f"o{i}": {"type": "noul", "instructions": o.question} for i, o in enumerate(event.options)},
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
        if market.is_yes_no:
            p, kind = float(data["answers"]["resolves_yes"]["noul"]), "noul"
        else:
            p, kind = float(data["answers"]["outcome"]["probabilities"][market.outcomes[0]]), "choice"
        return Prediction(p, f"{self.label} {data.get('model', self.model)} {kind}")

    def predict_options(self, event, context=None):
        self.last_request = build_options_request(event, self.model, context=context)
        data = self._post(self.last_request)
        note = f"{self.label} {data.get('model', self.model)} noul"
        return {o.id: Prediction(float(data["answers"][f"o{i}"]["noul"]), note) for i, o in enumerate(event.options)}

    def predict_event(self, event, context=None):
        self.last_request = build_event_request(event, self.model, context=context)
        data = self._post(self.last_request)
        probs = data["answers"]["winner"]["probabilities"]
        note = f"{self.label} {data.get('model', self.model)} choice over {len(event.options)} options"
        return {o.id: Prediction(float(probs.get(o.option_label, 0.0)), note) for o in event.options}
