from __future__ import annotations

import json
import os

from openai import OpenAI

from datetime import date

from .base import DecisionModel, Prediction

PROMPT = """You are a forecaster. Estimate the probability that the answer to this prediction-market question is "{yes}".

Question: {question}
Resolution details: {description}
Market closes: {end_date}
Today: {today}{search}

Reply with JSON only: {{"probability": <number between 0 and 1>, "rationale": "<one or two sentences>"}}"""


class OpenAIModel(DecisionModel):
    """Asks an OpenAI model for a probability. It does not see the market price."""

    name = "openai"

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI()

    def predict(self, market, context=None):
        resp = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": PROMPT.format(
                    yes=market.outcomes[0],
                    question=market.question,
                    description=market.description[:2000],
                    end_date=market.end_date,
                    today=date.today().isoformat(),
                    search="" if context is None else "\nWeb search results (JSON):\n" + json.dumps(context, ensure_ascii=False),
                ),
            }],
        )
        data = json.loads(resp.choices[0].message.content)
        p = min(max(float(data["probability"]), 0.0), 1.0)
        return Prediction(p, data.get("rationale", ""))
