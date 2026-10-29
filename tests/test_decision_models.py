from datetime import date

import httpx
import pytest

from odds_arena.models import decision_api
from odds_arena.models.perplexity_model import PerplexityModel
from odds_arena.models.typesafe_model import TypesafeModel
from odds_arena.polymarket import Market

MARKET = Market(id="1", question="Will it rain in Paris on Nov 1?", description="Resolves Yes if Météo-France reports rain.",
                outcomes=["Yes", "No"], prices=[0.4, 0.6], end_date="2026-11-01T23:59:00Z", closed=False, volume=1e5)


def test_request_shape_and_no_price():
    req = decision_api.build_request(MARKET, "jev-latest", today=date(2026, 10, 28))
    assert req["state"] == {"question": MARKET.question, "resolution_rules": MARKET.description,
                            "closes": "2026-11-01", "today": "2026-10-28"}
    assert req["questions"]["resolves_yes"]["type"] == "noul"
    assert "0.4" not in str(req)  # the market price is never sent


@pytest.mark.parametrize("cls,url,model", [
    (TypesafeModel, "https://api.typesafe.ai/v1/systemone", "jev-latest"),
    (PerplexityModel, "https://api.perplexity.ai/v1/decisions", "pplx-decider-latest"),
])
def test_predict_reads_noul(monkeypatch, cls, url, model):
    monkeypatch.delenv(cls.model_env, raising=False)
    seen = {}

    def fake_post(u, json, headers, timeout):
        seen.update(url=u, auth=headers["Authorization"], body=json)
        return httpx.Response(200, json={"model": "v9", "answers": {"resolves_yes": {"type": "noul", "noul": 0.83}}},
                              request=httpx.Request("POST", u))

    monkeypatch.setattr(decision_api.httpx, "post", fake_post)
    pred = cls(api_key="k").predict(MARKET)
    assert pred.probability == 0.83 and "v9" in pred.rationale
    assert seen == {"url": url, "auth": "Bearer k", "body": decision_api.build_request(MARKET, model)}


def test_failed_call_raises_not_no(monkeypatch):
    monkeypatch.setattr(decision_api.httpx, "post", lambda u, **kw: httpx.Response(500, request=httpx.Request("POST", u)))
    with pytest.raises(httpx.HTTPStatusError):
        PerplexityModel(api_key="k").predict(MARKET)
