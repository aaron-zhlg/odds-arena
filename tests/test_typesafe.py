from datetime import date

import httpx
import pytest

from odds_arena.models import typesafe_model as ts
from odds_arena.polymarket import Market

MARKET = Market(id="1", question="Will it rain in Paris on Nov 1?", description="Resolves Yes if Météo-France reports rain.",
                outcomes=["Yes", "No"], prices=[0.4, 0.6], end_date="2026-11-01T23:59:00Z", closed=False, volume=1e5)


def test_request_shape_and_no_price():
    req = ts.build_request(MARKET, "jev-latest", today=date(2026, 10, 28))
    assert req["state"] == {"question": MARKET.question, "resolution_rules": MARKET.description,
                            "closes": "2026-11-01", "today": "2026-10-28"}
    assert req["questions"]["resolves_yes"]["type"] == "noul"
    assert "0.4" not in str(req)  # the market price is never sent


def test_predict_reads_noul(monkeypatch):
    seen = {}

    def fake_post(url, json, headers, timeout):
        seen.update(url=url, auth=headers["Authorization"], body=json)
        return httpx.Response(200, json={"model": "jev-1.14.0", "answers": {"resolves_yes": {"type": "noul", "noul": 0.83}}},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(ts.httpx, "post", fake_post)
    pred = ts.TypesafeModel(api_key="k").predict(MARKET)
    assert pred.probability == 0.83 and "jev-1.14.0" in pred.rationale
    assert seen["auth"] == "Bearer k" and seen["url"] == ts.API_URL


def test_failed_call_raises_not_no(monkeypatch):
    monkeypatch.setattr(ts.httpx, "post", lambda url, **kw: httpx.Response(500, request=httpx.Request("POST", url)))
    with pytest.raises(httpx.HTTPStatusError):
        ts.TypesafeModel(api_key="k").predict(MARKET)
