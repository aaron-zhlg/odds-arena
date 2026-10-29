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


def test_search_context_goes_into_state():
    ctx = [{"source": "www.reuters.com", "title": "t", "published": "2026-10-27", "snippets": ["s"]}]
    req = decision_api.build_request(MARKET, "m", today=date(2026, 10, 28), context=ctx)
    assert req["state"]["search_results"] is ctx
    assert "search_results" not in decision_api.build_request(MARKET, "m")["state"]


NAMED = Market(id="2", question="Lakers vs. Celtics", description="Resolves to the winner.", outcomes=["Lakers", "Celtics"],
               prices=[0.55, 0.45], end_date="2026-11-01T23:59:00Z", closed=False, volume=1e5)


def test_named_outcomes_use_a_choice():
    req = decision_api.build_request(NAMED, "m", today=date(2026, 10, 28))
    q = req["questions"]["outcome"]
    assert q["type"] == "choice" and list(q["criteria"]) == ["Lakers", "Celtics"]
    assert req["state"]["outcomes"] == ["Lakers", "Celtics"] and "resolves_yes" not in req["questions"]
    # Yes/No markets keep the original Noul, unchanged
    assert decision_api.build_request(MARKET, "m")["questions"] == {"resolves_yes": decision_api.QUESTION}


def test_named_outcome_probability_is_first_option(monkeypatch):
    def fake_post(u, json, headers, timeout):
        return httpx.Response(200, json={"model": "v9", "answers": {"outcome": {"type": "choice", "choice": "Celtics",
                              "probabilities": {"Lakers": 0.3, "Celtics": 0.7}}}}, request=httpx.Request("POST", u))
    monkeypatch.setattr(decision_api.httpx, "post", fake_post)
    pred = PerplexityModel(api_key="k").predict(NAMED)
    assert pred.probability == 0.3 and pred.rationale.endswith("choice")


def test_event_choice_request_and_answer(monkeypatch):
    from odds_arena.polymarket import Event
    mk = lambda i, name: Market(id=i, question=f"{name}?", description="", outcomes=["Yes", "No"], prices=[0.5, 0.5],
                                end_date=None, closed=False, volume=1e5, event_id="e", group_title=name, neg_risk=True)
    ev = Event(id="e", title="Who wins?", description="rules", end_date="2026-11-04T00:00:00Z", options=[mk("a", "Ann"), mk("b", "Bob")])
    req = decision_api.build_event_request(ev, "m", today=date(2026, 10, 28))
    q = req["questions"]["winner"]
    assert q["type"] == "choice" and list(q["criteria"]) == ["Ann", "Bob"] and req["state"]["options"] == ["Ann", "Bob"]

    def fake_post(u, json, headers, timeout):
        return httpx.Response(200, json={"model": "v9", "answers": {"winner": {"type": "choice", "probabilities": {"Ann": 0.8, "Bob": 0.2}}}},
                              request=httpx.Request("POST", u))
    monkeypatch.setattr(decision_api.httpx, "post", fake_post)
    preds = TypesafeModel(api_key="k").predict_event(ev)
    assert {k: v.probability for k, v in preds.items()} == {"a": 0.8, "b": 0.2}
    assert "choice over 2 options" in preds["a"].rationale



def test_independent_options_request_and_answer(monkeypatch):
    from odds_arena.polymarket import Event
    mk = lambda i, q: Market(id=i, question=q, description="", outcomes=["Yes", "No"], prices=[0.5, 0.5], end_date=None,
                             closed=False, volume=1e5, event_id="e")
    ev = Event(id="e", title="Ceasefire through...?", description="rules", end_date=None,
               options=[mk("a", "Ceasefire through Oct 31?"), mk("b", "Ceasefire through Nov 15?")])
    req = decision_api.build_options_request(ev, "m", today=date(2026, 10, 28))
    assert req["questions"] == {"o0": {"type": "noul", "instructions": "Ceasefire through Oct 31?"},
                                "o1": {"type": "noul", "instructions": "Ceasefire through Nov 15?"}}

    def fake_post(u, json, headers, timeout):
        return httpx.Response(200, json={"model": "v9", "answers": {"o0": {"noul": 0.9}, "o1": {"noul": 0.6}}},
                              request=httpx.Request("POST", u))
    monkeypatch.setattr(decision_api.httpx, "post", fake_post)
    preds = PerplexityModel(api_key="k").predict_options(ev)
    assert {k: v.probability for k, v in preds.items()} == {"a": 0.9, "b": 0.6}
