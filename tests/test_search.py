from odds_arena import db, jobs, search
from odds_arena.models import load_models
from odds_arena.models.base import DecisionModel, Prediction
from odds_arena.polymarket import Market

BRAVE_RESPONSE = {
    "grounding": {"generic": [
        {"url": "https://polymarket.com/event/x", "title": "Odds", "snippets": ["Yes 93%"]},
        {"url": "https://news.polymarket.com/a", "title": "PM news", "snippets": ["subdomain"]},
        {"url": "https://www.reuters.com/a", "title": "Poll", "snippets": [
            "Democrats lead the generic ballot by 4 points.",
            "On Polymarket, traders give Democrats a 93% chance.",
        ]},
        {"url": "https://example.com/b", "title": "Only odds", "snippets": ["Kalshi odds moved to 90%."]},
        {"url": "https://notpolymarket.com/c", "title": "Lookalike", "snippets": ["Turnout is expected to be high."]},
    ]},
    "sources": {"https://www.reuters.com/a": {"age": ["October 27, 2026", "2026-10-27", "1 day ago"]}},
}


def test_clean_drops_prediction_markets():
    ctx = search.clean(BRAVE_RESPONSE)
    assert [c["source"] for c in ctx] == ["www.reuters.com", "notpolymarket.com"]
    assert ctx[0]["snippets"] == ["Democrats lead the generic ballot by 4 points."]
    assert ctx[0]["published"] == "2026-10-27" and ctx[1]["published"] == "unknown"


def test_query_respects_brave_limits():
    q = search._query("word " * 200)
    assert len(q.split()) <= 75 and len(q) <= 600


def test_load_search_variant():
    plain, searched = load_models(["coinflip", "coinflip+search"])
    assert (plain.name, plain.uses_search) == ("coinflip", False)
    assert (searched.name, searched.uses_search) == ("coinflip+search", True)


class Recorder(DecisionModel):
    def __init__(self, name, uses_search):
        self.name, self.uses_search, self.seen = name, uses_search, []

    def predict(self, market, context=None):
        self.seen.append(context)
        return Prediction(0.5)


def test_one_search_per_market_shared_by_all_search_models(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DATABASE_URL", None)
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))
    markets = [Market(id=str(i), question=f"Q{i}?", description="", outcomes=["Yes", "No"], prices=[0.5, 0.5],
                      end_date=None, closed=False, volume=1e5) for i in range(2)]
    monkeypatch.setattr(jobs.polymarket, "fetch_open_markets", lambda **kw: markets)
    calls = []
    monkeypatch.setattr(jobs.search, "search_context", lambda q: calls.append(q) or [{"source": q}])
    models = [Recorder("a", False), Recorder("b+search", True), Recorder("c+search", True)]
    monkeypatch.setattr(jobs, "load_models", lambda names: models)

    r = jobs.run_predict(model_names=["x"], log=lambda s: None)
    assert r["predictions"] == 6 and calls == ["Q0?", "Q1?"]
    assert models[0].seen == [None, None]
    assert models[1].seen == models[2].seen == [[{"source": "Q0?"}], [{"source": "Q1?"}]]
    assert models[1].seen[0] is models[2].seen[0]  # literally the same object


def test_search_failure_skips_search_models_only(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DATABASE_URL", None)
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))
    m = Market(id="1", question="Q?", description="", outcomes=["Yes", "No"], prices=[0.5, 0.5], end_date=None, closed=False, volume=1e5)
    monkeypatch.setattr(jobs.polymarket, "fetch_open_markets", lambda **kw: [m])

    def boom(q):
        raise RuntimeError("brave down")

    monkeypatch.setattr(jobs.search, "search_context", boom)
    models = [Recorder("a", False), Recorder("b+search", True)]
    monkeypatch.setattr(jobs, "load_models", lambda names: models)
    r = jobs.run_predict(model_names=["x"], log=lambda s: None)
    assert r == {"markets": 1, "predictions": 1, "failed": 1, "timed_out": False}
    assert models[1].seen == []  # never asked to predict without its context
