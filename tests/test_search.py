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
        {"url": "https://www.cnbc.com/d", "title": "Prediction markets favor Democrats", "snippets": ["Traders now see a sweep."]},
    ]},
    "sources": {"https://www.reuters.com/a": {"age": ["October 27, 2026", "2026-10-27", "1 day ago"]}},
}


def test_clean_drops_prediction_markets():
    ctx = search.clean(BRAVE_RESPONSE)
    assert [c["source"] for c in ctx] == ["www.reuters.com", "notpolymarket.com"]
    assert ctx[0]["snippets"] == ["Democrats lead the generic ballot by 4 points."]
    assert ctx[0]["published"] == "2026-10-27" and ctx[1]["published"] == "unknown"


def test_query_respects_brave_limits():
    q = search.query_for("word " * 200)
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


def test_stored_search_is_reused_and_served(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from odds_arena.web.app import app

    monkeypatch.setattr(db, "DATABASE_URL", None)
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))
    m = Market(id="7", question="Q7?", description="Resolves Yes if X.", outcomes=["Yes", "No"], prices=[0.3, 0.7],
               end_date="2026-11-04T00:00:00Z", closed=False, volume=1e5, slug="q7")
    monkeypatch.setattr(jobs.polymarket, "fetch_open_markets", lambda **kw: [m])
    calls = []
    monkeypatch.setattr(jobs.search, "search_context", lambda q: calls.append(q) or [{"source": "a.com", "snippets": ["x"]}])

    first = Recorder("b+search", True)
    monkeypatch.setattr(jobs, "load_models", lambda names: [first])
    jobs.run_predict(model_names=["x"], log=lambda s: None)
    later = Recorder("c+search", True)  # a contestant added on a later run
    monkeypatch.setattr(jobs, "load_models", lambda names: [first, later])
    jobs.run_predict(model_names=["x"], log=lambda s: None)
    assert calls == ["Q7?"] and later.seen == first.seen  # same results, no second search

    with db.connect() as conn:
        conn.execute("UPDATE markets SET outcome = 1 WHERE id = '7'")
    c = TestClient(app)
    d = c.get("/api/markets/7").json()
    assert d["description"] == "Resolves Yes if X." and d["slug"] == "q7" and d["outcome"] == 1
    assert d["search"]["query"] == "Q7?" and d["search"]["results"] == [{"source": "a.com", "snippets": ["x"]}]
    assert [(p["model"], p["probability"], p["brier"]) for p in d["predictions"]] == [("b+search", 0.5, 0.25), ("c+search", 0.5, 0.25)]
    assert c.get("/api/markets/nope").status_code == 404


def test_history_endpoint_caches_and_validates(monkeypatch):
    from fastapi.testclient import TestClient
    from odds_arena.web import app as web

    web._history_cache.clear(); web._token_cache.clear()
    calls = []
    monkeypatch.setattr(web.polymarket, "fetch_yes_token", lambda mid: calls.append(("token", mid)) or "tok")
    monkeypatch.setattr(web.polymarket, "fetch_price_history",
                        lambda tok, r: calls.append(("hist", tok, r)) or [(1, 0.4), (2, 0.6)])
    c = TestClient(web.app)
    assert c.get("/api/markets/9/history?range=1d").json() == {"range": "1d", "points": [[1, 0.4], [2, 0.6]]}
    c.get("/api/markets/9/history?range=1d")
    c.get("/api/markets/9/history?range=1m")
    assert calls == [("token", "9"), ("hist", "tok", "1d"), ("hist", "tok", "1m")]  # cached, token reused
    assert c.get("/api/markets/9/history?range=5y").status_code == 400

    monkeypatch.setattr(web.polymarket, "fetch_yes_token", lambda mid: (_ for _ in ()).throw(RuntimeError("404")))
    assert c.get("/api/markets/nope/history").status_code == 502


def test_markets_api_returns_outcomes_events_and_siblings(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from odds_arena.web.app import app

    monkeypatch.setattr(db, "DATABASE_URL", None)
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))
    mk = lambda i, outcomes, ev, grp, price=0.5: Market(id=i, question=f"Q{i}?", description="", outcomes=outcomes, prices=[price, 1 - price],
                                         end_date=None, closed=False, volume=1e5, event_id=ev, event_title="Which party?" if ev else None, group_title=grp)
    markets = [mk("d", ["Yes", "No"], "e1", "Democrats"), mk("r", ["Yes", "No"], "e1", "Republicans"), mk("g", ["Lakers", "Celtics"], None, None)]
    monkeypatch.setattr(jobs.polymarket, "fetch_open_markets", lambda **kw: markets)
    monkeypatch.setattr(jobs, "load_models", lambda names: [Recorder("a", False)])
    jobs.run_predict(model_names=["x"], log=lambda s: None)
    with db.connect() as conn:  # a row from before the outcomes column existed
        conn.execute("INSERT INTO markets (id, question) VALUES ('old', 'Old?')")

    c = TestClient(app)
    by_id = {m["id"]: m for m in c.get("/api/markets").json()}
    assert by_id["g"]["outcomes"] == ["Lakers", "Celtics"] and by_id["old"]["outcomes"] == ["Yes", "No"]
    assert (by_id["d"]["event_id"], by_id["d"]["event_title"], by_id["d"]["group_title"]) == ("e1", "Which party?", "Democrats")
    d = c.get("/api/markets/d").json()
    assert [s["group_title"] for s in d["siblings"]] == ["Republicans"] and c.get("/api/markets/g").json()["siblings"] == []


def test_settled_looking_markets_are_skipped(monkeypatch):
    from odds_arena import polymarket
    raw = lambda i, p: {"id": i, "question": "Q", "outcomes": '["A", "B"]', "outcomePrices": f'["{p}", "{1 - p}"]', "volume": "1e6"}
    class R:
        def raise_for_status(self): pass
        def json(self): return [raw("live", 0.6), raw("done", 0.995), raw("dead", 0.005)]
    monkeypatch.setattr(polymarket.httpx, "get", lambda *a, **k: R())
    assert [m.id for m in polymarket.fetch_open_markets(within_days=None)] == ["live"]
