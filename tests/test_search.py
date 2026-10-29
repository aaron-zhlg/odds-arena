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


def test_ai_models_search_and_old_suffix_maps_to_the_same_model(monkeypatch):
    from odds_arena.models import REGISTRY
    from odds_arena.models.base import DecisionModel as DM

    class Fake(DM):
        name = "fake"
        def predict(self, market, context=None): ...

    monkeypatch.setitem(REGISTRY, "fake", Fake)
    models = load_models(["market", "fake", "fake+search"])
    assert [(m.name, m.uses_search) for m in models] == [("market", False), ("fake", True)]  # deduped, AI searches


def as_events(markets):
    """Group markets into Polymarket events as iter_hot_events yields them (a market alone is its own event)."""
    from odds_arena.polymarket import Event
    events = {}
    for m in markets:
        key = m.event_id or m.id
        events.setdefault(key, Event(id=key, title=m.event_title or m.question, description="", end_date=None, options=[]))
        events[key].options.append(m)
    return list(events.values())


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
    monkeypatch.setattr(jobs.polymarket, "iter_hot_events", lambda **kw: as_events(markets))
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
    monkeypatch.setattr(jobs.polymarket, "iter_hot_events", lambda **kw: as_events([m]))

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
    monkeypatch.setattr(jobs.polymarket, "iter_hot_events", lambda **kw: as_events([m]))
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


def test_resolve_stores_prices_and_history_reads_the_db(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from odds_arena.web.app import app

    monkeypatch.setattr(db, "DATABASE_URL", None)
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))
    with db.connect() as conn:
        conn.execute("INSERT INTO markets (id, question) VALUES ('9', 'Q?')")
    live = Market(id="9", question="Q?", description="", outcomes=["Yes", "No"], prices=[0.6, 0.4], end_date=None,
                  closed=False, volume=1e5, first_token="tok")
    monkeypatch.setattr(jobs.polymarket, "fetch_market", lambda mid: live)
    calls = []
    day = 86400

    def hist(tok, rng="1w", since=None):
        calls.append(rng if since is None else since)
        if since is not None:
            return [(since + 3599, 0.7)]
        return [(0, 0.2), (10 * day, 0.4)] if rng == "all" else [(10 * day, 0.4), (10 * day + 3600, 0.6)]

    monkeypatch.setattr(jobs.polymarket, "fetch_price_history", hist)
    assert jobs.run_resolve(log=lambda s: None)["prices"] == 4
    assert jobs.run_resolve(log=lambda s: None)["prices"] == 1
    # full history only the first time; then only points after the latest stored one
    assert calls == ["all", "1w", 10 * day + 3601]

    c = TestClient(app)
    after = 10 * day + 3600 + 3600
    assert c.get("/api/markets/9/history?range=all").json()["points"] == [[0, 0.2], [10 * day, 0.4], [10 * day + 3600, 0.6], [after, 0.7]]
    assert c.get("/api/markets/9/history?range=1d").json()["points"] == [[10 * day, 0.4], [10 * day + 3600, 0.6], [after, 0.7]]
    assert c.get("/api/markets/none/history").json()["points"] == []
    assert c.get("/api/markets/9/history?range=5y").status_code == 400
    listed = {m["id"]: m for m in c.get("/api/markets").json()}
    assert (listed["9"]["price_now"], listed["9"]["price_at"]) == (0.7, after)  # the latest stored price


def test_markets_api_returns_outcomes_events_and_siblings(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from odds_arena.web.app import app

    monkeypatch.setattr(db, "DATABASE_URL", None)
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))
    mk = lambda i, outcomes, ev, grp, price=0.5: Market(id=i, question=f"Q{i}?", description="", outcomes=outcomes, prices=[price, 1 - price],
                                         end_date=None, closed=False, volume=1e5, event_id=ev, event_title="Which party?" if ev else None, group_title=grp)
    markets = [mk("d", ["Yes", "No"], "e1", "Democrats"), mk("r", ["Yes", "No"], "e1", "Republicans"), mk("g", ["Lakers", "Celtics"], None, None)]
    monkeypatch.setattr(jobs.polymarket, "iter_hot_events", lambda **kw: as_events(markets))
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


def test_hot_events_are_ranked_and_capped(monkeypatch):
    from odds_arena import polymarket
    mk = lambda i, p: {"id": i, "question": f"{i}?", "outcomes": '["Yes", "No"]', "outcomePrices": f'["{p}", "{1 - p}"]', "volume": "1e6"}
    seen_params = []

    class R:
        def __init__(self, params): self.params = params
        def raise_for_status(self): pass
        def json(self):
            if self.params["offset"]:
                return []
            return [
                {"id": "hot", "title": "Hot", "tags": [{"slug": "sports"}], "markets": [mk("a", 0.6), mk("done", 0.995)]},
                {"id": "dead", "title": "Settled", "markets": [mk("x", 0.999)]},           # nothing left to forecast
                {"id": "warm", "title": "Warm", "markets": [mk("b", 0.3)]},
                {"id": "cool", "title": "Cool", "markets": [mk("c", 0.5)]},
            ]

    def fake_get(url, params, timeout):
        seen_params.append(params); return R(params)

    monkeypatch.setattr(polymarket.httpx, "get", fake_get)
    events = list(polymarket.iter_hot_events(n=2, within_days=None))
    assert [e.id for e in events] == ["hot", "warm"]                        # hottest first, settled one skipped, capped
    assert [o.id for o in events[0].options] == ["a"] and events[0].options[0].category == "Sports"
    assert seen_params[0]["order"] == "volume24hr" and len(seen_params) == 1


def test_predict_moves_on_to_markets_that_still_need_predictions(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DATABASE_URL", None)
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))
    ms = [Market(id=str(i), question=f"Q{i}?", description="", outcomes=["Yes", "No"], prices=[0.5, 0.5],
                 end_date=None, closed=False, volume=1e5) for i in range(5)]
    monkeypatch.setattr(jobs.polymarket, "iter_hot_events", lambda **kw: as_events(ms))
    monkeypatch.setattr(jobs, "load_models", lambda names: [Recorder("a", False)])
    runs = [jobs.run_predict(limit=2, model_names=["x"], log=lambda s: None) for _ in range(3)]
    assert [r["markets"] for r in runs] == [2, 2, 1] and sum(r["predictions"] for r in runs) == 5


def test_independent_options_are_one_call_per_model(monkeypatch, tmp_path):
    from odds_arena.models.base import Prediction as P
    monkeypatch.setattr(db, "DATABASE_URL", None)
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))
    opt = lambda i: Market(id=i, question=f"BTC above {i}k?", description="", outcomes=["Yes", "No"], prices=[0.5, 0.5],
                           end_date=None, closed=False, volume=1e5, event_id="btc", event_title="Bitcoin above ___?")
    monkeypatch.setattr(jobs.polymarket, "iter_hot_events", lambda **kw: as_events([opt("80"), opt("90"), opt("100")]))
    monkeypatch.setattr(jobs.search, "search_context", lambda q: [{"source": "x", "snippets": ["s"]}])

    class Batcher(Recorder):
        calls = 0
        def predict_options(self, ev, context=None):
            Batcher.calls += 1
            return {o.id: P(0.1 * (i + 1), "b") for i, o in enumerate(ev.options)}

    monkeypatch.setattr(jobs, "load_models", lambda names: [Batcher("x", True)])
    r = jobs.run_predict(model_names=["x"], log=lambda s: None)
    assert (r["markets"], r["predictions"], Batcher.calls) == (1, 3, 1)


def test_one_winner_event_is_forecast_as_a_unit(monkeypatch, tmp_path):
    from odds_arena.models.base import Prediction as P
    monkeypatch.setattr(db, "DATABASE_URL", None)
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))
    opt = lambda i, name, price: Market(id=i, question=f"Will {name} win?", description="", outcomes=["Yes", "No"],
                                        prices=[price, 1 - price], end_date=None, closed=False, volume=1e5,
                                        event_id="e1", event_title="Who wins?", group_title=name, neg_risk=True)
    dem, rep = opt("d", "Democrats", 0.9), opt("r", "Republicans", 0.1)
    event = jobs.polymarket.Event(id="e1", title="Who wins?", description="rules", end_date=None, options=[dem, rep])
    monkeypatch.setattr(jobs.polymarket, "iter_hot_events", lambda **kw: as_events([dem, rep]))  # one event with both options
    fetches = []
    monkeypatch.setattr(jobs.polymarket, "fetch_event", lambda eid: fetches.append(eid) or event)
    searches = []
    monkeypatch.setattr(jobs.search, "search_context", lambda q: searches.append(q) or [{"source": "x"}])

    class Chooser(Recorder):
        calls = 0
        def predict_event(self, ev, context=None):
            Chooser.calls += 1
            self.seen.append(context)
            return {"d": P(0.7, "c"), "r": P(0.3, "c")}

    chooser, per_option = Chooser("c+search", True), Recorder("a", False)
    monkeypatch.setattr(jobs, "load_models", lambda names: [chooser, per_option])
    r = jobs.run_predict(model_names=["x"], log=lambda s: None)
    assert r["markets"] == 1 and r["predictions"] == 4 and fetches == ["e1"]   # the event counted once
    assert Chooser.calls == 1 and searches == ["Who wins?"]                     # one call, one search for the event
    assert per_option.seen == [None, None]                                      # default: each option on its own
    with db.connect() as conn:
        got = {(x["market_id"], x["model"]): x["probability"] for x in conn.execute("SELECT * FROM predictions").fetchall()}
        assert got[("d", "c+search")] == 0.7 and got[("r", "c+search")] == 0.3
        assert conn.execute("SELECT COUNT(*) AS n FROM search_contexts").fetchone()["n"] == 2  # stored for each option
        assert {x["neg_risk"] for x in conn.execute("SELECT neg_risk FROM markets").fetchall()} == {1}
    assert jobs.run_predict(model_names=["x"], log=lambda s: None)["predictions"] == 0      # nothing left to do


def test_question_kind_rules():
    from odds_arena.polymarket import question_kind
    assert question_kind(["Yes", "No"], True, "e") == "one_winner"
    assert question_kind(["Yes", "No"], False, "e") == "yes_no"        # independent option of an event
    assert question_kind(["Yes", "No"], False, None) == "yes_no"
    assert question_kind(["Lakers", "Celtics"], False, None) == "matchup"


def test_questions_api_groups_filters_and_pages(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from odds_arena.web.app import app

    monkeypatch.setattr(db, "DATABASE_URL", None)
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))
    rows = [  # id, event, category, end, question, group
        ("h1", "house", "Politics", "2026-11-04", "Will Dems win the House?", "Democrats"),
        ("h2", "house", "Politics", "2026-11-04", "Will GOP win the House?", "Republicans"),
        ("s1", "nba", "Sports", "2026-10-30", "Lakers vs. Clippers", None),
        ("c1", "btc", "Crypto", "2026-10-31", "Bitcoin above 80k?", "80,000"),
        ("p2", "gov", "Politics", "2026-11-03", "Who wins California?", "Becerra"),
    ]
    with db.connect() as conn:
        for i, ev, cat, end, q, grp in rows:
            conn.execute("INSERT INTO markets (id, event_id, category, end_date, question, group_title, event_title) VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (i, ev, cat, end, q, grp, ev.upper()))
            conn.execute("INSERT INTO predictions (market_id, model, probability) VALUES (?, 'openai', 0.5)", (i,))
        conn.execute("UPDATE markets SET outcome = 1 WHERE id = 's1'")
    c = TestClient(app)

    allq = c.get("/api/questions?limit=10").json()
    assert allq["total"] == 4 and [q["key"] for q in allq["questions"]][-1] == "nba"      # 4 questions, settled last
    assert {q["key"]: len(q["items"]) for q in allq["questions"]}["house"] == 2            # options grouped
    assert [q["key"] for q in allq["questions"]][:3] in (["btc", "gov", "house"], ["gov", "btc", "house"])  # categories interleaved
    house = next(q for q in allq["questions"] if q["key"] == "house")
    assert house["items"][0]["predictions"]["openai"]["p"] == 0.5

    p1, p2 = (c.get(f"/api/questions?limit=2&offset={o}").json() for o in (0, 2))
    assert [q["key"] for q in p1["questions"] + p2["questions"]] == [q["key"] for q in allq["questions"]]  # pages tile
    assert [q["key"] for q in c.get("/api/questions?category=Politics").json()["questions"]] == ["gov", "house"]
    assert [q["key"] for q in c.get("/api/questions?q=lakers").json()["questions"]] == ["nba"]
    assert [q["key"] for q in c.get("/api/questions?q=republicans").json()["questions"]] == ["house"]  # option names searched
    assert [q["key"] for q in c.get("/api/questions?status=resolved").json()["questions"]] == ["nba"]
    assert c.get("/api/questions?status=open").json()["total"] == 3

    meta = c.get("/api/meta").json()
    assert {k: meta[k] for k in ("categories", "resolved", "models")} == {"categories": {"Politics": 2, "Sports": 1, "Crypto": 1}, "resolved": 1, "models": ["openai"]}
    assert [m["id"] for m in c.get("/api/markets?resolved=1").json()] == ["s1"]



def test_empty_search_means_no_ai_prediction_and_a_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DATABASE_URL", None)
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))
    m = Market(id="1", question="Q?", description="", outcomes=["Yes", "No"], prices=[0.5, 0.5], end_date=None, closed=False, volume=1e5)
    monkeypatch.setattr(jobs.polymarket, "iter_hot_events", lambda **kw: as_events([m]))
    results = [[], [{"source": "a.com", "snippets": ["x"]}]]  # first search finds nothing, the retry does
    monkeypatch.setattr(jobs.search, "search_context", lambda q: results.pop(0))
    ai, crowd = Recorder("ai", True), Recorder("crowd", False)
    monkeypatch.setattr(jobs, "load_models", lambda names: [crowd, ai])
    first = jobs.run_predict(model_names=["x"], log=lambda s: None)
    assert (first["predictions"], first["failed"], ai.seen) == (1, 1, [])          # only the crowd row; AI skipped
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM search_contexts").fetchone()["n"] == 0  # nothing stored
    second = jobs.run_predict(model_names=["x"], log=lambda s: None)
    assert (second["predictions"], second["failed"]) == (1, 0) and ai.seen == [[{"source": "a.com", "snippets": ["x"]}]]


def test_backfill_fills_open_questions_that_left_the_hot_list(monkeypatch, tmp_path):
    from odds_arena.polymarket import Event
    monkeypatch.setattr(db, "DATABASE_URL", None)
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))
    mk = lambda i, ev, end: Market(id=i, question=f"{i}?", description="", outcomes=["Yes", "No"], prices=[0.5, 0.5],
                                   end_date=end, closed=False, volume=1e5, event_id=ev, event_title=ev)
    with db.connect() as conn:  # predicted earlier by the crowd only; the AI model's call failed back then
        for i, ev, end in (("o", "open-ev", "2099-01-01"), ("c", "closed-ev", "2000-01-01")):
            conn.execute("INSERT INTO markets (id, question, event_id, end_date) VALUES (?, ?, ?, ?)", (i, f"{i}?", ev, end))
            conn.execute("INSERT INTO predictions (market_id, model, probability) VALUES (?, 'crowd', 0.5)", (i,))
    monkeypatch.setattr(jobs.polymarket, "iter_hot_events", lambda **kw: [])          # neither is hot any more
    fetched = []
    monkeypatch.setattr(jobs.polymarket, "fetch_event", lambda eid: fetched.append(eid) or
                        Event(id=eid, title=eid, description="", end_date=None, options=[mk("o", eid, "2099-01-01")]))
    monkeypatch.setattr(jobs.search, "search_context", lambda q: [{"source": "a", "snippets": ["x"]}])
    ai, crowd = Recorder("ai", True), Recorder("crowd", False)
    monkeypatch.setattr(jobs, "load_models", lambda names: [crowd, ai])
    r = jobs.run_predict(model_names=["x"], log=lambda s: None)
    assert fetched == ["open-ev"] and r["predictions"] == 1                       # the closed question is left alone
    with db.connect() as conn:
        assert conn.execute("SELECT model FROM predictions WHERE market_id = 'o' AND model = 'ai'").fetchone()
