from fastapi.testclient import TestClient

from odds_arena.web import app as web


def _client(monkeypatch, secret):
    if secret is None:
        monkeypatch.delenv("CRON_SECRET", raising=False)
    else:
        monkeypatch.setenv("CRON_SECRET", secret)
    monkeypatch.setattr(web.jobs, "run_predict", lambda **kw: {"ran": "predict"})
    monkeypatch.setattr(web.jobs, "run_resolve", lambda **kw: {"ran": "resolve"})
    return TestClient(web.app)


def test_cron_requires_matching_secret(monkeypatch):
    c = _client(monkeypatch, "s3cret")
    assert c.get("/api/cron/predict").status_code == 401
    assert c.get("/api/cron/predict", headers={"Authorization": "Bearer wrong"}).status_code == 401
    r = c.get("/api/cron/resolve", headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 200 and r.json() == {"ran": "resolve"}


def test_cron_refuses_when_no_secret_configured(monkeypatch):
    c = _client(monkeypatch, None)
    assert c.get("/api/cron/predict", headers={"Authorization": "Bearer "}).status_code == 401


def test_cron_reports_why_a_job_failed(monkeypatch):
    c = _client(monkeypatch, "s3cret")

    def boom(**kw):
        raise ValueError("Unknown model(s): ['opneai']")

    monkeypatch.setattr(web.jobs, "run_predict", boom)
    r = c.get("/api/cron/predict", headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 500 and r.json() == {"error": "ValueError: Unknown model(s): ['opneai']"}
