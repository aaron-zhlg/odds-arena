from odds_arena.polymarket import _to_market
from odds_arena.scoring import brier, leaderboard


def test_brier():
    assert brier(0.5, 1) == 0.25
    assert brier(1.0, 1) == 0.0


def test_leaderboard_ranks_by_brier():
    rows = [("good", 0.9, 1), ("good", 0.1, 0), ("flip", 0.5, 1), ("flip", 0.5, 0)]
    board = leaderboard(rows)
    assert [r["model"] for r in board] == ["good", "flip"]
    assert board[0]["accuracy"] == 1.0


def test_parse_gamma_market():
    m = _to_market({"id": 1, "question": "Q?", "outcomes": '["Yes","No"]', "outcomePrices": '["1","0"]', "closed": True})
    assert m.is_binary and m.resolved_outcome == 1
