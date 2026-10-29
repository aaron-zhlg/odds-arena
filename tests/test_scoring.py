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


def test_sign_headers_verifies():
    import base64
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat
    from odds_arena.polymarket_us import load_private_key, sign_headers

    raw = Ed25519PrivateKey.generate().private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    # Secrets may carry the public key after the 32-byte seed; only the seed is used.
    key = load_private_key(base64.b64encode(raw + b"\0" * 32).decode())
    h = sign_headers("kid", key, "get", "/v1/portfolio/positions?limit=5", timestamp_ms=1700000000000)
    assert h["X-PM-Timestamp"] == "1700000000000"
    key.public_key().verify(base64.b64decode(h["X-PM-Signature"]), b"1700000000000GET/v1/portfolio/positions")
