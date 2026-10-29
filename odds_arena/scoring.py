"""Scoring rules. Lower Brier / log loss is better; higher accuracy is better."""
from __future__ import annotations

import math

EPS = 1e-6


def brier(p: float, outcome: int) -> float:
    return (p - outcome) ** 2


def log_loss(p: float, outcome: int) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return -math.log(p if outcome else 1 - p)


def leaderboard(rows) -> list[dict]:
    """rows: iterable of (model, probability, outcome) for resolved markets."""
    stats: dict[str, dict] = {}
    for model, p, outcome in rows:
        s = stats.setdefault(model, {"model": model, "n": 0, "brier": 0.0, "log_loss": 0.0, "correct": 0})
        s["n"] += 1
        s["brier"] += brier(p, outcome)
        s["log_loss"] += log_loss(p, outcome)
        s["correct"] += int((p >= 0.5) == bool(outcome))
    board = []
    for s in stats.values():
        n = s["n"]
        board.append({
            "model": s["model"],
            "resolved": n,
            "brier": s["brier"] / n,
            "log_loss": s["log_loss"] / n,
            "accuracy": s["correct"] / n,
        })
    return sorted(board, key=lambda r: r["brier"])
