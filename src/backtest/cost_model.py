"""Cost model for backtest simulation."""
from __future__ import annotations

COST_REGIMES = {
    "zero":     {"one_way_bps": 0},
    "base":     {"one_way_bps": 15},
    "punitive": {"one_way_bps": 35},
}


def get_one_way_cost(regime: str, overrides: dict | None = None) -> float:
    """Return one-way cost as a fraction (not bps)."""
    costs = dict(COST_REGIMES)
    if overrides:
        costs.update(overrides)
    bps = costs[regime]["one_way_bps"]
    return bps / 10_000.0
