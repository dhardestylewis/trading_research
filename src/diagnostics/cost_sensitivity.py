"""SOL-only cost sensitivity analysis (Branch 5).

Tests reduced-cost assumptions to determine whether the bottleneck is
signal half-life or pure cost drag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.diagnostics.robustness_grid import shift_predictions
from src.utils.logging import get_logger

log = get_logger("cost_sensitivity")


def cost_sensitivity_grid(
    preds: pd.DataFrame,
    ret_col: str = "fwd_ret_1h",
    cost_levels_bps: list[float] = (5.0, 7.5, 10.0, 15.0),
    threshold: float = 0.55,
    delays: list[int] = (0, 1),
    bars_per_year: float = 365.25 * 24,
) -> pd.DataFrame:
    """Run cost sensitivity grid for SOL-only predictions.

    Returns one row per (cost_bps, delay) with Sharpe, cumulative return,
    drawdown, hit rate, trade count.

    Diagnosis logic:
      - If Sharpe flips at similar cost for delay=0 and delay=1 → cost drag
      - If Sharpe flips much earlier with delay=1 → signal half-life
    """
    af = bars_per_year
    rows: list[dict] = []

    for delay in delays:
        shifted = shift_predictions(preds, delay)
        above = shifted[shifted["y_pred_prob"] > threshold]

        if len(above) < 5:
            log.warning("delay=%d: only %d trades above threshold — skipping", delay, len(above))
            continue

        gross = above[ret_col].values

        for cost_bps in cost_levels_bps:
            cost = cost_bps / 10_000.0
            net = gross - 2 * cost
            mean_net = np.nanmean(net)
            std_net = np.nanstd(net)
            sharpe = (mean_net / std_net * np.sqrt(af)) if std_net > 0 else 0.0

            cum = (1 + net).prod() - 1
            cum_series = pd.Series((1 + net).cumprod())
            dd = ((cum_series - cum_series.cummax()) / cum_series.cummax()).min()

            fold_rets = above.groupby("fold_id")[ret_col].mean()
            fold_prof = (fold_rets > 0).mean() if len(fold_rets) > 0 else 0.0

            rows.append({
                "cost_bps": cost_bps,
                "delay_bars": delay,
                "sharpe": sharpe,
                "cumulative_return": cum,
                "max_drawdown": dd,
                "mean_net_return": mean_net,
                "hit_rate": (gross > 0).mean(),
                "trade_count": len(above),
                "fold_profitability": fold_prof,
            })

    result = pd.DataFrame(rows)

    # Compute break-even cost (where Sharpe crosses zero at delay=0)
    delay0 = result[result["delay_bars"] == 0].sort_values("cost_bps")
    if len(delay0) >= 2:
        positive = delay0[delay0["sharpe"] > 0]
        if not positive.empty and len(positive) < len(delay0):
            be_cost = positive["cost_bps"].max()
            log.info("Break-even cost (delay=0): ~%.1f bps one-way", be_cost)

    return result
