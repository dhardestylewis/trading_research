"""Execution robustness analysis (Branch E).

Evaluate strategy under a delay × cost × fill-mode grid to detect
fragility in the signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger("robustness_grid")


def shift_predictions(
    preds: pd.DataFrame,
    delay_bars: int,
    prob_col: str = "y_pred_prob",
) -> pd.DataFrame:
    """Shift predictions by *delay_bars* within each (model, asset, fold_id).

    The forward return column is NOT shifted — it stays aligned to the
    execution bar, simulating the scenario where the signal is stale by
    *delay_bars*.
    """
    if delay_bars == 0:
        return preds.copy()

    parts: list[pd.DataFrame] = []
    for key, grp in preds.groupby(["model_name", "asset", "fold_id"]):
        g = grp.sort_values("timestamp").copy()
        g[prob_col] = g[prob_col].shift(delay_bars)
        parts.append(g)

    shifted = pd.concat(parts, ignore_index=True).dropna(subset=[prob_col])
    return shifted


def robustness_grid(
    preds: pd.DataFrame,
    delays: list[int] = (0, 1, 2),
    cost_regimes: dict[str, float] | None = None,
    fill_modes: dict[str, dict] | None = None,
    threshold: float = 0.55,
    ret_col: str = "fwd_ret_1h",
    bars_per_year: float = 365.25 * 24,
) -> pd.DataFrame:
    """Run the full delay × cost × fill robustness grid.

    Returns one row per (model, delay, cost_regime, fill_mode) with
    Sharpe, cumulative return, drawdown, trade count.
    """
    if cost_regimes is None:
        cost_regimes = {"zero": 0, "base": 15, "mild_worse": 20, "punitive": 35}
    if fill_modes is None:
        fill_modes = {
            "taker": {"cost_multiplier": 1.0, "slippage_bps": 0},
            "reduced_cost": {"cost_multiplier": 0.667, "slippage_bps": 0},
            "adverse_fill": {"cost_multiplier": 1.0, "slippage_bps": 5},
        }

    af = bars_per_year
    rows: list[dict] = []

    for delay in delays:
        shifted = shift_predictions(preds, delay)

        for model, mgrp in shifted.groupby("model_name"):
            above = mgrp[mgrp["y_pred_prob"] > threshold]
            if len(above) < 5:
                continue

            gross = above[ret_col].values

            for cost_name, cost_bps in cost_regimes.items():
                for fill_name, fill_cfg in fill_modes.items():
                    effective_cost_bps = cost_bps * fill_cfg["cost_multiplier"] + fill_cfg["slippage_bps"]
                    cost_frac = effective_cost_bps / 10_000.0
                    net = gross - 2 * cost_frac

                    mean_net = np.nanmean(net)
                    std_net = np.nanstd(net)
                    sharpe = (mean_net / std_net * np.sqrt(af)) if std_net > 0 else 0.0

                    cum = (1 + net).prod() - 1
                    cum_series = pd.Series((1 + net).cumprod())
                    dd = ((cum_series - cum_series.cummax()) / cum_series.cummax()).min()

                    rows.append({
                        "model_name": model,
                        "delay_bars": delay,
                        "cost_regime": cost_name,
                        "cost_bps": cost_bps,
                        "fill_mode": fill_name,
                        "effective_cost_bps": effective_cost_bps,
                        "sharpe": sharpe,
                        "cumulative_return": cum,
                        "max_drawdown": dd,
                        "mean_net_return": mean_net,
                        "hit_rate": (gross > 0).mean(),
                        "trade_count": len(above),
                    })

    return pd.DataFrame(rows)
