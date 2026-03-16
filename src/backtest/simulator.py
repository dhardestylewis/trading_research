"""Backtest simulator: convert positions → net returns → trade log."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.signal_to_position import long_flat_threshold
from src.backtest.cost_model import get_one_way_cost
from src.utils.logging import get_logger

log = get_logger("simulator")


def simulate(
    df: pd.DataFrame,
    prob_col: str,
    threshold: float,
    cost_regime: str,
    asset_ret_col: str = "fwd_ret_1h",
    cost_overrides: dict | None = None,
) -> pd.DataFrame:
    """Run a bar-by-bar backtest for a single asset slice.

    Args:
        df: DataFrame with at least [timestamp, asset, prob_col, asset_ret_col].
        prob_col: column name holding predicted probability.
        threshold: decision threshold for long/flat.
        cost_regime: one of 'zero', 'base', 'punitive'.
        asset_ret_col: column with the actual forward return used for P&L.
        cost_overrides: optional dict to override cost regime values.

    Returns:
        df with extra columns: position, turnover, gross_ret, cost_paid, net_ret, cumulative_net.
    """
    one_way = get_one_way_cost(cost_regime, cost_overrides)

    result = df.copy()
    positions = long_flat_threshold(result[prob_col].values, threshold)
    result["position"] = positions

    # Turnover = |change in position|
    result["turnover"] = np.abs(np.diff(positions, prepend=0))

    # Gross return: position_{t-1} * realized return_t
    # We use the current row's fwd return with the position taken at this bar
    # Convention: position decided at bar t, return earned from t to t+h
    result["gross_ret"] = result["position"] * result[asset_ret_col]

    # Cost: each unit of turnover incurs one_way cost
    result["cost_paid"] = result["turnover"] * one_way

    # Net return
    result["net_ret"] = result["gross_ret"] - result["cost_paid"]

    # Cumulative
    result["cumulative_net"] = (1 + result["net_ret"]).cumprod()

    return result


def simulate_all(
    predictions_df: pd.DataFrame,
    prob_col: str,
    thresholds: list[float],
    cost_regimes: list[str],
    asset_ret_col: str = "fwd_ret_1h",
    cost_overrides: dict | None = None,
) -> pd.DataFrame:
    """Run backtests across all (threshold, cost_regime, asset) combinations.

    Returns a single DataFrame with identifying columns.
    """
    parts: list[pd.DataFrame] = []

    for threshold in thresholds:
        for cost_regime in cost_regimes:
            for asset, grp in predictions_df.groupby("asset", sort=False):
                grp_sorted = grp.sort_values("timestamp").copy()
                sim = simulate(
                    grp_sorted,
                    prob_col=prob_col,
                    threshold=threshold,
                    cost_regime=cost_regime,
                    asset_ret_col=asset_ret_col,
                    cost_overrides=cost_overrides,
                )
                sim["threshold"] = threshold
                sim["cost_regime"] = cost_regime
                parts.append(sim)

    combined = pd.concat(parts, ignore_index=True)
    log.info("Simulated %d threshold × %d cost × %d asset combos = %d rows",
             len(thresholds), len(cost_regimes),
             predictions_df["asset"].nunique(), len(combined))
    return combined
