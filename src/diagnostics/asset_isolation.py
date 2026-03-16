"""Asset-isolation analysis (Branch A).

Re-slice existing pooled predictions by asset and compare metrics.
Optionally re-train models per asset for isolated comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.metrics import trading_metrics
from src.utils.logging import get_logger

log = get_logger("asset_isolation")


def pooled_per_asset_metrics(
    preds: pd.DataFrame,
    ret_col: str = "fwd_ret_1h",
    cost_bps: float = 15.0,
    thresholds: list[float] = (0.50, 0.55, 0.60, 0.65),
    bars_per_year: float = 365.25 * 24,
) -> pd.DataFrame:
    """Compute full metrics per asset from *pooled* model predictions.

    Unlike the simple ``asset_metrics_table`` in exp001 (which only
    ran at τ=0.55, base cost), this evaluates the full threshold grid
    and includes fold-stability metrics.
    """
    cost = cost_bps / 10_000.0
    af = bars_per_year

    rows: list[dict] = []
    for (model, asset), grp in preds.groupby(["model_name", "asset"]):
        for tau in thresholds:
            above = grp[grp["y_pred_prob"] > tau]
            if len(above) < 5:
                continue

            gross = above[ret_col].values
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
                "asset_mode": asset,
                "model_name": model,
                "threshold": tau,
                "sharpe": sharpe,
                "cumulative_return": cum,
                "max_drawdown": dd,
                "hit_rate": (gross > 0).mean(),
                "trade_count": len(above),
                "fold_profitability": fold_prof,
                "fold_count": above["fold_id"].nunique(),
            })

    return pd.DataFrame(rows)


def compare_asset_modes(
    pooled_preds: pd.DataFrame,
    isolated_preds: dict[str, pd.DataFrame] | None = None,
    ret_col: str = "fwd_ret_1h",
    cost_bps: float = 15.0,
    threshold: float = 0.55,
    bars_per_year: float = 365.25 * 24,
) -> pd.DataFrame:
    """Compare pooled vs isolated per-asset performance.

    *isolated_preds* is a dict like ``{"BTC-USD": df, "ETH-USD": df, ...}``
    where each df has the same schema as pooled_preds but was trained
    on a single asset only.
    """
    cost = cost_bps / 10_000.0
    af = bars_per_year

    rows: list[dict] = []

    def _row(label, model, asset, vals):
        net = vals - 2 * cost
        mean_net = np.nanmean(net)
        std_net = np.nanstd(net)
        sharpe = (mean_net / std_net * np.sqrt(af)) if std_net > 0 else 0.0
        return {
            "training_mode": label,
            "model_name": model,
            "asset": asset,
            "sharpe": sharpe,
            "mean_net_return": mean_net,
            "hit_rate": (vals > 0).mean(),
            "trade_count": len(vals),
        }

    # Pooled
    for (model, asset), grp in pooled_preds.groupby(["model_name", "asset"]):
        above = grp[grp["y_pred_prob"] > threshold]
        if len(above) > 0:
            rows.append(_row("pooled", model, asset, above[ret_col].values))

    # Isolated
    if isolated_preds:
        for asset, df in isolated_preds.items():
            for model, mgrp in df.groupby("model_name"):
                above = mgrp[mgrp["y_pred_prob"] > threshold]
                if len(above) > 0:
                    rows.append(_row("isolated", model, asset, above[ret_col].values))

    return pd.DataFrame(rows)
