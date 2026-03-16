"""Regime-conditional performance analysis.

For each regime flag, compute conditional Sharpe, hit rate, trade count,
PnL contribution, and calendar-time fraction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.diagnostics.regime_labeller import REGIME_COLS
from src.utils.logging import get_logger

log = get_logger("regime_performance")


def regime_conditional_metrics(
    preds: pd.DataFrame,
    ret_col: str = "fwd_ret_1h",
    cost_bps: float = 15.0,
    bars_per_year: float = 365.25 * 24,
    threshold: float = 0.55,
) -> pd.DataFrame:
    """Compute per-regime-flag trading metrics on *active trades only*.

    *preds* must already have regime columns from ``regime_labeller``.

    Only rows where ``y_pred_prob > threshold`` are considered, so that
    regime Sharpe is truly conditional on the model's selected trades.

    Returns one row per (model_name, regime_flag) with:
    conditional_sharpe, conditional_hit_rate, trade_count,
    pnl_contribution_frac, calendar_time_frac.
    """
    cost = cost_bps / 10_000.0
    af = bars_per_year

    available_regimes = [c for c in REGIME_COLS if c in preds.columns]
    if not available_regimes:
        log.warning("No regime columns found in predictions DataFrame")
        return pd.DataFrame()

    rows: list[dict] = []
    for model, mgrp in preds.groupby("model_name"):
        # ── FIX: filter to active trades above threshold ──
        active = mgrp[mgrp["y_pred_prob"] > threshold]
        if len(active) < 10:
            log.warning("Model %s: only %d active trades — skipping regime metrics", model, len(active))
            continue

        gross = active[ret_col].values
        total_gross_pnl = gross.sum()

        for regime_col in available_regimes:
            mask = active[regime_col] == 1
            regime_grp = active[mask]
            n = len(regime_grp)
            if n < 10:
                continue

            rg = regime_grp[ret_col].values
            rg_net = rg - 2 * cost

            mean_net = np.nanmean(rg_net)
            std_net = np.nanstd(rg_net)
            sharpe = (mean_net / std_net * np.sqrt(af)) if std_net > 0 else 0.0

            rows.append({
                "model_name": model,
                "regime": regime_col,
                "conditional_sharpe": sharpe,
                "conditional_hit_rate": (rg > 0).mean(),
                "mean_net_return": mean_net,
                "trade_count": n,
                "pnl_contribution_frac": rg.sum() / total_gross_pnl if total_gross_pnl != 0 else np.nan,
                "calendar_time_frac": n / len(active),
            })

    return pd.DataFrame(rows)


def regime_gate_evaluation(
    preds: pd.DataFrame,
    thresholds: list[float] = (0.55, 0.60, 0.65),
    ret_col: str = "fwd_ret_1h",
    cost_bps: float = 15.0,
    bars_per_year: float = 365.25 * 24,
) -> pd.DataFrame:
    """Evaluate regime-gated policies.

    For each regime flag + threshold, compute Sharpe and fold stability
    of the gated strategy: trade only when regime=1 AND prob > threshold.
    """
    cost = cost_bps / 10_000.0
    af = bars_per_year

    available_regimes = [c for c in REGIME_COLS if c in preds.columns]
    rows: list[dict] = []

    for model, mgrp in preds.groupby("model_name"):
        for regime_col in available_regimes:
            for tau in thresholds:
                mask = (mgrp[regime_col] == 1) & (mgrp["y_pred_prob"] > tau)
                gated = mgrp[mask]
                if len(gated) < 10:
                    continue

                rg = gated[ret_col].values
                rg_net = rg - 2 * cost
                mean_net = np.nanmean(rg_net)
                std_net = np.nanstd(rg_net)
                sharpe = (mean_net / std_net * np.sqrt(af)) if std_net > 0 else 0.0

                fold_rets = gated.groupby("fold_id")[ret_col].mean()
                fold_prof = (fold_rets > 0).mean() if len(fold_rets) > 0 else 0.0

                # Asset concentration
                if "asset" in gated.columns:
                    comp = gated["asset"].value_counts(normalize=True)
                    max_asset = comp.max()
                else:
                    max_asset = np.nan

                rows.append({
                    "model_name": model,
                    "regime": regime_col,
                    "threshold": tau,
                    "sharpe": sharpe,
                    "mean_net_return": mean_net,
                    "trade_count": len(gated),
                    "fold_profitability": fold_prof,
                    "max_asset_share": max_asset,
                })

    return pd.DataFrame(rows)
