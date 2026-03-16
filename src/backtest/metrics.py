"""Trading and forecast evaluation metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss


# ── Forecast metrics ────────────────────────────────────────────────

def forecast_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute forecast-quality metrics."""
    y_pred = (y_prob > 0.5).astype(int)
    m: dict = {}
    m["accuracy"] = accuracy_score(y_true, y_pred)
    try:
        m["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        m["roc_auc"] = np.nan
    m["brier"] = brier_score_loss(y_true, y_prob)
    m["hit_rate"] = (y_pred == y_true).mean()
    return m


# ── Trading metrics ─────────────────────────────────────────────────

def _annualization_factor(bars_per_year: float = 365.25 * 24) -> float:
    return bars_per_year


def trading_metrics(sim_df: pd.DataFrame, bars_per_year: float = 365.25 * 24) -> dict:
    """Compute trading metrics from a simulated trade log.

    Expects columns: net_ret, gross_ret, turnover, position, cumulative_net.
    """
    net = sim_df["net_ret"].values
    gross = sim_df["gross_ret"].values
    pos = sim_df["position"].values
    turn = sim_df["turnover"].values

    n = len(net)
    af = bars_per_year

    m: dict = {}

    # ── Return metrics ───────────────────────────────────────────
    cum = (1 + net).prod()
    m["cumulative_return"] = cum - 1
    m["annualized_return"] = cum ** (af / max(n, 1)) - 1

    vol = np.std(net) * np.sqrt(af) if n > 1 else 0
    m["annualized_vol"] = vol

    mean_ret = np.mean(net)
    m["sharpe"] = (mean_ret / np.std(net) * np.sqrt(af)) if np.std(net) > 0 else 0

    downside = net[net < 0]
    down_std = np.std(downside) if len(downside) > 1 else 1e-9
    m["sortino"] = (mean_ret / down_std * np.sqrt(af))

    # ── Drawdown ─────────────────────────────────────────────────
    cum_series = (1 + pd.Series(net)).cumprod()
    peak = cum_series.cummax()
    dd = (cum_series - peak) / peak
    m["max_drawdown"] = dd.min()
    m["calmar"] = m["annualized_return"] / abs(m["max_drawdown"]) if m["max_drawdown"] != 0 else 0

    # ── Trade stats ──────────────────────────────────────────────
    trades = np.where(turn > 0)[0]
    m["num_trades"] = len(trades)
    m["avg_turnover"] = turn.mean()
    m["total_turnover"] = turn.sum()
    m["exposure_fraction"] = pos.mean()

    # Per-trade returns (bars where position is active)
    active_mask = pos > 0
    if active_mask.sum() > 0:
        active_net = net[active_mask]
        m["avg_return_per_bar"] = active_net.mean()
        m["hit_rate_trades"] = (active_net > 0).mean()
        wins = active_net[active_net > 0]
        losses = active_net[active_net < 0]
        m["profit_factor"] = (wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else np.inf
    else:
        m["avg_return_per_bar"] = 0
        m["hit_rate_trades"] = 0
        m["profit_factor"] = 0

    # Return per unit turnover
    m["return_per_turnover"] = m["cumulative_return"] / max(m["total_turnover"], 1e-9)

    return m
