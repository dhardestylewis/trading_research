"""Sparse event-style policies (Branch 4).

Three policy types designed for rare, high-conviction entries:
  1. top_pct       — trade only when score is in top N% cross-sectionally
  2. threshold_sep — threshold + minimum gap between consecutive entries
  3. threshold_cd  — threshold + cooldown after each entry
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.diagnostics.robustness_grid import shift_predictions
from src.utils.logging import get_logger

log = get_logger("sparse_policy")


def top_pct_policy(
    probs: np.ndarray,
    pct: float,
) -> np.ndarray:
    """Trade only when score is in top *pct* percentile (e.g. 0.10 = top 10%)."""
    cutoff = np.nanquantile(probs, 1 - pct)
    return (probs >= cutoff).astype(float)


def threshold_separation_policy(
    probs: np.ndarray,
    threshold: float,
    min_gap: int,
) -> np.ndarray:
    """Threshold + minimum *min_gap* bars between consecutive entries."""
    positions = np.zeros(len(probs))
    last_entry = -min_gap - 1

    for i in range(len(probs)):
        if probs[i] > threshold and (i - last_entry) > min_gap:
            positions[i] = 1.0
            last_entry = i

    return positions


def threshold_cooldown_policy(
    probs: np.ndarray,
    threshold: float,
    cooldown: int,
) -> np.ndarray:
    """Threshold + *cooldown* bars of inactivity after each entry."""
    positions = np.zeros(len(probs))
    cooldown_until = -1

    for i in range(len(probs)):
        if i <= cooldown_until:
            continue
        if probs[i] > threshold:
            positions[i] = 1.0
            cooldown_until = i + cooldown

    return positions


def evaluate_sparse_policies(
    preds: pd.DataFrame,
    ret_col: str = "fwd_ret_1h",
    cost_bps: float = 15.0,
    bars_per_year: float = 365.25 * 24,
    top_pcts: list[float] = (0.10, 0.05, 0.025),
    sep_gaps: list[int] = (3, 6, 12),
    cooldowns: list[int] = (3, 6, 12),
    threshold: float = 0.55,
    delays: list[int] = (0, 1),
) -> pd.DataFrame:
    """Evaluate sparse event-style policies with delay robustness.

    Parameters
    ----------
    preds : DataFrame filtered to the target asset/model with y_pred_prob, fwd_ret, fold_id.
    ret_col : forward return column.
    top_pcts : fractions for top-percentile policy.
    sep_gaps : minimum bar gaps for separation policy.
    cooldowns : cooldown bars for cooldown policy.
    threshold : base threshold for separation and cooldown policies.
    delays : signal delays to test robustness.

    Returns one row per (policy_name, param, delay) with metrics.
    """
    cost = cost_bps / 10_000.0
    af = bars_per_year
    rows: list[dict] = []

    for delay in delays:
        shifted = shift_predictions(preds, delay)
        shifted = shifted.sort_values("timestamp")

        probs = shifted["y_pred_prob"].values
        gross_all = shifted[ret_col].values
        fold_ids = shifted["fold_id"].values

        policies = []

        # Top-pct policies
        for pct in top_pcts:
            pos = top_pct_policy(probs, pct)
            policies.append((f"top_{pct*100:.1f}pct", pct, pos))

        # Separation policies
        for gap in sep_gaps:
            pos = threshold_separation_policy(probs, threshold, gap)
            policies.append((f"sep_{gap}bar_t{threshold}", gap, pos))

        # Cooldown policies
        for cd in cooldowns:
            pos = threshold_cooldown_policy(probs, threshold, cd)
            policies.append((f"cd_{cd}bar_t{threshold}", cd, pos))

        for policy_name, param, positions in policies:
            active = positions > 0
            if active.sum() < 5:
                continue

            gross = gross_all[active]
            net = gross - 2 * cost
            mean_net = np.nanmean(net)
            std_net = np.nanstd(net)
            sharpe = (mean_net / std_net * np.sqrt(af)) if std_net > 0 else 0.0

            cum = (1 + net).prod() - 1
            cum_series = pd.Series((1 + net).cumprod())
            dd = ((cum_series - cum_series.cummax()) / cum_series.cummax()).min()

            fold_s = pd.Series(fold_ids[active])
            fold_rets = pd.DataFrame({"fold_id": fold_s, "ret": gross}).groupby("fold_id")["ret"].mean()
            fold_prof = (fold_rets > 0).mean() if len(fold_rets) > 0 else 0.0

            rows.append({
                "policy": policy_name,
                "param": param,
                "delay_bars": delay,
                "sharpe": sharpe,
                "cumulative_return": cum,
                "max_drawdown": dd,
                "mean_net_return": mean_net,
                "return_per_trade": mean_net if active.sum() > 0 else 0.0,
                "hit_rate": (gross > 0).mean(),
                "trade_count": int(active.sum()),
                "fold_profitability": fold_prof,
            })

    return pd.DataFrame(rows)
