"""Branch F — Regime-gated sparse event policy.

Combines sep_3bar_t0.55 with regime gates, evaluated under
multiple fill types to test fill-robustness improvement.

Variants:
  1. sep_3bar_t0.55                        (baseline)
  2. sep_3bar_t0.55 AND regime_vol_high
  3. sep_3bar_t0.55 AND regime_trend_down
  4. sep_3bar_t0.55 AND regime_drawdown
  5. sep_3bar_t0.55 AND NOT regime_rebound
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.sparse_policy import threshold_separation_policy
from src.diagnostics.fill_simulation import _compute_fill_returns
from src.utils.logging import get_logger

log = get_logger("regime_gated_policy")

# Gate definitions: (name, column, condition)
REGIME_GATES = [
    ("baseline", None, None),
    ("AND_vol_high", "regime_vol_high", "== 1"),
    ("AND_trend_down", "regime_trend_down", "== 1"),
    ("AND_drawdown", "regime_drawdown", "== 1"),
    ("NOT_rebound", "regime_rebound", "== 0"),
]


def _apply_gate(
    df: pd.DataFrame,
    gate_col: str | None,
    gate_cond: str | None,
) -> pd.DataFrame:
    """Apply a regime gate to filter a DataFrame."""
    if gate_col is None:
        return df
    if gate_col not in df.columns:
        log.warning("Gate column %s not in DataFrame — skipping", gate_col)
        return df.head(0)
    if gate_cond == "== 1":
        return df[df[gate_col] == 1]
    elif gate_cond == "== 0":
        return df[df[gate_col] == 0]
    return df


def regime_gated_policy_study(
    panel: pd.DataFrame,
    preds: pd.DataFrame,
    *,
    threshold: float = 0.55,
    sep_gap: int = 3,
    cost_bps: float = 15.0,
    fill_types: list[str] | None = None,
    bars_per_year: float = 365.25 * 24,
) -> pd.DataFrame:
    """Evaluate regime-gated sparse policies across multiple fill types.

    Parameters
    ----------
    panel : OHLCV panel (filtered to SOL-USD).
    preds : predictions with regime columns, y_pred_prob, etc.
    threshold, sep_gap : sparse policy parameters.
    cost_bps : one-way cost.
    fill_types : fill types to evaluate under.

    Returns
    -------
    DataFrame with one row per (gate, fill_type) showing metrics.
    """
    if fill_types is None:
        fill_types = ["close_to_next_open", "next_bar_midpoint", "next_bar_vwap"]

    cost = cost_bps / 10_000.0
    af = bars_per_year
    rows: list[dict] = []

    for gate_name, gate_col, gate_cond in REGIME_GATES:
        log.info("  Gate: %s", gate_name)
        gated = _apply_gate(preds, gate_col, gate_cond)

        if len(gated) < 10:
            log.warning("  Gate %s: only %d rows — skipping", gate_name, len(gated))
            continue

        # Apply sparse separation policy
        sorted_gated = gated.sort_values("timestamp")
        probs = sorted_gated["y_pred_prob"].values
        positions = threshold_separation_policy(probs, threshold, sep_gap)
        active_mask = positions > 0

        if active_mask.sum() < 5:
            continue

        active_df = sorted_gated[active_mask].copy()

        for fill_type in fill_types:
            if fill_type == "close_to_next_open":
                # Use fwd_ret_1h directly (baseline fill)
                ret_col = "fwd_ret_1h"
                if ret_col not in active_df.columns:
                    continue
                gross = active_df[ret_col].values
            else:
                # Compute fill returns via fill_simulation
                filled = _compute_fill_returns(panel, active_df, fill_type)
                if len(filled) < 5:
                    continue
                gross = filled["fill_ret"].values

            net = gross - 2 * cost
            mean_net = np.nanmean(net)
            std_net = np.nanstd(net)
            sharpe = (mean_net / std_net * np.sqrt(af)) if std_net > 0 else 0.0

            cum = (1 + net).prod() - 1
            cum_series = pd.Series((1 + net).cumprod())
            dd = ((cum_series - cum_series.cummax()) / cum_series.cummax()).min()

            # Fold profitability
            source_df = active_df if fill_type == "close_to_next_open" else filled
            if "fold_id" in source_df.columns:
                ret_key = ret_col if fill_type == "close_to_next_open" else "fill_ret"
                fold_rets = source_df.groupby("fold_id")[ret_key].mean()
                fold_prof = (fold_rets > 0).mean() if len(fold_rets) > 0 else 0.0
            else:
                fold_prof = np.nan

            rows.append({
                "gate": gate_name,
                "fill_type": fill_type,
                "total_gated_signals": len(gated),
                "active_trades": int(active_mask.sum()),
                "evaluated_trades": len(gross),
                "sharpe": sharpe,
                "cumulative_return": cum,
                "max_drawdown": dd,
                "mean_net_return": mean_net,
                "hit_rate": (gross > 0).mean(),
                "fold_profitability": fold_prof,
            })

    return pd.DataFrame(rows)
