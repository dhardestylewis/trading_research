"""Branch A — Strict reconciliation between exp003 Branch 2 and Branch 3.

Produces a side-by-side table comparing the *same* policy object
(SOL-only, LightGBM, threshold 0.55, delay 0) as materialised by:
  - Branch 2 (fill-simulation path using exp001 predictions)
  - Branch 3 (horizon-study path with SOL-only retraining)

Columns: prediction_count, active_trade_count, threshold_rule, fill_rule,
cost_rule, mean_score, mean_position, mean_turnover, fold_count.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.diagnostics.robustness_grid import shift_predictions
from src.utils.logging import get_logger

log = get_logger("branch_reconciliation")


def _describe_branch(
    preds: pd.DataFrame,
    *,
    label: str,
    threshold: float,
    delay: int,
    cost_bps: float,
    ret_col: str,
    fill_rule: str,
) -> dict:
    """Build a descriptor row for one branch materialisation."""
    shifted = shift_predictions(preds, delay)
    active = shifted[shifted["y_pred_prob"] > threshold]

    cost = cost_bps / 10_000.0
    gross = active[ret_col].values if len(active) > 0 else np.array([])
    net = gross - 2 * cost if len(gross) > 0 else np.array([])
    positions = (active["y_pred_prob"] > threshold).astype(float).values if len(active) > 0 else np.array([])

    # Turnover: binary positions so turnover = abs(diff(position))
    turnover = np.abs(np.diff(positions, prepend=0)).mean() if len(positions) > 0 else 0.0

    return {
        "branch": label,
        "prediction_count": len(shifted),
        "active_trade_count": len(active),
        "threshold_rule": f"> {threshold}",
        "fill_rule": fill_rule,
        "cost_rule": f"2 x {cost_bps} bps round-trip",
        "mean_score": shifted["y_pred_prob"].mean() if len(shifted) > 0 else np.nan,
        "mean_position": positions.mean() if len(positions) > 0 else 0.0,
        "mean_turnover": turnover,
        "fold_count": shifted["fold_id"].nunique() if "fold_id" in shifted.columns else 0,
        "mean_gross_return": np.nanmean(gross) if len(gross) > 0 else np.nan,
        "mean_net_return": np.nanmean(net) if len(net) > 0 else np.nan,
        "sharpe_net": (
            np.nanmean(net) / np.nanstd(net) * np.sqrt(365.25 * 24)
            if len(net) > 1 and np.nanstd(net) > 0 else 0.0
        ),
    }


def reconcile_branches(
    branch2_preds: pd.DataFrame,
    branch3_preds: pd.DataFrame,
    *,
    threshold: float = 0.55,
    delay: int = 0,
    cost_bps: float = 15.0,
    branch2_ret_col: str = "fwd_ret_1h",
    branch3_ret_col: str = "fwd_ret",
) -> pd.DataFrame:
    """Produce a reconciliation table comparing Branch 2 and Branch 3.

    Parameters
    ----------
    branch2_preds : exp001 predictions filtered to SOL / LightGBM
        (as used by exp003 Branch 2).
    branch3_preds : SOL-only retrained predictions
        (as produced by exp003 Branch 3 horizon=1 path).
    threshold, delay, cost_bps : matching policy parameters.

    Returns
    -------
    DataFrame with one row per branch and comparison columns.
    """
    log.info("Reconciling Branch 2 vs Branch 3 (τ=%.2f, delay=%d)", threshold, delay)

    row_b2 = _describe_branch(
        branch2_preds,
        label="Branch 2 — exp001 predictions (SOL timing audit)",
        threshold=threshold,
        delay=delay,
        cost_bps=cost_bps,
        ret_col=branch2_ret_col,
        fill_rule="close_to_next_open",
    )
    row_b3 = _describe_branch(
        branch3_preds,
        label="Branch 3 — SOL-only retrained (horizon 1h)",
        threshold=threshold,
        delay=delay,
        cost_bps=cost_bps,
        ret_col=branch3_ret_col,
        fill_rule="close_to_next_close (retrained label)",
    )

    df = pd.DataFrame([row_b2, row_b3])
    log.info("Reconciliation:\n%s", df.to_string(index=False))
    return df
