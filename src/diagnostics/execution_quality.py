"""Execution quality metrics for paper-trade validation.

Computes the six metrics that bridge simulated and realized fills:

  1. Realized fill rate
  2. Realized slippage vs simulated
  3. Signed adverse selection after fill
  4. Cancel rate
  5. Missed-trade opportunity cost
  6. Realized return per filled trade

All functions accept a DataFrame in the paper-trade log schema
produced by paper_trade_logger.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger("execution_quality")


def realized_fill_rate(log_df: pd.DataFrame) -> dict:
    """Fraction of submitted orders that received a fill.

    Denominator = all orders submitted (after gating + spacing).
    Numerator = rows where cancel_status == 'filled' or 'partial_fill'.
    """
    n_submitted = len(log_df)
    if n_submitted == 0:
        return {"realized_fill_rate": np.nan, "submitted": 0, "filled": 0}

    filled_mask = log_df["cancel_status"].isin(["filled", "partial_fill"])
    n_filled = int(filled_mask.sum())
    return {
        "realized_fill_rate": n_filled / n_submitted,
        "submitted": n_submitted,
        "filled": n_filled,
    }


def realized_slippage_vs_simulated(log_df: pd.DataFrame) -> dict:
    """Mean(realized_fill_price − simulated_fill_price) in bps.

    Only computed over filled trades.
    Positive = realized fill was worse (higher) than simulated for longs.
    """
    filled = log_df[log_df["cancel_status"].isin(["filled", "partial_fill"])].copy()
    if filled.empty or "simulated_fill_price" not in filled.columns:
        return {"mean_slippage_bps": np.nan, "median_slippage_bps": np.nan}

    slip = (
        (filled["fill_price"] - filled["simulated_fill_price"])
        / filled["simulated_fill_price"]
        * 10_000
    )
    return {
        "mean_slippage_bps": float(slip.mean()),
        "median_slippage_bps": float(slip.median()),
        "std_slippage_bps": float(slip.std()),
        "max_slippage_bps": float(slip.max()),
        "min_slippage_bps": float(slip.min()),
        "slippage_values": slip,  # kept for plotting
    }


def signed_adverse_selection(
    log_df: pd.DataFrame,
    window_col: str = "midprice_after_15m",
) -> dict:
    """Mean signed price move against position within window of fill.

    For long-only: adverse = midprice moved DOWN after fill.
    Reported in bps. Positive = adverse (bad).
    """
    filled = log_df[log_df["cancel_status"].isin(["filled", "partial_fill"])].copy()
    if filled.empty or window_col not in filled.columns:
        return {"mean_adverse_selection_bps": np.nan}

    valid = filled.dropna(subset=["fill_price", window_col])
    if valid.empty:
        return {"mean_adverse_selection_bps": np.nan}

    # For longs: adverse = fill_price > midprice_after → price fell
    price_move_bps = (
        (valid[window_col] - valid["fill_price"])
        / valid["fill_price"]
        * 10_000
    )
    # Signed: negative = price fell (adverse for longs)
    adverse_bps = -price_move_bps  # positive = adverse

    return {
        "mean_adverse_selection_bps": float(adverse_bps.mean()),
        "median_adverse_selection_bps": float(adverse_bps.median()),
        "pct_adverse": float((adverse_bps > 0).mean()),
        "adverse_values": adverse_bps,
    }


def cancel_rate(log_df: pd.DataFrame) -> dict:
    """Cancelled or expired orders / total submitted orders."""
    n = len(log_df)
    if n == 0:
        return {"cancel_rate": np.nan, "cancelled": 0}

    cancelled = int((log_df["cancel_status"] == "cancelled").sum())
    partial = int((log_df["cancel_status"] == "partial_fill").sum())
    return {
        "cancel_rate": cancelled / n,
        "cancelled": cancelled,
        "partial_fills": partial,
        "total_submitted": n,
    }


def missed_trade_opportunity_cost(log_df: pd.DataFrame) -> dict:
    """PnL of signals that triggered but did not fill.

    Uses midprice_after_1h as the hypothetical exit to measure
    what would have been earned if the trade had filled.
    """
    missed = log_df[log_df["cancel_status"] == "cancelled"].copy()
    if missed.empty:
        return {
            "missed_count": 0,
            "missed_mean_hypothetical_return_bps": np.nan,
        }

    # Hypothetical return = (mid_after_1h - submitted_price) / submitted_price
    valid = missed.dropna(subset=["submitted_order_price", "midprice_after_1h"])
    if valid.empty:
        return {
            "missed_count": len(missed),
            "missed_mean_hypothetical_return_bps": np.nan,
        }

    hyp_ret = (
        (valid["midprice_after_1h"] - valid["submitted_order_price"])
        / valid["submitted_order_price"]
        * 10_000
    )
    return {
        "missed_count": len(missed),
        "missed_mean_hypothetical_return_bps": float(hyp_ret.mean()),
        "missed_hit_rate": float((hyp_ret > 0).mean()),
        "missed_total_return_bps": float(hyp_ret.sum()),
    }


def realized_return_per_filled_trade(log_df: pd.DataFrame) -> dict:
    """Net PnL per filled trade after costs."""
    filled = log_df[log_df["cancel_status"].isin(["filled", "partial_fill"])].copy()
    if filled.empty or "realized_pnl_at_horizon" not in filled.columns:
        return {"mean_return_per_trade_bps": np.nan}

    rets = filled["realized_pnl_at_horizon"].dropna()
    if rets.empty:
        return {"mean_return_per_trade_bps": np.nan}

    return {
        "mean_return_per_trade_bps": float(rets.mean() * 10_000),
        "median_return_per_trade_bps": float(rets.median() * 10_000),
        "hit_rate": float((rets > 0).mean()),
        "total_pnl": float(rets.sum()),
        "sharpe": float(
            rets.mean() / rets.std() * np.sqrt(252 * 24)
            if rets.std() > 0 else np.nan
        ),
    }


def shortfall_vs_simulated_entry(log_df: pd.DataFrame) -> dict:
    """Realized shortfall vs simulated entry expectation.

    Measures the signed difference between live fill price and the
    simulated entry price.  Positive = live fills were worse (paid more)
    than the backtest assumed.

    This is the key metric for distinguishing "live is merely noisier"
    from "live is structurally worse than the backtest execution model."
    """
    filled = log_df[log_df["cancel_status"].isin(["filled", "partial_fill"])].copy()
    if filled.empty or "realized_shortfall_vs_simulated" not in filled.columns:
        return {
            "shortfall_vs_simulated_bps": np.nan,
            "shortfall_std_bps": np.nan,
            "shortfall_max_bps": np.nan,
            "pct_structurally_worse": np.nan,
        }

    shortfall = filled["realized_shortfall_vs_simulated"].dropna()
    if shortfall.empty:
        return {
            "shortfall_vs_simulated_bps": np.nan,
            "shortfall_std_bps": np.nan,
            "shortfall_max_bps": np.nan,
            "pct_structurally_worse": np.nan,
        }

    return {
        "shortfall_vs_simulated_bps": float(shortfall.mean()),
        "shortfall_std_bps": float(shortfall.std()),
        "shortfall_max_bps": float(shortfall.max()),
        "pct_structurally_worse": float((shortfall > 0).mean()),
    }


def compute_all_metrics(log_df: pd.DataFrame) -> dict:
    """Compute all execution quality metrics.

    Returns a flat dict suitable for one row of a summary table.
    """
    result = {}

    fr = realized_fill_rate(log_df)
    result["realized_fill_rate"] = fr["realized_fill_rate"]
    result["submitted"] = fr["submitted"]
    result["filled"] = fr["filled"]

    slip = realized_slippage_vs_simulated(log_df)
    result["mean_slippage_bps"] = slip["mean_slippage_bps"]
    result["median_slippage_bps"] = slip["median_slippage_bps"]

    adv = signed_adverse_selection(log_df)
    result["mean_adverse_selection_bps"] = adv["mean_adverse_selection_bps"]

    cr = cancel_rate(log_df)
    result["cancel_rate"] = cr["cancel_rate"]

    missed = missed_trade_opportunity_cost(log_df)
    result["missed_count"] = missed["missed_count"]
    result["missed_mean_hyp_return_bps"] = missed["missed_mean_hypothetical_return_bps"]

    ret = realized_return_per_filled_trade(log_df)
    result["mean_return_per_trade_bps"] = ret["mean_return_per_trade_bps"]
    result["sharpe"] = ret.get("sharpe", np.nan)
    result["hit_rate"] = ret.get("hit_rate", np.nan)

    sf = shortfall_vs_simulated_entry(log_df)
    result["shortfall_vs_simulated_bps"] = sf["shortfall_vs_simulated_bps"]
    result["pct_structurally_worse"] = sf["pct_structurally_worse"]

    return result
