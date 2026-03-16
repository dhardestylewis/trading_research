"""Live fill validation metrics.

Computes execution-quality metrics that bridge simulated and
realized order fills.  Initially operates on simulated data
(exp004 predictions + penalty haircuts); later consumes live
order-fill logs.

Metrics:
  1. Realized fill rate vs simulated fill rate
  2. Realized slippage vs simulated midpoint/VWAP
  3. Post-fill adverse selection (price move against position)
  4. Cancel rate (signals triggered but limit never touched)
  5. Missed-trade opportunity cost (PnL of signaled-but-unfilled)
  6. Fold-level PnL stability under realized fills
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger("live_fill_validation")


def compute_fill_rate_comparison(
    simulated_fills: pd.DataFrame,
    realized_fills: pd.DataFrame | None = None,
    signal_col: str = "signal_active",
) -> dict:
    """Compare simulated vs realized fill rates.

    Parameters
    ----------
    simulated_fills : pd.DataFrame
        Must contain `signal_col` (bool) and `simulated_filled` (bool).
    realized_fills : pd.DataFrame or None
        If provided, must contain `signal_col` and `realized_filled` (bool).
        If None, returns simulated-only metrics.

    Returns
    -------
    dict with fill rate metrics.
    """
    signals = simulated_fills[signal_col].sum()
    sim_filled = simulated_fills.get("simulated_filled", pd.Series(dtype=bool)).sum()
    sim_fill_rate = sim_filled / signals if signals > 0 else np.nan

    result = {
        "total_signals": int(signals),
        "simulated_filled": int(sim_filled),
        "simulated_fill_rate": float(sim_fill_rate),
    }

    if realized_fills is not None and "realized_filled" in realized_fills.columns:
        real_filled = realized_fills["realized_filled"].sum()
        real_fill_rate = real_filled / signals if signals > 0 else np.nan
        result["realized_filled"] = int(real_filled)
        result["realized_fill_rate"] = float(real_fill_rate)
        result["fill_rate_gap"] = float(sim_fill_rate - real_fill_rate)

    return result


def compute_slippage(
    trades: pd.DataFrame,
    simulated_price_col: str = "simulated_fill_price",
    realized_price_col: str = "realized_fill_price",
    reference_price_col: str = "reference_price",
) -> pd.DataFrame:
    """Compute slippage statistics.

    Parameters
    ----------
    trades : pd.DataFrame
        Must contain price columns for comparison.

    Returns
    -------
    pd.DataFrame with slippage metrics per trade.
    """
    out = trades.copy()

    if simulated_price_col in out.columns and reference_price_col in out.columns:
        out["simulated_slippage_bps"] = (
            (out[simulated_price_col] - out[reference_price_col])
            / out[reference_price_col] * 10_000
        )

    if realized_price_col in out.columns and reference_price_col in out.columns:
        out["realized_slippage_bps"] = (
            (out[realized_price_col] - out[reference_price_col])
            / out[reference_price_col] * 10_000
        )

    if "simulated_slippage_bps" in out.columns and "realized_slippage_bps" in out.columns:
        out["slippage_gap_bps"] = out["realized_slippage_bps"] - out["simulated_slippage_bps"]

    return out


def compute_adverse_selection(
    trades: pd.DataFrame,
    fill_price_col: str = "fill_price",
    price_after_col: str = "price_15min_after_fill",
    direction_col: str = "direction",
) -> pd.DataFrame:
    """Compute post-fill adverse selection.

    Adverse selection = price movement against position within a
    lookback window after fill.

    Parameters
    ----------
    trades : pd.DataFrame
        Must contain fill price, post-fill price, and direction.

    Returns
    -------
    pd.DataFrame with adverse_selection_bps column added.
    """
    out = trades.copy()

    if fill_price_col in out.columns and price_after_col in out.columns:
        price_change = (out[price_after_col] - out[fill_price_col]) / out[fill_price_col]

        # For longs, adverse = price went down; for shorts, adverse = price went up
        if direction_col in out.columns:
            direction = out[direction_col].map({"long": 1, "short": -1}).fillna(1)
        else:
            direction = 1  # assume long-only

        out["adverse_selection_bps"] = -price_change * direction * 10_000
        out["adverse_selection_bps"] = out["adverse_selection_bps"].clip(lower=0)

    return out


def compute_cancel_metrics(
    signals: pd.DataFrame,
    signal_col: str = "signal_active",
    filled_col: str = "filled",
    limit_touched_col: str = "limit_touched",
) -> dict:
    """Compute cancel rate and related metrics.

    Cancel = signal triggered, limit order placed, but never touched/filled.

    Returns
    -------
    dict with cancel metrics.
    """
    active = signals[signals[signal_col]] if signal_col in signals.columns else signals
    n_signals = len(active)

    if n_signals == 0:
        return {"cancel_rate": np.nan, "total_signals": 0, "cancelled": 0}

    filled = active[filled_col].sum() if filled_col in active.columns else 0

    if limit_touched_col in active.columns:
        touched_not_filled = (active[limit_touched_col] & ~active[filled_col]).sum()
    else:
        touched_not_filled = 0

    cancelled = n_signals - filled
    cancel_rate = cancelled / n_signals if n_signals > 0 else np.nan

    return {
        "total_signals": int(n_signals),
        "filled": int(filled),
        "touched_not_filled": int(touched_not_filled),
        "cancelled": int(cancelled),
        "cancel_rate": float(cancel_rate),
    }


def compute_missed_opportunity_cost(
    signals: pd.DataFrame,
    filled_col: str = "filled",
    fwd_return_col: str = "fwd_ret",
) -> dict:
    """Compute PnL of trades we signaled but didn't fill.

    This measures the opportunity cost of missed fills.

    Returns
    -------
    dict with opportunity cost metrics.
    """
    if filled_col not in signals.columns or fwd_return_col not in signals.columns:
        return {"missed_opportunity_return": np.nan, "missed_count": 0}

    missed = signals[~signals[filled_col]]
    if missed.empty:
        return {"missed_opportunity_return": 0.0, "missed_count": 0}

    missed_ret = missed[fwd_return_col].values
    return {
        "missed_count": len(missed),
        "missed_opportunity_return": float(np.sum(missed_ret)),
        "missed_opportunity_mean_return": float(np.mean(missed_ret)),
        "missed_opportunity_hit_rate": float(np.mean(missed_ret > 0)),
    }


def compute_fold_pnl_stability(
    trades: pd.DataFrame,
    net_return_col: str = "net_return",
    fold_col: str = "fold_id",
) -> pd.DataFrame:
    """Compute fold-level PnL stability metrics.

    Returns
    -------
    pd.DataFrame with one row per fold: mean return, Sharpe, trade count.
    """
    if fold_col not in trades.columns or net_return_col not in trades.columns:
        return pd.DataFrame()

    groups = trades.groupby(fold_col)[net_return_col]
    fold_stats = groups.agg(["mean", "std", "count", "sum"]).reset_index()
    fold_stats.columns = [fold_col, "mean_return", "std_return", "trade_count", "total_return"]

    fold_stats["fold_sharpe"] = np.where(
        fold_stats["std_return"] > 0,
        fold_stats["mean_return"] / fold_stats["std_return"] * np.sqrt(252 * 24),
        np.nan,
    )
    fold_stats["profitable"] = fold_stats["mean_return"] > 0

    return fold_stats


def run_fill_validation_report(
    trades: pd.DataFrame,
    entry_mode: str,
    net_return_col: str = "net_return",
    fold_col: str = "fold_id",
) -> dict:
    """Aggregate all fill-validation metrics for one entry mode.

    Parameters
    ----------
    trades : pd.DataFrame
        Trade-level data with columns:
        - net_return, gross_return, fold_id (required)
        - filled, fwd_ret, simulated_filled (optional, for fill comparison)
        - fill_price, price_15min_after_fill (optional, for adverse selection)

    entry_mode : str
        Label for this entry mode configuration.

    Returns
    -------
    dict with all validation metrics.
    """
    result = {"entry_mode": entry_mode}

    # Basic trade stats
    if net_return_col in trades.columns:
        rets = trades[net_return_col].dropna()
        n = len(rets)
        mean_r = rets.mean()
        std_r = rets.std(ddof=1) if n > 1 else np.nan
        sharpe = (mean_r / std_r * np.sqrt(252 * 24)) if (std_r and std_r > 0) else np.nan

        result["trade_count"] = n
        result["sharpe"] = float(sharpe) if np.isfinite(sharpe) else np.nan
        result["mean_net_return"] = float(mean_r)
        result["hit_rate"] = float((rets > 0).mean())

    # Fold stability
    fold_df = compute_fold_pnl_stability(trades, net_return_col, fold_col)
    if not fold_df.empty:
        result["fold_count"] = len(fold_df)
        result["fold_profitability"] = float(fold_df["profitable"].mean())
        result["fold_sharpe_mean"] = float(fold_df["fold_sharpe"].mean())
        result["fold_sharpe_std"] = float(fold_df["fold_sharpe"].std())

    log.info("Fill validation [%s]: %d trades, Sharpe=%.2f",
             entry_mode, result.get("trade_count", 0), result.get("sharpe", np.nan))

    return result
