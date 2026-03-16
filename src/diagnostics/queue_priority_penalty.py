"""Queue/priority penalty stress test.

Stress-tests passive-entry assumptions by:
  1. Haircutting fill probability (not all touch → fill)
  2. Worsening realized fill price by a fixed slippage penalty
  3. Re-computing strategy metrics under each penalty scenario

This bridges the gap between simulated touch-model fills and
realistic queue-position outcomes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger("queue_priority_penalty")


def _apply_fill_haircut(
    trades: pd.DataFrame,
    haircut: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Randomly drop (1 - haircut) fraction of trades to simulate queue misses."""
    if haircut >= 1.0:
        return trades.copy()
    n = len(trades)
    keep_mask = rng.random(n) < haircut
    return trades[keep_mask].copy()


def _apply_price_penalty(
    trades: pd.DataFrame,
    penalty_bps: float,
    net_return_col: str = "net_return",
    gross_return_col: str = "gross_return",
) -> pd.DataFrame:
    """Worsen realized fill price by adding penalty to costs."""
    out = trades.copy()
    penalty_frac = penalty_bps / 10_000
    if net_return_col in out.columns:
        out[net_return_col] = out[net_return_col] - penalty_frac
    if gross_return_col in out.columns:
        # gross stays the same — penalty is an execution cost
        pass
    return out


def _compute_metrics(trades: pd.DataFrame, net_return_col: str = "net_return") -> dict:
    """Compute strategy metrics on a trade DataFrame."""
    if trades.empty or net_return_col not in trades.columns:
        return {
            "trade_count": 0,
            "sharpe": np.nan,
            "cumulative_return": np.nan,
            "max_drawdown": np.nan,
            "mean_net_return": np.nan,
            "hit_rate": np.nan,
            "fold_profitability": np.nan,
        }

    rets = trades[net_return_col].values
    n = len(rets)
    mean_r = np.mean(rets)
    std_r = np.std(rets, ddof=1) if n > 1 else np.nan
    sharpe = (mean_r / std_r * np.sqrt(252 * 24)) if (std_r and std_r > 0) else np.nan

    cum = np.cumsum(rets)
    running_max = np.maximum.accumulate(cum)
    dd = cum - running_max
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

    hit_rate = float(np.mean(rets > 0))

    # Fold profitability
    fold_prof = np.nan
    if "fold_id" in trades.columns:
        fold_means = trades.groupby("fold_id")[net_return_col].mean()
        fold_prof = float((fold_means > 0).mean())

    return {
        "trade_count": n,
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "cumulative_return": float(cum[-1]) if len(cum) > 0 else 0.0,
        "max_drawdown": max_dd,
        "mean_net_return": float(mean_r),
        "hit_rate": hit_rate,
        "fold_profitability": fold_prof,
    }


def queue_penalty_study(
    trades: pd.DataFrame,
    fill_probability_haircuts: list[float] | None = None,
    fill_price_penalties_bps: list[float] | None = None,
    net_return_col: str = "net_return",
    gross_return_col: str = "gross_return",
    seed: int = 42,
    n_samples: int = 50,
) -> pd.DataFrame:
    """Run queue/priority penalty stress test.

    For each (haircut, price_penalty) combination, randomly subsample
    trades `n_samples` times to get robust statistics.

    Parameters
    ----------
    trades : pd.DataFrame
        Must contain `net_return_col` and optionally `fold_id`.
    fill_probability_haircuts : list of float
        Fraction of fills that actually execute (1.0 = no haircut).
    fill_price_penalties_bps : list of float
        Additional slippage in bps on top of simulated fill price.
    seed : int
        Random seed for reproducibility.
    n_samples : int
        Number of random subsamples per haircut level.

    Returns
    -------
    pd.DataFrame
        One row per (haircut, penalty) combination with averaged metrics.
    """
    if fill_probability_haircuts is None:
        fill_probability_haircuts = [1.0, 0.8, 0.6, 0.4]
    if fill_price_penalties_bps is None:
        fill_price_penalties_bps = [0.0, 2.0, 5.0]

    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for haircut in fill_probability_haircuts:
        for penalty_bps in fill_price_penalties_bps:
            sample_metrics: list[dict] = []

            for _ in range(n_samples if haircut < 1.0 else 1):
                sampled = _apply_fill_haircut(trades, haircut, rng)
                penalized = _apply_price_penalty(
                    sampled, penalty_bps,
                    net_return_col=net_return_col,
                    gross_return_col=gross_return_col,
                )
                m = _compute_metrics(penalized, net_return_col=net_return_col)
                sample_metrics.append(m)

            # Average across samples
            avg = {}
            for key in sample_metrics[0]:
                vals = [s[key] for s in sample_metrics if np.isfinite(s[key])]
                avg[key] = float(np.mean(vals)) if vals else np.nan

            rows.append({
                "fill_probability_haircut": haircut,
                "fill_price_penalty_bps": penalty_bps,
                "effective_fill_rate": haircut,
                **avg,
            })

    result = pd.DataFrame(rows)
    log.info(
        "Queue penalty study: %d scenarios, trades range %d–%d",
        len(result),
        int(result["trade_count"].min()) if not result.empty else 0,
        int(result["trade_count"].max()) if not result.empty else 0,
    )
    return result
