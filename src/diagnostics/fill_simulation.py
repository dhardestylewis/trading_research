"""Fill-type simulation for execution realism (Branch 2).

Simulate five realistic fill assumptions and compute metrics for each,
to determine whether the SOL signal survives under different execution
timing conventions.

Fill types:
  1. close_to_next_open  — existing convention (entry at close, measure to next close)
  2. next_bar_vwap       — VWAP proxy = (open + high + low + close) / 4
  3. next_bar_midpoint   — midpoint = (high + low) / 2
  4. delayed_open_1bar   — open price 2 bars forward (1-bar signal delay)
  5. delayed_open_2bar   — open price 3 bars forward (2-bar signal delay)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger("fill_simulation")


def _compute_fill_returns(
    panel: pd.DataFrame,
    preds: pd.DataFrame,
    fill_type: str,
    horizon_bars: int = 1,
) -> pd.DataFrame:
    """Compute adjusted forward returns for a given fill type.

    Uses a vectorized merge approach (not row-by-row iteration) for speed
    and to avoid timestamp type-mismatch issues.

    Parameters
    ----------
    panel : DataFrame with [asset, timestamp, open, high, low, close].
    preds : prediction rows (must contain asset, timestamp, y_pred_prob, model_name, fold_id).
    fill_type : one of the 5 fill types.
    horizon_bars : forward horizon in bars for the exit price.

    Returns
    -------
    preds with an added ``fill_ret`` column (the return under this fill assumption).
    """
    result_parts: list[pd.DataFrame] = []

    for asset in preds["asset"].unique():
        asset_panel = panel[panel["asset"] == asset].sort_values("timestamp").reset_index(drop=True)
        asset_preds = preds[preds["asset"] == asset].copy()

        if len(asset_panel) < horizon_bars + 3:
            continue

        # Build shifted price columns on the panel
        ap = asset_panel.copy()
        ap["next_open"] = ap["open"].shift(-1)
        ap["next_high"] = ap["high"].shift(-1)
        ap["next_low"] = ap["low"].shift(-1)
        ap["next_close"] = ap["close"].shift(-1)
        ap["exit_close"] = ap["close"].shift(-horizon_bars)
        # For delayed fills
        ap["open_plus2"] = ap["open"].shift(-2)
        ap["close_plus1_h"] = ap["close"].shift(-(1 + horizon_bars))
        ap["open_plus3"] = ap["open"].shift(-3)
        ap["close_plus2_h"] = ap["close"].shift(-(2 + horizon_bars))

        # Merge panel prices into preds on timestamp
        merge_cols = ["timestamp", "close", "next_open", "next_high", "next_low",
                      "next_close", "exit_close", "open_plus2", "close_plus1_h",
                      "open_plus3", "close_plus2_h"]
        ap_subset = ap[merge_cols].copy()

        merged = asset_preds.merge(ap_subset, on="timestamp", how="inner", suffixes=("", "_panel"))

        if len(merged) == 0:
            log.warning("  %s: no timestamp matches between panel and preds", asset)
            continue

        entry_close = merged["close_panel"] if "close_panel" in merged.columns else merged["close"]
        # Resolve the close column from the panel merge
        if "close_panel" in merged.columns:
            entry_close = merged["close_panel"]
        else:
            entry_close = merged["close"]

        if fill_type == "close_to_next_open":
            # Standard: entry at close, exit at close + horizon bars
            exit_price = merged["exit_close"]
            merged["fill_ret"] = exit_price / entry_close - 1

        elif fill_type == "next_bar_vwap":
            vwap = (merged["next_open"] + merged["next_high"] + merged["next_low"] + merged["next_close"]) / 4
            exit_price = merged["exit_close"]
            merged["fill_ret"] = exit_price / vwap - 1

        elif fill_type == "next_bar_midpoint":
            mid = (merged["next_high"] + merged["next_low"]) / 2
            exit_price = merged["exit_close"]
            merged["fill_ret"] = exit_price / mid - 1

        elif fill_type == "delayed_open_1bar":
            merged["fill_ret"] = merged["close_plus1_h"] / merged["open_plus2"] - 1

        elif fill_type == "delayed_open_2bar":
            merged["fill_ret"] = merged["close_plus2_h"] / merged["open_plus3"] - 1

        else:
            raise ValueError(f"Unknown fill_type: {fill_type}")

        # Keep only the original pred columns + fill_ret
        keep_cols = [c for c in preds.columns if c in merged.columns] + ["fill_ret"]
        result_parts.append(merged[keep_cols])

    if not result_parts:
        return preds.head(0).assign(fill_ret=np.nan)

    return pd.concat(result_parts, ignore_index=True).dropna(subset=["fill_ret"])


def fill_simulation_grid(
    panel: pd.DataFrame,
    preds: pd.DataFrame,
    thresholds: list[float] = (0.50, 0.55, 0.60),
    cost_bps: float = 15.0,
    bars_per_year: float = 365.25 * 24,
    fill_types: list[str] | None = None,
) -> pd.DataFrame:
    """Run the full fill-type simulation grid.

    Returns one row per (fill_type, threshold) with Sharpe, cumulative return,
    drawdown, hit rate, trade count, and fold profitability.
    """
    if fill_types is None:
        fill_types = [
            "close_to_next_open",
            "next_bar_vwap",
            "next_bar_midpoint",
            "delayed_open_1bar",
            "delayed_open_2bar",
        ]

    cost = cost_bps / 10_000.0
    af = bars_per_year
    rows: list[dict] = []

    for fill_type in fill_types:
        log.info("  Fill type: %s", fill_type)
        filled = _compute_fill_returns(panel, preds, fill_type)

        if len(filled) == 0:
            log.warning("  %s: no rows after fill computation", fill_type)
            continue

        for threshold in thresholds:
            above = filled[filled["y_pred_prob"] > threshold]
            if len(above) < 5:
                continue

            gross = above["fill_ret"].values
            net = gross - 2 * cost
            mean_net = np.nanmean(net)
            std_net = np.nanstd(net)
            sharpe = (mean_net / std_net * np.sqrt(af)) if std_net > 0 else 0.0

            cum = (1 + net).prod() - 1
            cum_series = pd.Series((1 + net).cumprod())
            dd = ((cum_series - cum_series.cummax()) / cum_series.cummax()).min()

            # Fold profitability
            if "fold_id" in above.columns:
                fold_rets = above.groupby("fold_id")["fill_ret"].mean()
                fold_prof = (fold_rets > 0).mean() if len(fold_rets) > 0 else 0.0
            else:
                fold_prof = np.nan

            rows.append({
                "fill_type": fill_type,
                "threshold": threshold,
                "sharpe": sharpe,
                "cumulative_return": cum,
                "max_drawdown": dd,
                "mean_net_return": mean_net,
                "hit_rate": (gross > 0).mean(),
                "trade_count": len(above),
                "fold_profitability": fold_prof,
            })

    return pd.DataFrame(rows)
