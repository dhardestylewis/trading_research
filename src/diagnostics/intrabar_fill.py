"""Branch B — Intrabar entry approximation.

Finer execution proxies than exp003's whole-bar fill types.
Since we only have 1h bars, the 15-min granularity is *approximated*
from OHLC using proportional interpolation assumptions.

Fill types:
  1. next_15min_midpoint  — mid of first 15-min: open + 0.25*(close-open)
  2. next_15min_vwap      — OHLCV proxy of first 15-min
  3. first_15min_pullback  — min(open, open - pullback_frac * range)
  4. limit_touch_open_minus_xbps — fill at open-x bps if next-bar low ≤ that price
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger("intrabar_fill")


def _intrabar_fill_returns(
    panel: pd.DataFrame,
    preds: pd.DataFrame,
    fill_type: str,
    limit_bps: float = 5.0,
    pullback_frac: float = 0.25,
) -> pd.DataFrame:
    """Compute fill returns for one intrabar fill type.

    Returns preds augmented with `fill_ret` and `filled` (bool for
    conditional-fill types like limit-touch).
    """
    result_parts: list[pd.DataFrame] = []

    for asset in preds["asset"].unique():
        ap = panel[panel["asset"] == asset].sort_values("timestamp").copy()
        asset_preds = preds[preds["asset"] == asset].copy()

        if len(ap) < 4:
            continue

        # Build next-bar price columns
        ap["next_open"] = ap["open"].shift(-1)
        ap["next_high"] = ap["high"].shift(-1)
        ap["next_low"] = ap["low"].shift(-1)
        ap["next_close"] = ap["close"].shift(-1)
        ap["next_range"] = ap["next_high"] - ap["next_low"]

        merge_cols = [
            "timestamp", "close", "next_open", "next_high",
            "next_low", "next_close", "next_range",
        ]
        merged = asset_preds.merge(
            ap[merge_cols], on="timestamp", how="inner",
            suffixes=("", "_panel"),
        )
        if len(merged) == 0:
            continue

        entry_close = merged["close_panel"] if "close_panel" in merged.columns else merged["close"]
        exit_price = merged["next_close"]  # exit at end of next bar

        merged["filled"] = True  # default: always filled

        if fill_type == "next_15min_midpoint":
            # Approximate: first quarter of the bar
            entry = merged["next_open"] + 0.25 * (merged["next_close"] - merged["next_open"])
            merged["fill_ret"] = exit_price / entry - 1

        elif fill_type == "next_15min_vwap":
            # VWAP of first quarter ≈ lean toward open
            entry = (3 * merged["next_open"] + merged["next_high"] +
                     merged["next_low"] + merged["next_close"]) / 6
            merged["fill_ret"] = exit_price / entry - 1

        elif fill_type == "first_15min_pullback":
            # Entry at min(open, open - frac * range)
            pullback = merged["next_open"] - pullback_frac * merged["next_range"]
            entry = np.minimum(merged["next_open"].values, pullback.values)
            # Fill only if low ≤ entry (i.e. price actually reached there)
            merged["filled"] = merged["next_low"].values <= entry
            merged["fill_ret"] = exit_price / pd.Series(entry, index=merged.index) - 1

        elif fill_type.startswith("limit_touch_open_minus_"):
            # Passive buy limit below open: entry at open - x bps
            target = merged["next_open"] * (1 - limit_bps / 10_000.0)
            merged["filled"] = merged["next_low"].values <= target.values
            merged["fill_ret"] = exit_price / target - 1

        else:
            raise ValueError(f"Unknown intrabar fill_type: {fill_type}")

        keep = [c for c in preds.columns if c in merged.columns] + ["fill_ret", "filled"]
        result_parts.append(merged[keep])

    if not result_parts:
        return preds.head(0).assign(fill_ret=np.nan, filled=False)

    return pd.concat(result_parts, ignore_index=True).dropna(subset=["fill_ret"])


def intrabar_fill_grid(
    panel: pd.DataFrame,
    preds: pd.DataFrame,
    thresholds: list[float] = (0.55,),
    cost_bps: float = 15.0,
    limit_bps_levels: list[float] = (5.0, 10.0),
    bars_per_year: float = 365.25 * 24,
) -> pd.DataFrame:
    """Run the full intrabar fill simulation grid.

    Returns one row per (fill_type, threshold) with Sharpe, fill rate,
    trade count on filled trades, and conditional metrics.
    """
    cost = cost_bps / 10_000.0
    af = bars_per_year
    rows: list[dict] = []

    fill_types = [
        "next_15min_midpoint",
        "next_15min_vwap",
        "first_15min_pullback",
    ]
    # Add limit-touch variants
    for bps in limit_bps_levels:
        fill_types.append(f"limit_touch_open_minus_{bps:.0f}bps")

    for fill_type in fill_types:
        log.info("  Intrabar fill: %s", fill_type)

        # Parse limit_bps from name if applicable
        lbps = 5.0
        if "limit_touch" in fill_type:
            parts = fill_type.split("_")
            lbps = float(parts[-1].replace("bps", ""))

        filled_df = _intrabar_fill_returns(
            panel, preds, fill_type, limit_bps=lbps,
        )

        if len(filled_df) == 0:
            continue

        for threshold in thresholds:
            above = filled_df[filled_df["y_pred_prob"] > threshold]
            if len(above) < 5:
                continue

            total_signals = len(above)
            filled_only = above[above["filled"]] if "filled" in above.columns else above
            fill_rate = len(filled_only) / total_signals if total_signals > 0 else 0

            if len(filled_only) < 3:
                continue

            gross = filled_only["fill_ret"].values
            net = gross - 2 * cost
            mean_net = np.nanmean(net)
            std_net = np.nanstd(net)
            sharpe = (mean_net / std_net * np.sqrt(af)) if std_net > 0 else 0.0

            cum = (1 + net).prod() - 1
            cum_series = pd.Series((1 + net).cumprod())
            dd = ((cum_series - cum_series.cummax()) / cum_series.cummax()).min()

            # Fold profitability
            if "fold_id" in filled_only.columns:
                fold_rets = filled_only.groupby("fold_id")["fill_ret"].mean()
                fold_prof = (fold_rets > 0).mean() if len(fold_rets) > 0 else 0.0
            else:
                fold_prof = np.nan

            # Opportunity cost: mean return of missed signals
            missed = above[~above["filled"]] if "filled" in above.columns else above.head(0)
            if len(missed) > 0 and "fwd_ret_1h" in missed.columns:
                missed_mean = missed["fwd_ret_1h"].mean()
            else:
                missed_mean = np.nan

            rows.append({
                "fill_type": fill_type,
                "threshold": threshold,
                "total_signals": total_signals,
                "filled_trades": len(filled_only),
                "fill_rate": fill_rate,
                "sharpe": sharpe,
                "cumulative_return": cum,
                "max_drawdown": dd,
                "mean_net_return": mean_net,
                "hit_rate": (gross > 0).mean(),
                "fold_profitability": fold_prof,
                "missed_opportunity_return": missed_mean,
            })

    return pd.DataFrame(rows)
