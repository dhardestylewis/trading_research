"""Branch C — Passive-entry feasibility.

Simple passive limit-order fill model:
  For each signal, quote a limit order at open, open-5bps, open-10bps.
  Assume fill only if next-bar low touches the limit price.
  Track fill rate, conditional Sharpe, missed-trade opportunity cost,
  and realized edge on filled trades.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger("passive_entry")


def passive_entry_study(
    panel: pd.DataFrame,
    preds: pd.DataFrame,
    limit_offsets_bps: list[float] = (0.0, 5.0, 10.0),
    threshold: float = 0.55,
    cost_bps: float = 15.0,
    bars_per_year: float = 365.25 * 24,
) -> pd.DataFrame:
    """Evaluate passive limit-order entry feasibility.

    Parameters
    ----------
    panel : OHLCV panel (filtered to SOL-USD already or will be filtered).
    preds : predictions with y_pred_prob, asset, timestamp, fwd_ret_1h.
    limit_offsets_bps : offsets below next-bar open for limit price.
        0 = limit at open, 5 = limit at open - 5 bps, etc.
    threshold : score threshold to define active signals.
    cost_bps : one-way cost in bps (applied to filled trades only,
        but reduced since passive → half-spread savings).

    Returns
    -------
    DataFrame with one row per limit_offset showing fill statistics.
    """
    cost = cost_bps / 10_000.0
    af = bars_per_year
    rows: list[dict] = []

    # Merge panel prices into preds
    merged_parts: list[pd.DataFrame] = []
    for asset in preds["asset"].unique():
        ap = panel[panel["asset"] == asset].sort_values("timestamp").copy()
        ap_preds = preds[preds["asset"] == asset].copy()

        if len(ap) < 3:
            continue

        ap["next_open"] = ap["open"].shift(-1)
        ap["next_low"] = ap["low"].shift(-1)
        ap["next_high"] = ap["high"].shift(-1)
        ap["next_close"] = ap["close"].shift(-1)

        merge_cols = ["timestamp", "next_open", "next_low", "next_high", "next_close"]
        m = ap_preds.merge(ap[merge_cols], on="timestamp", how="inner")
        merged_parts.append(m)

    if not merged_parts:
        return pd.DataFrame()

    merged = pd.concat(merged_parts, ignore_index=True)
    active = merged[merged["y_pred_prob"] > threshold].copy()

    if len(active) < 5:
        log.warning("Too few active signals (%d) for passive entry study", len(active))
        return pd.DataFrame()

    log.info("Passive entry study: %d active signals", len(active))

    for offset_bps in limit_offsets_bps:
        offset_frac = offset_bps / 10_000.0
        limit_price = active["next_open"] * (1 - offset_frac)

        # Fill condition: next-bar low ≤ limit price
        filled_mask = active["next_low"] <= limit_price

        total = len(active)
        n_filled = filled_mask.sum()
        fill_rate = n_filled / total if total > 0 else 0.0

        filled_df = active[filled_mask].copy()
        missed_df = active[~filled_mask].copy()

        # Returns for filled trades: entry at limit, exit at next close
        if n_filled > 0:
            entry_price = limit_price[filled_mask]
            exit_price = filled_df["next_close"]
            gross_ret = (exit_price / entry_price - 1).values

            # Passive fill → reduced cost (maker rebate assumption)
            # Use cost_bps * 0.5 for passive (one side maker)
            passive_cost = cost * 0.5  # half the cost for passive entry
            net_ret = gross_ret - 2 * passive_cost

            mean_net = np.nanmean(net_ret)
            std_net = np.nanstd(net_ret)
            sharpe = (mean_net / std_net * np.sqrt(af)) if std_net > 0 else 0.0

            cum = (1 + net_ret).prod() - 1
            cum_series = pd.Series((1 + net_ret).cumprod())
            dd = ((cum_series - cum_series.cummax()) / cum_series.cummax()).min()
            hit_rate = (gross_ret > 0).mean()

            # Fold profitability
            if "fold_id" in filled_df.columns:
                filled_df = filled_df.copy()
                filled_df["_gross_ret"] = gross_ret
                fold_rets = filled_df.groupby("fold_id")["_gross_ret"].mean()
                fold_prof = (fold_rets > 0).mean()
            else:
                fold_prof = np.nan
        else:
            mean_net = np.nan
            sharpe = 0.0
            cum = 0.0
            dd = 0.0
            hit_rate = 0.0
            fold_prof = 0.0

        # Missed-trade opportunity cost
        if len(missed_df) > 0 and "fwd_ret_1h" in missed_df.columns:
            missed_return = missed_df["fwd_ret_1h"].mean()
        else:
            missed_return = np.nan

        rows.append({
            "limit_offset_bps": offset_bps,
            "total_signals": total,
            "filled_count": int(n_filled),
            "fill_rate": fill_rate,
            "sharpe_filled": sharpe,
            "cumulative_return": cum,
            "max_drawdown": dd,
            "mean_net_return": mean_net,
            "hit_rate": hit_rate,
            "fold_profitability": fold_prof,
            "missed_opportunity_return": missed_return,
        })

    return pd.DataFrame(rows)
