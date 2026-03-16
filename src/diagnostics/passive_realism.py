"""Passive Realism — realistic passive-fill feasibility study.

Extends the simple passive_entry.py touch model with:
  - Touch probability: P(next-bar low ≤ limit)
  - Fill probability | touch: configurable queue haircut
  - Cancel probability: 1 - fill_prob
  - Adverse selection: mean return after fill vs random
  - Queue haircut sensitivity grid
  - Net expectancy under each assumption set

This is a first-class branch (not a nice-to-have) because
passive entry may be the only path to monetization given
that market-entry friction killed the prior branch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger("passive_realism")


def passive_realism_study(
    panel: pd.DataFrame,
    horizon: int = 1,
    offset_grid_bps: list[float] = (0.0, 5.0, 10.0, 15.0, 20.0),
    queue_haircuts: list[float] = (0.0, 0.25, 0.50),
    round_trip_cost_bps: float = 30.0,
    passive_cost_mult: float = 0.3,
    regime_col: str | None = None,
    regime_value: int = 1,
) -> pd.DataFrame:
    """Evaluate passive limit-order entry realism with queue haircuts.

    Parameters
    ----------
    panel : OHLCV panel with [asset, timestamp, open, high, low, close, volume].
    horizon : forward bars for exit.
    offset_grid_bps : limit price offsets below next-bar open.
    queue_haircuts : fraction of touches that are NOT filled (0 = all fill, 0.5 = half miss).
    round_trip_cost_bps : baseline market round-trip cost.
    passive_cost_mult : fraction of market cost applied to passive fills
                        (e.g. 0.3 → passive pays 30% of market friction).
    regime_col : optional regime column to filter by.
    regime_value : value to filter regime_col == regime_value.

    Returns
    -------
    DataFrame with one row per (asset, offset, haircut) combination.
    """
    passive_cost_frac = round_trip_cost_bps / 10_000 * passive_cost_mult

    rows: list[dict] = []

    for asset, g in panel.groupby("asset", sort=False):
        g = g.sort_values("timestamp").copy()

        # Apply regime filter if specified
        if regime_col and regime_col in g.columns:
            g = g[g[regime_col] == regime_value]
            if len(g) < 20:
                continue

        # Compute entry/exit fields
        g["next_open"] = g["open"].shift(-1)
        g["next_low"] = g["low"].shift(-1)
        g["next_high"] = g["high"].shift(-1)
        g["exit_close"] = g["close"].shift(-horizon)

        # Forward return from close-to-close (baseline)
        g["fwd_ret_bps"] = (g["close"].shift(-horizon) / g["close"] - 1) * 10_000
        g = g.dropna(subset=["next_open", "next_low", "exit_close"])

        if len(g) < 20:
            continue

        for offset_bps in offset_grid_bps:
            offset_frac = offset_bps / 10_000
            limit_price = g["next_open"] * (1 - offset_frac)

            # Touch condition
            touched = g["next_low"] <= limit_price
            n_total = len(g)
            n_touched = touched.sum()
            touch_rate = n_touched / n_total if n_total > 0 else 0

            for haircut in queue_haircuts:
                # Simulate partial fills: randomly drop (haircut %) of touched trades
                if haircut > 0 and n_touched > 0:
                    np.random.seed(42)  # reproducible
                    n_filled = int(n_touched * (1 - haircut))
                    if n_filled < 1:
                        rows.append({
                            "asset": asset,
                            "offset_bps": offset_bps,
                            "queue_haircut": haircut,
                            "n_total": n_total,
                            "n_touched": int(n_touched),
                            "touch_rate": touch_rate,
                            "n_filled": 0,
                            "fill_rate": 0.0,
                            "fill_given_touch": 0.0,
                        })
                        continue

                    touched_idx = g.index[touched]
                    filled_idx = np.random.choice(touched_idx, size=n_filled, replace=False)
                    filled_mask = g.index.isin(filled_idx)
                else:
                    filled_mask = touched
                    n_filled = n_touched

                fill_rate = n_filled / n_total if n_total > 0 else 0
                fill_given_touch = n_filled / n_touched if n_touched > 0 else 0

                filled_df = g[filled_mask]
                not_filled_df = g[~filled_mask & touched]

                if n_filled == 0:
                    rows.append({
                        "asset": asset,
                        "offset_bps": offset_bps,
                        "queue_haircut": haircut,
                        "n_total": n_total,
                        "n_touched": int(n_touched),
                        "touch_rate": touch_rate,
                        "n_filled": 0,
                        "fill_rate": 0.0,
                        "fill_given_touch": 0.0,
                    })
                    continue

                # Returns for filled trades
                entry_price = filled_df["next_open"] * (1 - offset_frac)
                exit_price = filled_df["exit_close"]
                gross_ret_bps = (exit_price / entry_price - 1) * 10_000
                net_ret_bps = gross_ret_bps - (passive_cost_frac * 10_000)

                # Adverse selection: compare filled vs all-bar mean return
                all_bar_mean_ret = g["fwd_ret_bps"].mean()
                filled_mean_ret = gross_ret_bps.mean()
                adverse_sel_bps = filled_mean_ret - all_bar_mean_ret

                # Cancel analysis
                if len(not_filled_df) > 0:
                    cancel_ret = not_filled_df["fwd_ret_bps"].mean()
                else:
                    cancel_ret = np.nan

                rows.append({
                    "asset": asset,
                    "offset_bps": offset_bps,
                    "queue_haircut": haircut,
                    "n_total": n_total,
                    "n_touched": int(n_touched),
                    "touch_rate": touch_rate,
                    "n_filled": int(n_filled),
                    "fill_rate": fill_rate,
                    "fill_given_touch": fill_given_touch,
                    "gross_bps_mean": gross_ret_bps.mean(),
                    "gross_bps_median": gross_ret_bps.median(),
                    "gross_bps_p75": gross_ret_bps.quantile(0.75),
                    "net_bps_mean": net_ret_bps.mean(),
                    "net_bps_median": net_ret_bps.median(),
                    "adverse_sel_bps": adverse_sel_bps,
                    "cancel_mean_ret_bps": cancel_ret,
                    "hit_rate": (net_ret_bps > 0).mean(),
                    "net_positive": net_ret_bps.mean() > 0,
                })

    result = pd.DataFrame(rows)
    if not result.empty:
        log.info(
            "Passive realism: %d rows, %d net-positive",
            len(result), result.get("net_positive", pd.Series(dtype=bool)).sum(),
        )
    return result


def passive_realism_summary(result: pd.DataFrame) -> pd.DataFrame:
    """Aggregate passive realism results for reporting."""
    if result.empty:
        return result

    summary = result.groupby(["offset_bps", "queue_haircut"]).agg(
        assets=("asset", "nunique"),
        avg_touch_rate=("touch_rate", "mean"),
        avg_fill_rate=("fill_rate", "mean"),
        avg_gross_bps=("gross_bps_mean", "mean"),
        avg_net_bps=("net_bps_mean", "mean"),
        avg_adverse_sel=("adverse_sel_bps", "mean"),
        pct_net_positive=("net_positive", "mean"),
    ).reset_index()

    return summary.sort_values(["offset_bps", "queue_haircut"])
