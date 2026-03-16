"""Branch D — Signal-age decay curve.

Computes E[r | signal, Δ] for signal age Δ ∈ {0, 0.25, 0.5, 1.0, 2.0} bars.
Uses OHLC interpolation within the entry bar to approximate sub-bar entry timing.

Δ=0    : entry at bar close (baseline)
Δ=0.25 : entry ≈ 25% into next bar
Δ=0.5  : entry ≈ midpoint of next bar
Δ=1.0  : entry at next open (1-bar delay)
Δ=2.0  : entry at open+2 (2-bar delay)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger("signal_decay")


def _interpolated_entry_price(
    row: pd.Series,
    delta: float,
) -> float:
    """Estimate entry price at fractional bar offset from signal bar close.

    For delta in [0, 1): interpolates within the NEXT bar using OHLC.
    For delta >= 1: uses shifted open prices.
    """
    if delta == 0:
        return row["close"]  # entry at signal bar close

    if delta < 1:
        # Interpolate within next bar: linear between open and close
        # weighted by delta (0→open, 1→close)
        return row["next_open"] + delta * (row["next_close"] - row["next_open"])

    if delta == 1.0:
        return row["next_open"]

    if delta == 2.0:
        return row["open_plus2"]

    # Fractional > 1: interpolate within the 2nd bar
    frac = delta - 1.0
    return row["open_plus2"] + frac * (row.get("close_plus2", row["open_plus2"]) - row["open_plus2"])


def signal_decay_curve(
    panel: pd.DataFrame,
    preds: pd.DataFrame,
    deltas: list[float] = (0.0, 0.25, 0.5, 1.0, 2.0),
    threshold: float = 0.55,
    cost_bps: float = 15.0,
    exit_horizon_bars: int = 1,
    bars_per_year: float = 365.25 * 24,
) -> pd.DataFrame:
    """Compute the signal decay curve E[r | signal, Δ].

    Parameters
    ----------
    panel : OHLCV panel (will be filtered per asset).
    preds : predictions with y_pred_prob, asset, timestamp.
    deltas : signal-age offsets in bars.
    threshold : score threshold for active signals.
    exit_horizon_bars : bars after entry for exit (default: 1 bar).

    Returns
    -------
    DataFrame with [delta, mean_return, median_return, sharpe, trade_count].
    """
    cost = cost_bps / 10_000.0
    af = bars_per_year
    max_shift = int(max(deltas)) + exit_horizon_bars + 1

    # Merge panel data into preds
    merged_parts: list[pd.DataFrame] = []
    for asset in preds["asset"].unique():
        ap = panel[panel["asset"] == asset].sort_values("timestamp").copy()
        ap_preds = preds[preds["asset"] == asset].copy()

        if len(ap) < max_shift + 2:
            continue

        ap["next_open"] = ap["open"].shift(-1)
        ap["next_high"] = ap["high"].shift(-1)
        ap["next_low"] = ap["low"].shift(-1)
        ap["next_close"] = ap["close"].shift(-1)
        ap["open_plus2"] = ap["open"].shift(-2)
        ap["close_plus2"] = ap["close"].shift(-2)
        # Exit prices for different entry points
        ap["exit_close_1"] = ap["close"].shift(-1)  # exit 1 bar from signal
        ap["exit_close_2"] = ap["close"].shift(-2)  # exit 2 bars from signal
        ap["exit_close_3"] = ap["close"].shift(-3)  # exit 3 bars from signal

        merge_cols = [
            "timestamp", "close", "next_open", "next_high", "next_low",
            "next_close", "open_plus2", "close_plus2",
            "exit_close_1", "exit_close_2", "exit_close_3",
        ]
        m = ap_preds.merge(ap[merge_cols], on="timestamp", how="inner", suffixes=("", "_panel"))
        merged_parts.append(m)

    if not merged_parts:
        return pd.DataFrame()

    merged = pd.concat(merged_parts, ignore_index=True)

    # Resolve close column
    close_col = "close_panel" if "close_panel" in merged.columns else "close"
    merged["close"] = merged[close_col]

    active = merged[merged["y_pred_prob"] > threshold].copy()
    if len(active) < 5:
        log.warning("Too few active signals (%d) for decay curve", len(active))
        return pd.DataFrame()

    log.info("Signal decay curve: %d active signals, %d deltas", len(active), len(deltas))

    rows: list[dict] = []
    for delta in deltas:
        # Compute entry price for each signal
        entries: list[float] = []
        exits: list[float] = []

        for _, row in active.iterrows():
            try:
                entry = _interpolated_entry_price(row, delta)
            except (KeyError, TypeError):
                continue

            if np.isnan(entry) or entry <= 0:
                continue

            # Exit price: 1 bar after the entry bar
            # For delta < 1, entry is within next bar → exit at close of next bar
            # For delta = 1, entry at next open → exit at next close
            # For delta = 2, entry at open+2 → exit at close+2
            if delta < 1:
                exit_p = row.get("exit_close_1", row.get("next_close", np.nan))
            elif delta == 1.0:
                exit_p = row.get("exit_close_1", np.nan)
            elif delta == 2.0:
                exit_p = row.get("exit_close_2", np.nan)
            else:
                exit_p = row.get("exit_close_2", np.nan)

            if np.isnan(exit_p) or exit_p <= 0:
                continue

            entries.append(entry)
            exits.append(exit_p)

        if len(entries) < 5:
            continue

        entries_arr = np.array(entries)
        exits_arr = np.array(exits)
        gross = exits_arr / entries_arr - 1
        net = gross - 2 * cost

        mean_net = np.nanmean(net)
        median_net = np.nanmedian(net)
        std_net = np.nanstd(net)
        sharpe = (mean_net / std_net * np.sqrt(af)) if std_net > 0 else 0.0

        rows.append({
            "delta_bars": delta,
            "mean_return": mean_net,
            "median_return": median_net,
            "mean_gross_return": np.nanmean(gross),
            "sharpe": sharpe,
            "hit_rate": (gross > 0).mean(),
            "trade_count": len(entries),
        })

    return pd.DataFrame(rows)
