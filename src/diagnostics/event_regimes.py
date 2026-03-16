"""Event-driven regime labelling for magnitude screening.

Extends the standard regime_labeller (12 flags based on vol/trend/calendar)
with structural market events that may produce larger gross moves:

  event_breakout           — price breaks N-bar high/low with volume expansion
  event_compression_expansion — volatility contracts then expands (squeeze)
  event_wick_reversal      — large wick relative to body on prior bar
  event_cascade_exhaustion — sharp drawdown + volume spike + reversal
  event_range_expansion    — bar range > 2× median range

These are designed to be used as regime slices in the gross-move atlas.
They should fire at low-moderate rates (typically 5–15% of bars).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger("event_regimes")


EVENT_REGIME_COLS = [
    "event_breakout",
    "event_compression_expansion",
    "event_wick_reversal",
    "event_cascade_exhaustion",
    "event_range_expansion",
]


def label_event_regimes(
    panel: pd.DataFrame,
    *,
    breakout_lookback: int = 24,
    breakout_vol_mult: float = 1.5,
    squeeze_lookback: int = 48,
    squeeze_contraction_q: float = 0.20,
    squeeze_expansion_q: float = 0.80,
    wick_threshold: float = 2.0,
    cascade_dd_threshold: float = -0.05,
    cascade_vol_mult: float = 2.0,
    range_mult: float = 2.0,
    range_lookback: int = 48,
) -> pd.DataFrame:
    """Add event-driven regime flags to a panel DataFrame.

    Parameters
    ----------
    panel : DataFrame with [asset, timestamp, open, high, low, close, volume].
    breakout_lookback : bars to look back for high/low breakout reference.
    breakout_vol_mult : volume must be this × rolling median to confirm breakout.
    squeeze_lookback : bars for Bollinger-style squeeze detection.
    squeeze_contraction_q : quantile below which vol is "contracted".
    squeeze_expansion_q : quantile above which vol is "expanded".
    wick_threshold : wick must be this × body size to count as wick reversal.
    cascade_dd_threshold : return threshold for drawdown (e.g. -5%).
    cascade_vol_mult : volume spike multiplier for cascade detection.
    range_mult : bar range must be this × median range for range expansion.
    range_lookback : bars for rolling median range.

    Returns
    -------
    DataFrame aligned to panel index with event flag columns.
    """
    parts: list[pd.DataFrame] = []

    for _asset, g in panel.groupby("asset", sort=False):
        g = g.sort_values("timestamp")
        n = len(g)
        flags = pd.DataFrame(index=g.index)

        o, h, l, c, v = g["open"], g["high"], g["low"], g["close"], g["volume"]
        bar_range = h - l
        body = (c - o).abs()

        # ── Breakout continuation ────────────────────────────────
        # Price breaks above rolling N-bar high with volume expansion
        rolling_high = h.rolling(breakout_lookback, min_periods=breakout_lookback).max().shift(1)
        rolling_low = l.rolling(breakout_lookback, min_periods=breakout_lookback).min().shift(1)
        vol_median = v.rolling(breakout_lookback, min_periods=breakout_lookback).median().shift(1)

        breakout_up = (c > rolling_high) & (v > vol_median * breakout_vol_mult)
        breakout_down = (c < rolling_low) & (v > vol_median * breakout_vol_mult)
        flags["event_breakout"] = (breakout_up | breakout_down).astype(np.int8)

        # ── Compression → expansion (Bollinger squeeze) ──────────
        # Volatility contracts then expands
        rolling_std = c.rolling(squeeze_lookback, min_periods=squeeze_lookback).std()
        std_q_low = rolling_std.quantile(squeeze_contraction_q)
        std_q_high = rolling_std.quantile(squeeze_expansion_q)

        was_contracted = rolling_std.shift(1) < std_q_low
        now_expanded = rolling_std > std_q_high
        flags["event_compression_expansion"] = (was_contracted & now_expanded).astype(np.int8)

        # ── Wick reversal ────────────────────────────────────────
        # Large wick relative to body on the current bar
        upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
        lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l

        # Use max wick (upper or lower)
        max_wick = pd.concat([upper_wick, lower_wick], axis=1).max(axis=1)
        body_safe = body.replace(0, np.nan)
        wick_ratio = max_wick / body_safe
        flags["event_wick_reversal"] = (wick_ratio > wick_threshold).fillna(False).astype(np.int8)

        # ── Cascade exhaustion ───────────────────────────────────
        # Sharp drawdown + volume spike + reversal (close above midpoint)
        ret_lookback = 4  # 4-bar return for cascade detection
        recent_ret = c / c.shift(ret_lookback) - 1
        vol_spike = v > vol_median * cascade_vol_mult
        bar_mid = (h + l) / 2
        close_above_mid = c > bar_mid

        cascade = (recent_ret < cascade_dd_threshold) & vol_spike & close_above_mid
        flags["event_cascade_exhaustion"] = cascade.astype(np.int8)

        # ── Range expansion ──────────────────────────────────────
        # Bar range significantly exceeds recent median
        median_range = bar_range.rolling(range_lookback, min_periods=range_lookback).median().shift(1)
        flags["event_range_expansion"] = (bar_range > median_range * range_mult).astype(np.int8)

        parts.append(flags)

    result = pd.concat(parts).loc[panel.index]

    # Log firing rates
    for col in EVENT_REGIME_COLS:
        if col in result.columns:
            rate = result[col].mean() * 100
            log.info("  %s firing rate: %.1f%%", col, rate)

    return result
