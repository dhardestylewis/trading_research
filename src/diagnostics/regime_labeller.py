"""Deterministic regime labelling from existing features.

Adds 12 binary regime columns to a DataFrame that already has the
baseline feature set.  Quantile thresholds are computed from a
*reference* set (training data) to avoid look-ahead bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger("regime_labeller")


def _quantile_flag(
    series: pd.Series,
    ref_series: pd.Series,
    quantile: float,
    operator: str,
) -> pd.Series:
    """Return binary flag comparing *series* to a quantile of *ref_series*."""
    thresh = ref_series.quantile(quantile)
    if operator == ">":
        return (series > thresh).astype(np.int8)
    elif operator == "<":
        return (series < thresh).astype(np.int8)
    elif operator == ">=":
        return (series >= thresh).astype(np.int8)
    elif operator == "<=":
        return (series <= thresh).astype(np.int8)
    raise ValueError(f"Unknown operator: {operator}")


def label_regimes(
    df: pd.DataFrame,
    ref: pd.DataFrame | None = None,
    *,
    high_vol_q: float = 0.70,
    low_vol_q: float = 0.30,
    trend_up_q: float = 0.70,
    trend_down_q: float = 0.30,
    liquidity_q: float = 0.70,
    drawdown_threshold: float = -0.10,
) -> pd.DataFrame:
    """Add 12 regime flag columns to *df*.

    Parameters
    ----------
    df : DataFrame with columns from baseline_features_v1 (at minimum:
         realized_vol_24h, ret_24h, ret_1h, drawdown_168h,
         dollar_volume_24h, is_weekend, hour_of_day).
    ref : Reference DataFrame for computing quantile thresholds.
          If None, uses *df* itself (fine for post-hoc diagnostics on
          full predictions file, but should be training-set only when
          running fold-by-fold).
    """
    if ref is None:
        ref = df

    out = df.copy()

    # ── Volatility ──────────────────────────────────────────────
    out["regime_vol_high"] = _quantile_flag(
        df["realized_vol_24h"], ref["realized_vol_24h"], high_vol_q, ">"
    )
    out["regime_vol_low"] = _quantile_flag(
        df["realized_vol_24h"], ref["realized_vol_24h"], low_vol_q, "<"
    )

    # ── Trend ───────────────────────────────────────────────────
    out["regime_trend_up"] = _quantile_flag(
        df["ret_24h"], ref["ret_24h"], trend_up_q, ">"
    )
    out["regime_trend_down"] = _quantile_flag(
        df["ret_24h"], ref["ret_24h"], trend_down_q, "<"
    )

    # ── Chop: low |return| but moderate+ vol ────────────────────
    abs_ret = df["ret_24h"].abs()
    ref_abs_ret = ref["ret_24h"].abs()
    low_ret = abs_ret < ref_abs_ret.quantile(0.70)
    high_vol = df["realized_vol_24h"] > ref["realized_vol_24h"].quantile(0.50)
    out["regime_chop"] = (low_ret & high_vol).astype(np.int8)

    # ── Drawdown / Rebound ──────────────────────────────────────
    dd_col = "drawdown_168h" if "drawdown_168h" in df.columns else "drawdown_24h"
    in_dd = df[dd_col] < drawdown_threshold
    out["regime_drawdown"] = in_dd.astype(np.int8)

    rebounding = in_dd & (df["ret_1h"] > 0)
    out["regime_rebound"] = rebounding.astype(np.int8)

    # ── Liquidity ───────────────────────────────────────────────
    out["regime_liquidity_high"] = _quantile_flag(
        df["dollar_volume_24h"], ref["dollar_volume_24h"], liquidity_q, ">"
    )

    # ── Calendar ────────────────────────────────────────────────
    out["regime_weekend"] = df["is_weekend"].astype(np.int8) if "is_weekend" in df.columns else 0

    hour = df["hour_of_day"] if "hour_of_day" in df.columns else df["timestamp"].dt.hour
    out["regime_us_hours"] = hour.isin(range(13, 21)).astype(np.int8)
    out["regime_asia_hours"] = hour.isin(range(0, 8)).astype(np.int8)

    return out


REGIME_COLS = [
    "regime_vol_high",
    "regime_vol_low",
    "regime_trend_up",
    "regime_trend_down",
    "regime_chop",
    "regime_drawdown",
    "regime_rebound",
    "regime_liquidity_high",
    "regime_weekend",
    "regime_us_hours",
    "regime_asia_hours",
]
