"""Price and volatility features computed per-asset."""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_price_features(g: pd.DataFrame) -> pd.DataFrame:
    """Compute price-derived features for a single-asset group (sorted by timestamp).

    Expects columns: open, high, low, close, volume.
    Returns a DataFrame aligned to the same index with new feature columns.
    """
    c = g["close"]
    h = g["high"]
    lo = g["low"]

    feats = pd.DataFrame(index=g.index)

    # ── Lagged returns ───────────────────────────────────────────────
    for lag in (1, 2, 4, 8, 24):
        feats[f"ret_{lag}h"] = c.pct_change(lag)
    feats["log_ret_1h"] = np.log(c / c.shift(1))

    # ── Bar shape ────────────────────────────────────────────────────
    bar_range = h - lo
    feats["range_pct"] = bar_range / c.shift(1)
    feats["close_to_high"] = (c - h) / bar_range.replace(0, np.nan)
    feats["close_to_low"] = (c - lo) / bar_range.replace(0, np.nan)

    # ── Realized volatility ──────────────────────────────────────────
    log_ret = feats["log_ret_1h"]
    for win in (6, 24):
        feats[f"realized_vol_{win}h"] = log_ret.rolling(win).std()

    # ── Downside / upside volatility (24h) ───────────────────────────
    neg_ret = log_ret.clip(upper=0)
    pos_ret = log_ret.clip(lower=0)
    feats["downside_vol_24h"] = neg_ret.rolling(24).std()
    feats["upside_vol_24h"] = pos_ret.rolling(24).std()

    # ── Drawdown from rolling high ───────────────────────────────────
    for win in (24, 168):
        roll_max = c.rolling(win, min_periods=1).max()
        feats[f"drawdown_{win}h"] = (c - roll_max) / roll_max

    return feats
