"""Regime and context features (market-wide + calendar)."""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_regime_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute market-context features on the full panel.

    Steps:
      1. Build equal-weight market return from all assets.
      2. Merge market features back per row.
      3. Add calendar features.

    Expects panel sorted by (asset, timestamp) with columns: close, timestamp.
    Returns a DataFrame aligned to the panel index with new columns.
    """
    # ── Market-wide return (equal weight) ────────────────────────────
    wide = panel.pivot(index="timestamp", columns="asset", values="close")
    mkt_ret_1h = wide.pct_change(1).mean(axis=1).rename("market_ret_1h")
    mkt_ret_24h = wide.pct_change(24).mean(axis=1).rename("market_ret_24h")

    log_rets = np.log(wide / wide.shift(1))
    mkt_vol_24h = log_rets.mean(axis=1).rolling(24).std().rename("market_vol_24h")

    mkt = pd.DataFrame({"market_ret_1h": mkt_ret_1h, "market_ret_24h": mkt_ret_24h, "market_vol_24h": mkt_vol_24h})

    # ── Merge back to panel ──────────────────────────────────────────
    feats = panel[["timestamp"]].merge(mkt, left_on="timestamp", right_index=True, how="left")
    feats.index = panel.index  # realign

    # ── Calendar features ────────────────────────────────────────────
    ts = panel["timestamp"]
    feats["hour_of_day"] = ts.dt.hour
    feats["day_of_week"] = ts.dt.dayofweek
    feats["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    return feats.drop(columns=["timestamp"])
