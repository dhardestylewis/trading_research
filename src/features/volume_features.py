"""Volume and liquidity features computed per-asset."""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_volume_features(g: pd.DataFrame) -> pd.DataFrame:
    """Compute volume-derived features for a single-asset group.

    Expects columns: close, volume, dollar_volume.
    """
    v = g["volume"]
    dv = g["dollar_volume"]
    c = g["close"]

    feats = pd.DataFrame(index=g.index)

    # ── Volume change ────────────────────────────────────────────────
    feats["vol_chg_1h"] = v.pct_change(1)

    # ── Relative volume vs 24h rolling mean ──────────────────────────
    vol_ma24 = v.rolling(24).mean()
    feats["rel_volume_24h"] = v / vol_ma24.replace(0, np.nan)

    # ── Dollar volume 24h sum ────────────────────────────────────────
    feats["dollar_volume_24h"] = dv.rolling(24).sum()

    # ── Amihud illiquidity proxy (|ret| / dollar_volume) ─────────────
    abs_ret = c.pct_change(1).abs()
    feats["amihud_proxy_24h"] = (abs_ret / dv.replace(0, np.nan)).rolling(24).mean()

    return feats
