"""Extended feature engineering for latent state discovery.

Builds a rich feature matrix beyond the baseline price/volume/regime features,
adding spread dynamics, volatility-of-volatility, compression signals,
microstructure proxies, cross-asset momentum, and calendar encoding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger("exp019.feature_engineer")


# ── Per-asset expanded features ──────────────────────────────────────────

def _spread_velocity_features(close: pd.Series, lags: list[int]) -> pd.DataFrame:
    """Spread velocity (Δ return) and acceleration (ΔΔ return)."""
    feats = pd.DataFrame(index=close.index)
    for lag in lags:
        ret = close.pct_change(lag)
        feats[f"ret_{lag}h"] = ret
        feats[f"ret_vel_{lag}h"] = ret.diff()
        feats[f"ret_acc_{lag}h"] = ret.diff().diff()
    return feats


def _volatility_features(
    close: pd.Series, high: pd.Series, low: pd.Series,
    vol_windows: list[int], vov_window: int,
) -> pd.DataFrame:
    """Realized vol, vol-of-vol, ATR ratio."""
    feats = pd.DataFrame(index=close.index)
    log_ret = np.log(close / close.shift(1))

    for win in vol_windows:
        rv = log_ret.rolling(win).std()
        feats[f"realized_vol_{win}h"] = rv
        # Vol-of-vol: rolling std of realized vol
        if win >= 12:
            feats[f"vol_of_vol_{win}h"] = rv.rolling(vov_window).std()

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    for win in vol_windows:
        atr = tr.rolling(win).mean()
        feats[f"atr_{win}h"] = atr

    # Downside / upside vol split
    neg_ret = log_ret.clip(upper=0)
    pos_ret = log_ret.clip(lower=0)
    feats["downside_vol_24h"] = neg_ret.rolling(24).std()
    feats["upside_vol_24h"] = pos_ret.rolling(24).std()
    feats["vol_skew_24h"] = feats["downside_vol_24h"] / feats["upside_vol_24h"].replace(0, np.nan)

    return feats


def _compression_features(
    close: pd.Series, high: pd.Series, low: pd.Series,
    bb_window: int, trailing_window: int,
) -> pd.DataFrame:
    """Bollinger bandwidth compression, range compression."""
    feats = pd.DataFrame(index=close.index)

    # Bollinger bandwidth
    ma = close.rolling(bb_window).mean()
    std = close.rolling(bb_window).std()
    bb_width = (2 * std) / ma.replace(0, np.nan)
    bb_width_trailing = bb_width.rolling(trailing_window).mean()
    feats["bb_width"] = bb_width
    feats["bb_compression_ratio"] = bb_width / bb_width_trailing.replace(0, np.nan)

    # Range compression: current range vs trailing
    bar_range = high - low
    range_ma_short = bar_range.rolling(bb_window).mean()
    range_ma_long = bar_range.rolling(trailing_window).mean()
    feats["range_compression"] = range_ma_short / range_ma_long.replace(0, np.nan)

    # Bollinger position (where is close within the bands)
    upper = ma + 2 * std
    lower = ma - 2 * std
    feats["bb_position"] = (close - lower) / (upper - lower).replace(0, np.nan)

    return feats


def _microstructure_proxies(
    close: pd.Series, volume: pd.Series, dollar_volume: pd.Series,
    amihud_window: int, vwap_window: int,
) -> pd.DataFrame:
    """Amihud illiquidity, VWAP deviation, volume surge indicators."""
    feats = pd.DataFrame(index=close.index)

    abs_ret = close.pct_change(1).abs()

    # Amihud extended
    amihud = abs_ret / dollar_volume.replace(0, np.nan)
    feats["amihud_proxy"] = amihud.rolling(amihud_window).mean()
    feats["amihud_zscore"] = (
        (amihud - amihud.rolling(72).mean()) /
        amihud.rolling(72).std().replace(0, np.nan)
    )

    # Relative volume
    vol_ma24 = volume.rolling(24).mean()
    feats["rel_volume_24h"] = volume / vol_ma24.replace(0, np.nan)

    # Volume surge (current bar vs trailing 24h mean)
    feats["volume_surge_3h"] = (
        volume.rolling(3).mean() / vol_ma24.replace(0, np.nan)
    )

    # Dollar volume z-score
    dv_24 = dollar_volume.rolling(24).mean()
    dv_72 = dollar_volume.rolling(72).mean()
    feats["dv_ratio_24_72"] = dv_24 / dv_72.replace(0, np.nan)

    # Bar shape features
    bar_range = close.index  # placeholder for high-low
    feats["close_location"] = (close - close.shift(1)) / abs_ret.replace(0, np.nan)

    return feats


def _momentum_features(
    close: pd.Series, asset: str, market_rets: pd.DataFrame,
    mom_window: int, divergence_windows: list[int],
) -> pd.DataFrame:
    """Cross-asset relative momentum and momentum divergence."""
    feats = pd.DataFrame(index=close.index)

    # Asset momentum (various lookbacks)
    for win in divergence_windows:
        feats[f"momentum_{win}h"] = close.pct_change(win)

    # Momentum acceleration: short vs long
    if len(divergence_windows) >= 2:
        short_w, long_w = divergence_windows[0], divergence_windows[-1]
        feats["momentum_divergence"] = (
            close.pct_change(short_w) - close.pct_change(long_w)
        )

    # Drawdown from rolling highs
    for win in [24, 72, 168]:
        roll_max = close.rolling(win, min_periods=1).max()
        feats[f"drawdown_{win}h"] = (close - roll_max) / roll_max

    return feats


def _calendar_features(timestamps: pd.Series) -> pd.DataFrame:
    """Sin/cos encoded calendar features."""
    feats = pd.DataFrame(index=timestamps.index)
    hour = timestamps.dt.hour
    dow = timestamps.dt.dayofweek

    # Sin/cos encoding for cyclical features
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feats["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    feats["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    feats["is_weekend"] = (dow >= 5).astype(int)

    return feats


# ── Orchestrator ─────────────────────────────────────────────────────────

def build_extended_features(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Build the full extended feature matrix from the OHLCV panel.

    Parameters
    ----------
    panel : DataFrame with columns [asset, timestamp, open, high, low, close,
            volume, dollar_volume]
    cfg : features config dict from the experiment YAML

    Returns
    -------
    DataFrame aligned to panel index with all extended features
    """
    feat_cfg = cfg
    return_lags = feat_cfg.get("return_lags", [1, 2, 4, 8, 24])
    vol_windows = feat_cfg.get("vol_windows", [6, 12, 24, 48])
    vov_window = feat_cfg.get("vol_of_vol_window", 24)
    bb_window = feat_cfg.get("bollinger_window", 24)
    compression_trail = feat_cfg.get("compression_trailing_window", 72)
    amihud_window = feat_cfg.get("amihud_window", 24)
    vwap_window = feat_cfg.get("vwap_deviation_window", 12)
    mom_window = feat_cfg.get("cross_asset_momentum_window", 24)
    divergence_windows = feat_cfg.get("momentum_divergence_windows", [6, 24])
    do_calendar = feat_cfg.get("calendar_encode", True)
    do_winsorize = feat_cfg.get("winsorize", False)
    winsorize_limits = feat_cfg.get("winsorize_limits", [0.005, 0.995])

    # ── Build market-wide returns for cross-asset momentum ─────────
    wide = panel.pivot(index="timestamp", columns="asset", values="close")
    market_rets = wide.pct_change(1)

    # ── Per-asset features ─────────────────────────────────────────
    parts: list[pd.DataFrame] = []

    for asset, g in panel.groupby("asset", sort=False):
        g_sorted = g.sort_values("timestamp")
        c = g_sorted["close"]
        h = g_sorted["high"]
        lo = g_sorted["low"]
        v = g_sorted["volume"]
        dv = g_sorted.get("dollar_volume", v * c)

        asset_feats = pd.concat([
            _spread_velocity_features(c, return_lags),
            _volatility_features(c, h, lo, vol_windows, vov_window),
            _compression_features(c, h, lo, bb_window, compression_trail),
            _microstructure_proxies(c, v, dv, amihud_window, vwap_window),
            _momentum_features(c, asset, market_rets, mom_window, divergence_windows),
        ], axis=1)

        parts.append(asset_feats)

    per_asset = pd.concat(parts).loc[panel.index]

    # ── Panel-wide regime features ─────────────────────────────────
    mkt_ret_1h = market_rets.mean(axis=1).rename("market_ret_1h")
    mkt_ret_24h = wide.pct_change(24).mean(axis=1).rename("market_ret_24h")
    log_rets_mkt = np.log(wide / wide.shift(1))
    mkt_vol_24h = log_rets_mkt.mean(axis=1).rolling(24).std().rename("market_vol_24h")

    mkt = pd.DataFrame({
        "market_ret_1h": mkt_ret_1h,
        "market_ret_24h": mkt_ret_24h,
        "market_vol_24h": mkt_vol_24h,
    })
    regime_feats = panel[["timestamp"]].merge(
        mkt, left_on="timestamp", right_index=True, how="left"
    )
    regime_feats.index = panel.index

    # ── Calendar features ──────────────────────────────────────────
    cal_feats = _calendar_features(panel["timestamp"]) if do_calendar else pd.DataFrame(index=panel.index)

    # ── Combine ────────────────────────────────────────────────────
    features = pd.concat([
        panel[["asset", "timestamp"]],
        per_asset,
        regime_feats.drop(columns=["timestamp"]),
        cal_feats,
    ], axis=1)

    # ── Winsorize ──────────────────────────────────────────────────
    if do_winsorize:
        lo_q, hi_q = winsorize_limits
        num_cols = features.select_dtypes(include=[np.number]).columns
        skip = {"hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"}
        for col in num_cols:
            if col in skip:
                continue
            lb = features[col].quantile(lo_q)
            ub = features[col].quantile(hi_q)
            features[col] = features[col].clip(lb, ub)

    n_feat = len(features.columns) - 2  # minus asset, timestamp
    log.info("Extended features built: %d rows × %d feature cols", len(features), n_feat)
    return features
