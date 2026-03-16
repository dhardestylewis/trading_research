"""Build rich perp-state, catalyst, cross-asset, and sequence features for exp026."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.utils.logging import get_logger

log = get_logger("build_rich_perp_features")

def build_rich_perp_state_features(panel: pd.DataFrame, horizons: list[int], sequence_len: int = 6) -> pd.DataFrame:
    """
    Appends rich perp features, catalyst features, cross-asset features, and sequence lags.
    """
    df = panel.copy()
    
    # Needs to be sorted by asset then time
    df = df.sort_values(by=["asset", "timestamp"]).reset_index(drop=True)
    
    if "close" in df.columns:
        df["ret_1"] = df.groupby("asset")["close"].pct_change()
    elif "return_1h" in df.columns:
        df["ret_1"] = df["return_1h"]
    else:
        df["ret_1"] = 0.0

    # ── 1. Core Perp-State Features ──
    if "funding_rate" in df.columns:
        df["funding_level"] = df["funding_rate"]
        df["funding_change_1"] = df.groupby("asset")["funding_rate"].diff(1)
        df["funding_zscore_24"] = df.groupby("asset")["funding_rate"].transform(
            lambda x: (x - x.rolling(24).mean()) / (x.rolling(24).std() + 1e-8)
        )
        df["rolling_cum_funding_24"] = df.groupby("asset")["funding_rate"].transform(
            lambda x: x.rolling(24).sum()
        )
    
    if "open_interest_notional" in df.columns:
        df["oi_level"] = df["open_interest_notional"]
        df["oi_change_1"] = df.groupby("asset")["open_interest_notional"].pct_change()
        df["oi_shock_zscore_24"] = df.groupby("asset")["oi_change_1"].transform(
            lambda x: (x - x.rolling(24).mean()) / (x.rolling(24).std() + 1e-8)
        )
        df["oi_ret_interaction"] = df["oi_change_1"] * df["ret_1"]
        
    if "mark_price" in df.columns and "close" in df.columns:
        df["mark_premium"] = (df["mark_price"] / df["close"]) - 1.0
        df["premium_persistence_6"] = df.groupby("asset")["mark_premium"].transform(
            lambda x: x.rolling(6).mean()
        )

    # ── 2. Market Stress / Catalyst Features ──
    # Volatility and range
    df["rv_6"] = df.groupby("asset")["ret_1"].transform(lambda x: x.rolling(6).std())
    df["rv_24"] = df.groupby("asset")["ret_1"].transform(lambda x: x.rolling(24).std())
    df["vol_expansion"] = df["rv_6"] / (df["rv_24"] + 1e-8)
    
    if "high" in df.columns and "low" in df.columns:
        df["range_1"] = (df["high"] - df["low"]) / df["close"]
        df["range_expansion"] = df.groupby("asset")["range_1"].transform(
            lambda x: x / (x.rolling(24).mean() + 1e-8)
        )

    # Return shock and asymmetry
    df["ret_shock_mag"] = df["ret_1"].abs()
    
    def downside_asym(x):
        neg = x[x < 0]
        pos = x[x > 0]
        neg_vol = neg.std() if len(neg) > 1 else 1e-8
        pos_vol = pos.std() if len(pos) > 1 else 1e-8
        return neg_vol / (pos_vol + 1e-8)
        
    df["downside_vol_asym_24"] = df.groupby("asset")["ret_1"].transform(
        lambda x: x.rolling(24).apply(downside_asym, raw=False)
    ).fillna(1.0)

    # Drawdown / rebound
    df["roll_max_24"] = df.groupby("asset")["close"].transform(lambda x: x.rolling(24).max())
    df["drawdown_24"] = (df["close"] / df["roll_max_24"]) - 1.0

    # Volume surge
    if "volume" in df.columns:
        df["vol_change"] = df.groupby("asset")["volume"].pct_change()
        df["vol_surge"] = df.groupby("asset")["volume"].transform(
            lambda x: (x - x.rolling(24).mean()) / (x.rolling(24).std() + 1e-8)
        )
        
    # Temporal effects
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(float)
    df["is_overnight_utc"] = ((df["hour"] >= 23) | (df["hour"] <= 5)).astype(float)

    # ── 3. Cross-sectional Structure ──
    time_grouped = df.groupby("timestamp")
    
    def get_market_ret(ticker: str) -> pd.Series:
        idx_ret = df[df["asset"].str.contains(ticker, case=False)]
        if idx_ret.empty:
            return pd.Series(0.0, index=df["timestamp"].unique())
        return idx_ret.set_index("timestamp")["ret_1"]
        
    btc_ret = get_market_ret("BTC")
    eth_ret = get_market_ret("ETH")
    
    df["btc_ret"] = df["timestamp"].map(btc_ret).fillna(0.0)
    df["eth_ret"] = df["timestamp"].map(eth_ret).fillna(0.0)
    
    df["resid_btc"] = df["ret_1"] - df["btc_ret"]
    df["resid_eth"] = df["ret_1"] - df["eth_ret"]
    
    df["cross_rank_ret"] = time_grouped["ret_1"].rank(pct=True)
    
    if "vol_surge" in df.columns:
        df["cross_rank_vol_surge"] = time_grouped["vol_surge"].rank(pct=True)
        
    if "funding_zscore_24" in df.columns:
        df["cross_rank_funding"] = time_grouped["funding_zscore_24"].rank(pct=True)
        
    df["cross_dispersion"] = time_grouped["ret_1"].transform("std").fillna(0)

    # ── 4. Sequence Features (Lags for Challenger) ──
    seq_cols = []
    if "ret_1" in df.columns: seq_cols.append("ret_1")
    if "oi_change_1" in df.columns: seq_cols.append("oi_change_1")
    if "funding_level" in df.columns: seq_cols.append("funding_level")
    if "mark_premium" in df.columns: seq_cols.append("mark_premium")
    if "vol_surge" in df.columns: seq_cols.append("vol_surge")
    if "rv_6" in df.columns: seq_cols.append("rv_6")

    for col in seq_cols:
        for lag in range(1, sequence_len + 1):
            df[f"seq_{col}_lag_{lag}"] = df.groupby("asset")[col].shift(lag)

    # ── Multi-horizon targets ──
    for h in horizons:
        col_fwd = f"fwd_ret_{h}"
        df[col_fwd] = df.groupby("asset")["close"].shift(-h) / df["close"] - 1.0
        
        # We need targets in bps for the models
        df[f"gross_move_bps_{h}"] = df[col_fwd] * 10000.0
        # Categorical tail targets (in bps)
        df[f"prob_tail_25_{h}"] = (df[f"gross_move_bps_{h}"].abs() > 25.0).astype(int)
        df[f"prob_tail_50_{h}"] = (df[f"gross_move_bps_{h}"].abs() > 50.0).astype(int)
        df[f"prob_tail_100_{h}"] = (df[f"gross_move_bps_{h}"].abs() > 100.0).astype(int)
        
    log.info(f"Built rich perp-state features. New shape: {df.shape}")
    return df
