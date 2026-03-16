"""Build perp-state and catalyst features for exp025 panel."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.utils.logging import get_logger

log = get_logger("build_perp_features")

def build_perp_features(panel: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """
    Appends perp features to a base panel.
    Returns the enriched dataframe with multiple horizon targets explicitly calculated.
    """
    df = panel.copy()
    
    # Needs to be sorted by asset then time
    df = df.sort_values(by=["asset", "timestamp"]).reset_index(drop=True)
    
    # Base returns (assuming a 'close' column exists; mostly it's provided as 'return_1h' or similar)
    if "close" in df.columns:
        df["ret_1"] = df.groupby("asset")["close"].pct_change()
    elif "return_1h" in df.columns:
        df["ret_1"] = df["return_1h"]
    else:
        log.warning("No base return column found to use for lag calculations.")
        df["ret_1"] = 0.0

    # 1. Perp State Features
    # Requires: funding_rate, open_interest_notional, mark_price (assume merged prior to this step or padded)
    if "funding_rate" in df.columns:
        df["funding_level"] = df["funding_rate"]
        df["funding_change_1"] = df.groupby("asset")["funding_rate"].diff(1)
        df["funding_zscore_24"] = df.groupby("asset")["funding_rate"].transform(
            lambda x: (x - x.rolling(24).mean()) / (x.rolling(24).std() + 1e-8)
        )
    
    if "open_interest_notional" in df.columns:
        df["oi_level"] = df["open_interest_notional"]
        df["oi_change_1"] = df.groupby("asset")["open_interest_notional"].pct_change()
        df["oi_change_4"] = df.groupby("asset")["open_interest_notional"].pct_change(4)
        df["oi_ret_interaction"] = df["oi_change_1"] * df["ret_1"]
        
    if "mark_price" in df.columns and "close" in df.columns:
        df["mark_premium"] = (df["mark_price"] / df["close"]) - 1.0

    # 2. Market Stress / Catalyst Features
    # Realized vol expansion
    df["rv_6"] = df.groupby("asset")["ret_1"].transform(lambda x: x.rolling(6).std())
    df["rv_24"] = df.groupby("asset")["ret_1"].transform(lambda x: x.rolling(24).std())
    df["vol_expansion"] = df["rv_6"] / (df["rv_24"] + 1e-8)
    
    # Large return shocks
    df["ret_shock"] = (df["ret_1"].abs() > 0.02).astype(float) # > 2% absolute move
    
    # Volume surge
    if "volume" in df.columns:
        df["vol_change"] = df.groupby("asset")["volume"].pct_change()
        df["vol_surge"] = df.groupby("asset")["volume"].transform(
            lambda x: (x - x.rolling(24).mean()) / (x.rolling(24).std() + 1e-8)
        )
        
    # Overnight / Weekend regime
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(float)
    df["is_overnight_utc"] = ((df["hour"] >= 23) | (df["hour"] <= 5)).astype(float)
    
    # Liquidation proxy (Large down return + OI collapse)
    if "oi_change_1" in df.columns:
        df["liq_long_stress"] = ((df["ret_1"] < -0.01) & (df["oi_change_1"] < -0.01)).astype(float)
        df["liq_short_stress"] = ((df["ret_1"] > 0.01) & (df["oi_change_1"] < -0.01)).astype(float)

    # 3. Cross-sectional Structure
    # BTC / ETH residual returns
    time_grouped = df.groupby("timestamp")
    
    def get_market_ret(ticker: str) -> pd.Series:
        idx_ret = df[df["asset"].str.contains(ticker, case=False)]
        if idx_ret.empty:
            return pd.Series(0, index=df["timestamp"].unique())
        return idx_ret.set_index("timestamp")["ret_1"]
        
    btc_ret = get_market_ret("BTC")
    eth_ret = get_market_ret("ETH")
    
    df["btc_ret"] = df["timestamp"].map(btc_ret).fillna(0)
    df["eth_ret"] = df["timestamp"].map(eth_ret).fillna(0)
    
    df["resid_btc"] = df["ret_1"] - df["btc_ret"]
    df["resid_eth"] = df["ret_1"] - df["eth_ret"]
    
    # Dispersion (std of returns across cross-section at time t)
    cross_dispersion = time_grouped["ret_1"].std().fillna(0)
    df["cross_dispersion"] = df["timestamp"].map(cross_dispersion)

    # 4. Multi-horizon targets
    for h in horizons:
        fwd_ret = df.groupby("asset")["ret_1"].apply(
            lambda x: (1 + x).rolling(h).apply(np.prod, raw=True).shift(-h) - 1
        ).reset_index(level=0, drop=True)
        
        # Sort back to original index since groupby returns ordered by asset
        # We can just use shift(-h) within groupby and reassemble
        
        # Safer way:
        col_fwd = f"fwd_ret_{h}"
        df[col_fwd] = df.groupby("asset")["close"].shift(-h) / df["close"] - 1.0
        
        # Categorical targets
        df[f"prob_tail_25_{h}"] = (df[col_fwd].abs() > 0.0025).astype(int)
        df[f"prob_tail_50_{h}"] = (df[col_fwd].abs() > 0.0050).astype(int)
        df[f"prob_tail_100_{h}"] = (df[col_fwd].abs() > 0.0100).astype(int)
    
    log.info(f"Built perp features. New shape: {df.shape}")
    return df
