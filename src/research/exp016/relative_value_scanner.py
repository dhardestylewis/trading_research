"""Relative value scanner for pairs."""
import pandas as pd
import numpy as np
from src.utils.logging import get_logger

log = get_logger("rv_scanner")

def build_ratio_panel(panel: pd.DataFrame, pairs: list[list[str]]) -> pd.DataFrame:
    """Combine two spot assets into a single synthetic RV ratio asset."""
    log.info(f"Building synthetic RV pairs: {pairs}")
    
    # Needs timestamp and asset index
    if "asset" in panel.columns:
        pvt = panel.pivot(index="timestamp", columns="asset")
    else:
        pvt = panel.pivot(index="timestamp", columns="symbol")
        
    ratio_dfs = []
    
    for pair in pairs:
        base, quote = pair
        base_label = f"{base}-USD" if f"{base}-USD" in pvt["close"].columns else f"{base}-USDT"
        quote_label = f"{quote}-USD" if f"{quote}-USD" in pvt["close"].columns else f"{quote}-USDT"
        
        if base_label not in pvt["close"].columns or quote_label not in pvt["close"].columns:
            log.warning(f"Could not find {base_label} or {quote_label} in panel.")
            continue
            
        r_close = pvt["close"][base_label] / pvt["close"][quote_label]
        r_high = pvt["high"][base_label] / pvt["low"][quote_label] # generous estimate of high
        r_low = pvt["low"][base_label] / pvt["high"][quote_label]  # conservative estimate of low
        r_open = pvt["open"][base_label] / pvt["open"][quote_label]
        
        rdf = pd.DataFrame({
            "close": r_close,
            "high": r_high,
            "low": r_low,
            "open": r_open
        }).dropna().reset_index()
        
        rdf["asset"] = f"{base}/{quote}_RV"
        ratio_dfs.append(rdf)
        
    if not ratio_dfs:
        return pd.DataFrame()
        
    return pd.concat(ratio_dfs, ignore_index=True)

def evaluate_rv_excursions(group: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Compute excursions for the RV ratio."""
    g = group.copy()
    g = g.sort_values("timestamp").reset_index(drop=True)
    
    close_prices = g["close"].values
    high_prices = g["high"].values
    low_prices = g["low"].values
    n = len(g)
    
    for h in horizons:
        mfe_upside_bps = np.full(n, np.nan)
        mfe_downside_bps = np.full(n, np.nan)
        
        for i in range(n):
            if i + 1 >= n:
                continue
            end_idx = min(i + 1 + h, n)
            window_high = np.max(high_prices[i+1:end_idx])
            window_low = np.min(low_prices[i+1:end_idx])
            
            entry_price = close_prices[i]
            
            mfe_upside_bps[i] = ((window_high / entry_price) - 1.0) * 10000.0
            mfe_downside_bps[i] = ((window_low / entry_price) - 1.0) * 10000.0
            
        g[f"mfe_up_{h}h_bps"] = mfe_upside_bps
        g[f"mfe_down_{h}h_bps"] = mfe_downside_bps
        
    return g
