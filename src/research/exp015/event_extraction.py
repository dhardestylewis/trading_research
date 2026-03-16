"""Extract intrinsic structural event states from OHLCV panel data."""
import pandas as pd
import numpy as np
from src.utils.logging import get_logger

log = get_logger("event_extraction")

def add_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def extract_events(panel: pd.DataFrame, events_cfg: dict) -> pd.DataFrame:
    """
    Take a panel DataFrame (multi-index or asset-grouped) and create boolean event columns.
    Returns the panel with added boolean columns for each event.
    """
    log.info("Extracting event states...")
    
    # Needs to be grouped by asset for rolling operations
    def process_asset(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        
        # Pre-compute indicators if needed
        has_vol = "vol_expansion" in events_cfg
        has_brk = "range_breakout_72h" in events_cfg
        has_rsi = "rsi_extreme" in events_cfg
        
        if has_vol:
            cfg = events_cfg["vol_expansion"]
            atr = add_atr(g, cfg["atr_window"])
            baseline = atr.rolling(cfg["baseline_window"]).mean()
            # vol expansion if current ATR is X times larger than the baseline average
            g["event_vol_expansion"] = atr > (cfg["multiplier"] * baseline)
            
        if has_brk:
            cfg = events_cfg["range_breakout_72h"]
            # strict breakout above highest high of the last N bars (not including current)
            rolling_high = g["high"].shift(1).rolling(cfg["window"]).max()
            rolling_low = g["low"].shift(1).rolling(cfg["window"]).min()
            
            g["event_breakout_up"] = g["close"] > rolling_high
            g["event_breakout_down"] = g["close"] < rolling_low
            
        if has_rsi:
            cfg = events_cfg["rsi_extreme"]
            rsi = add_rsi(g, cfg["window"])
            g["event_rsi_overbought"] = rsi > cfg["upper"]
            g["event_rsi_oversold"] = rsi < cfg["lower"]
            
        return g

    # Check how panel is structured
    if "asset" in panel.columns:
        result = panel.groupby("asset", group_keys=False).apply(process_asset)
    elif "symbol" in panel.columns:
        result = panel.groupby("symbol", group_keys=False).apply(process_asset)
    else:
        # fallback single asset
        result = process_asset(panel)
        
    # ensure timestamp ordering
    if "timestamp" in result.columns:
        result = result.sort_values(["asset" if "asset" in result.columns else "symbol", "timestamp"])
        
    return result
