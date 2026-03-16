"""Touch-to-fill probability and queue position proxy functions for exp014."""
from __future__ import annotations
import pandas as pd
import numpy as np
from numba import njit
from src.utils.logging import get_logger

log = get_logger("touch_to_fill")

def compute_queue_features(trades_df: pd.DataFrame) -> pd.DataFrame:
    """ Compute rolling volume features that can proxy for queue depth. """
    # For a high-frequency trades DataFrame, we can approximate queue by looking at 
    # the volume of trades executed at the same price level.
    # In a full L2 setup, we'd have the orderbook. With just trades, we 
    # estimate the queue by tracking continuous trading at the same price.
    
    # We will just return the df as is for now, but ensure it's sorted
    df = trades_df.copy()
    if not df["timestamp_ms"].is_monotonic_increasing:
        df = df.sort_values("timestamp_ms").reset_index(drop=True)
    return df

@njit
def simulate_passive_fills(
    timestamps: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    is_buyer_maker: np.ndarray,
    target_prices: np.ndarray,
    target_times: np.ndarray,
    target_directions: np.ndarray, # 1 for long (buy limit), -1 for short (sell limit)
    queue_haircut_bps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate passive limit order fills given an array of target entry times and prices.
    Returns array of (fill_time, fill_price, filled_boolean).
    """
    n_targets = len(target_prices)
    n_trades = len(prices)
    
    fill_times = np.zeros(n_targets, dtype=np.float64)
    fill_prices = np.zeros(n_targets, dtype=np.float64)
    filled_flags = np.zeros(n_targets, dtype=np.bool_)
    
    max_wait_ms = 3600000.0 # 1 hour limit order duration
    
    trade_idx = 0
    for i in range(n_targets):
        t_time = target_times[i]
        t_price = target_prices[i]
        direction = target_directions[i]
        
        while trade_idx < n_trades and timestamps[trade_idx] < t_time:
            trade_idx += 1
            
        if trade_idx >= n_trades:
            break
            
        curr_idx = trade_idx
        
        if direction == 1: # Long
            required_price = t_price * (1.0 - queue_haircut_bps / 10000.0)
            while curr_idx < n_trades and timestamps[curr_idx] <= t_time + max_wait_ms:
                if prices[curr_idx] <= required_price:
                    fill_times[i] = timestamps[curr_idx]
                    fill_prices[i] = t_price
                    filled_flags[i] = True
                    break
                curr_idx += 1
                
        else: # Short
            required_price = t_price * (1.0 + queue_haircut_bps / 10000.0)
            while curr_idx < n_trades and timestamps[curr_idx] <= t_time + max_wait_ms:
                if prices[curr_idx] >= required_price:
                    fill_times[i] = timestamps[curr_idx]
                    fill_prices[i] = t_price
                    filled_flags[i] = True
                    break
                curr_idx += 1
                
    return fill_times, fill_prices, filled_flags

def run_touch_simulation(trades_df: pd.DataFrame, signals_df: pd.DataFrame, queue_haircuts: list[float]) -> dict:
    """
    Given a dataframe of trades and a dataframe of theoretical 1h signal entries,
    compute the touch-to-fill and true fill rates across different queue haircuts.
    """
    ts = trades_df["timestamp_ms"].values
    prices = trades_df["price"].values
    vols = trades_df["qty"].values
    is_buyer_maker = trades_df["is_buyer_maker"].values
    
    sig_times = signals_df["timestamp_ms"].values
    sig_prices = signals_df["target_entry_price"].values
    sig_dirs = signals_df["direction"].values # 1 or -1
    
    results = {}
    
    for hc in queue_haircuts:
        f_times, f_prices, f_flags = simulate_passive_fills(
            ts, prices, vols, is_buyer_maker,
            sig_prices, sig_times, sig_dirs, hc
        )
        results[hc] = {
            "fill_times": f_times,
            "fill_prices": f_prices,
            "filled": f_flags
        }
    
    return results
