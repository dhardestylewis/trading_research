"""Markout and adverse selection analysis for exp014."""
from __future__ import annotations
import pandas as pd
import numpy as np
from numba import njit
from src.utils.logging import get_logger

log = get_logger("markout_analysis")

@njit
def compute_markouts(
    fill_times: np.ndarray,
    fill_prices: np.ndarray,
    directions: np.ndarray,
    filled_flags: np.ndarray,
    trade_times: np.ndarray,
    trade_prices: np.ndarray,
    horizons_sec: np.ndarray
) -> np.ndarray:
    """
    Compute forward return (markout) at specific horizons after a fill.
    Returns a 2D array of shape (n_fills, n_horizons) containing returns in bps.
    """
    n_fills = len(fill_times)
    n_horizons = len(horizons_sec)
    n_trades = len(trade_times)
    
    markouts_bps = np.full((n_fills, n_horizons), np.nan, dtype=np.float64)
    
    horizon_ms = horizons_sec * 1000.0
    
    for i in range(n_fills):
        if not filled_flags[i]:
            continue
            
        f_time = fill_times[i]
        f_price = fill_prices[i]
        direction = directions[i]
        
        # Find index in trades for the fill time
        # (Could be optimized with binary search, but numba loop is fast enough for 1D)
        start_idx = np.searchsorted(trade_times, f_time)
        
        for h_idx in range(n_horizons):
            target_time = f_time + horizon_ms[h_idx]
            
            # Find the trade closest to target_time
            t_idx = start_idx
            while t_idx < n_trades and trade_times[t_idx] < target_time:
                t_idx += 1
                
            if t_idx < n_trades:
                markout_price = trade_prices[t_idx]
                ret = (markout_price / f_price - 1.0) * direction * 10000.0
                markouts_bps[i, h_idx] = ret
                
    return markouts_bps

def build_markout_table(
    trades_df: pd.DataFrame, 
    signals_df: pd.DataFrame, 
    simulation_results: dict, 
    horizons: list[int]
) -> pd.DataFrame:
    """
    Combine touch_to_fill results into a structured table of markouts by haircut.
    """
    horizons_arr = np.array(horizons, dtype=np.float64)
    trade_ts = trades_df["timestamp_ms"].values
    trade_prices = trades_df["price"].values
    dirs = signals_df["direction"].values
    
    summary_rows = []
    
    for hc, res in simulation_results.items():
        filled = res["filled"]
        fill_count = filled.sum()
        total_count = len(filled)
        fill_prob = fill_count / total_count if total_count > 0 else 0.0
        
        if fill_count == 0:
            row = {"haircut_bps": hc, "fills": 0, "fill_prob": 0.0}
            for h in horizons:
                row[f"markout_{h}s_bps"] = np.nan
            summary_rows.append(row)
            continue
            
        m_bps = compute_markouts(
            res["fill_times"], res["fill_prices"], dirs, filled,
            trade_ts, trade_prices, horizons_arr
        )
        
        row = {
            "haircut_bps": hc,
            "fills": fill_count,
            "fill_prob": fill_prob,
        }
        
        # Average markout for filled trades
        for h_idx, h in enumerate(horizons):
            valid_markouts = m_bps[filled, h_idx]
            valid_markouts = valid_markouts[~np.isnan(valid_markouts)]
            avg_m = np.mean(valid_markouts) if len(valid_markouts) > 0 else np.nan
            row[f"markout_{h}s_bps"] = avg_m
            
        summary_rows.append(row)
        
    return pd.DataFrame(summary_rows)
