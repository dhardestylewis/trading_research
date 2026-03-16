"""Compute gross magnitude distributions post-event."""
import pandas as pd
import numpy as np
from src.utils.logging import get_logger

log = get_logger("magnitude_distributions")

def compute_excursions(group: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Compute maximum favorable and adverse excursions for specific horizons ahead."""
    g = group.copy()
    g = g.sort_values("timestamp").reset_index(drop=True)
    
    close_prices = g["close"].values
    high_prices = g["high"].values
    low_prices = g["low"].values
    n = len(g)
    
    for h in horizons:
        # absolute maximum up/down excursion within next H bars
        mfe_upside_bps = np.full(n, np.nan)
        mfe_downside_bps = np.full(n, np.nan)
        
        for i in range(n):
            if i + 1 >= n:
                continue
            end_idx = min(i + 1 + h, n)
            window_high = np.max(high_prices[i+1:end_idx])
            window_low = np.min(low_prices[i+1:end_idx])
            
            entry_price = close_prices[i]
            
            # max % move up
            mfe_upside_bps[i] = ((window_high / entry_price) - 1.0) * 10000.0
            # max % move down
            mfe_downside_bps[i] = ((window_low / entry_price) - 1.0) * 10000.0
            
        g[f"mfe_up_{h}h_bps"] = mfe_upside_bps
        g[f"mfe_down_{h}h_bps"] = mfe_downside_bps
        
    return g

def build_exceedance_tables(df_with_events: pd.DataFrame, event_cols: list[str], horizons: list[int], thresholds: list[int]) -> dict:
    """Calculate exceedance probabilities for the maximum gross moves within N hours of an event."""
    log.info("Computing magnitude distributions...")
    
    # First compute excursions
    if "asset" in df_with_events.columns:
        df_exc = df_with_events.groupby("asset", group_keys=False).apply(lambda g: compute_excursions(g, horizons))
    else:
        df_exc = compute_excursions(df_with_events, horizons)

    tables = {}
    
    for event in event_cols:
        # Filter strictly where event is true
        event_df = df_exc[df_exc[event] == True]
        obs = len(event_df)
        
        if obs == 0:
            log.warning(f"No observations for event {event}")
            continue
            
        metrics = {"Observations": obs}
        
        # Evaluate up and down moves based on the event type if directional
        # For neutral events like vol expansion, we look at the absolute max move
        # Let's standardize by just looking at the MAXIMUM absolute excursion in either direction
        # for volatility, and specific directions for breakouts.
        
        for h in horizons:
            col_up = f"mfe_up_{h}h_bps"
            col_down = f"mfe_down_{h}h_bps"
            
            if "down" in event or "oversold" in event:
                # Expect downward move or upward reversal? Let's simplify:
                # we just look at the maximum excursion in the path of highest resistance.
                pass
                
            # A universal metric for how structurally "large" the state is:
            # The maximum absolute excursion in EITHER direction. This defines "gross tradable path".
            max_abs_move = np.maximum(event_df[col_up].values, event_df[col_down].abs().values)
            
            # Remove NaNs
            max_abs_move = max_abs_move[~np.isnan(max_abs_move)]
            if len(max_abs_move) == 0:
                continue
                
            metrics[f"{h}h_median_bps"] = np.median(max_abs_move)
            metrics[f"{h}h_mean_bps"] = np.mean(max_abs_move)
            
            for t in thresholds:
                exceedance_prob = np.mean(max_abs_move > t)
                metrics[f"{h}h_prob_>{t}bps"] = exceedance_prob
                
        tables[event] = metrics
        
    return tables
