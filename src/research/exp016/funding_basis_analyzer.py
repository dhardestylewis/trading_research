"""Analyze perpetual funding basis edge."""
import pandas as pd
import numpy as np
from src.utils.logging import get_logger

log = get_logger("funding_basis")

def evaluate_funding_opportunities(funding_df: pd.DataFrame, target_annualized_percent: float) -> pd.DataFrame:
    """
    Find contiguous periods or states where the structured funding rate 
    exceeds the target annualized yield.
    """
    log.info("Evaluating structural funding basis.")
    df = funding_df.copy()
    
    if df.empty:
        return pd.DataFrame()
        
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    
    # Funding rate is typically per 8 hours.
    # Annualized = funding_rate * 3 * 365 * 100
    df["annualized_yield_pct"] = df["funding_rate"] * 3 * 365 * 100
    
    # Identify environments where the trailing 72h average yield > target
    df["trailing_72h_yield"] = df["annualized_yield_pct"].rolling(9).mean() # 9 intervals * 8h = 72h
    
    df["passes_target"] = df["trailing_72h_yield"] > target_annualized_percent
    
    return df
