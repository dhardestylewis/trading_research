"""Exp021 Data Backfiller
Fetches exactly 6 months of execution-bearing Binance tick data to unblock immediate ML training.
"""
import os
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from src.data.ingestion.binance_archive import download_aggtrades, process_microstructure
import pandas as pd

log = logging.getLogger("backfiller")
logging.basicConfig(level=logging.INFO)

def backfill_binance(symbols: list, days: int):
    out_dir = Path("data/raw/binance_aggtrades")
    
    end_date = datetime.now() - timedelta(days=2) # archives delay
    start_date = end_date - timedelta(days=days)
    
    date_list = [start_date + timedelta(days=x) for x in range((end_date-start_date).days + 1)]
    
    all_features = []
    
    for symbol in symbols:
        log.info(f"--- Backfilling {symbol} for {days} days ---")
        for dt_obj in date_list:
            dt_str = dt_obj.strftime("%Y-%m-%d")
            try:
                df = download_aggtrades(symbol, dt_str, out_dir)
                if not df.empty:
                    feats = process_microstructure(df)
                    feats["asset"] = symbol
                    feats["date"] = dt_str
                    all_features.append(feats)
            except Exception as e:
                log.error(f"Failed to process {symbol} on {dt_str}: {e}")
                
    if all_features:
        feat_df = pd.DataFrame(all_features)
        summary_path = out_dir / "microstructure_features_summary.parquet"
        
        # If exists, append
        if summary_path.exists():
            old_df = pd.read_parquet(summary_path)
            feat_df = pd.concat([old_df, feat_df]).drop_duplicates(subset=["asset", "date"], keep="last")
            
        feat_df.to_parquet(summary_path)
        log.info(f"Saved {len(feat_df)} total daily rows to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Backfill Script")
    parser.add_argument("--days", type=int, default=7, help="Number of days to backfill (max 180)")
    args = parser.parse_args()
    
    universe = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    log.info(f"Starting backfill for past {args.days} days on {universe}")
    backfill_binance(universe, args.days)
