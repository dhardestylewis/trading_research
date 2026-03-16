"""Build Flow Bars from Raw Binance aggTrades
Constructs short-horizon flow bars (e.g., 1S, 5S) from tick-level data, 
extracting microstructural features and forward markouts for exp022.
"""
import os
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path

log = logging.getLogger("build_flow_bars")
logging.basicConfig(level=logging.INFO)

def process_aggtrades_file(csv_path: Path, freq: str) -> pd.DataFrame:
    """Read raw aggTrades CSV, resample to `freq`, and calculate flow features."""
    log.info(f"Processing {csv_path} at frequency {freq}")
    
    # agg_trade_id, price, quantity, first_trade_id, last_trade_id, timestamp, is_buyer_maker, is_best_match
    cols = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "timestamp", "is_buyer_maker", "is_best_match"]
    df = pd.read_csv(csv_path, names=cols, header=None)
    if df.empty:
        return pd.DataFrame()

    # Determine timestamp unit based on the first row's value
    ts_val = df['timestamp'].iloc[0]
    ts_unit = 'ms' if ts_val < 1e14 else 'us'
    df['datetime'] = pd.to_datetime(df['timestamp'], unit=ts_unit)
    
    # Calculate maker volumes
    df['buyer_maker_vol'] = np.where(df['is_buyer_maker'], df['quantity'], 0)
    df['seller_maker_vol'] = np.where(~df['is_buyer_maker'], df['quantity'], 0)
    df['notional'] = df['price'] * df['quantity']
    df['trade_count'] = 1

    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)

    # Resample
    resampled = df.resample(freq).agg({
        'price': 'last',           # Use last price as close
        'quantity': 'sum',
        'buyer_maker_vol': 'sum',
        'seller_maker_vol': 'sum',
        'notional': 'sum',
        'trade_count': 'sum'
    })
    
    # Forward fill price if there are gaps with no trades, but volume/count is 0
    resampled['price'] = resampled['price'].ffill()
    resampled.fillna({'quantity': 0, 'buyer_maker_vol': 0, 'seller_maker_vol': 0, 'notional': 0, 'trade_count': 0}, inplace=True)

    # Calculate Flow Imbalance
    resampled['flow_imbalance'] = np.where(
        resampled['quantity'] > 0,
        (resampled['seller_maker_vol'] - resampled['buyer_maker_vol']) / resampled['quantity'],
        0
    )
    
    # Calculate VWAP
    resampled['vwap'] = np.where(
        resampled['quantity'] > 0,
        resampled['notional'] / resampled['quantity'],
        resampled['price']
    )

    # Calculate markouts (forward returns in basis points)
    # 1s, 5s, 30s, 5m (300s)
    freq_s = pd.Timedelta(freq).total_seconds()
    
    for fwd_s in [1, 5, 30, 300]:
        shift_periods = int(fwd_s / freq_s)
        if shift_periods > 0:
            resampled[fwd_s] = resampled['price'].shift(-shift_periods)
            resampled[f'markout_{fwd_s}s'] = (resampled[fwd_s] / resampled['price'] - 1) * 10000
    
    # Clean up intermediate columns
    cols_to_drop = [c for c in [1, 5, 30, 300] if c in resampled.columns]
    resampled.drop(columns=cols_to_drop, inplace=True)

    # Drop rows at the end where markout cannot be computed
    resampled.dropna(subset=[f'markout_{fwd_s}s' for fwd_s in [1, 5, 30, 300] if int(fwd_s / freq_s) > 0], inplace=True)

    return resampled

def build_dataset(symbol: str, freq: str, raw_dir: Path, out_dir: Path):
    """Process all available CSVs for a symbol and save to a parquet file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    symbol_upper = symbol.replace("-", "").upper()
    
    # Find all csvs for this symbol
    all_csvs = sorted(list(raw_dir.glob(f"{symbol_upper}-aggTrades-*.csv")))
    
    if not all_csvs:
        log.warning(f"No CSVs found for {symbol} in {raw_dir}")
        return

    log.info(f"Found {len(all_csvs)} daily files for {symbol}. Building flow bars...")
    
    dfs = []
    for csv_path in all_csvs:
        df = process_aggtrades_file(csv_path, freq)
        if not df.empty:
            dfs.append(df)
            
    if dfs:
        final_df = pd.concat(dfs)
        out_file = out_dir / f"{symbol_upper}_{freq}_flow.parquet"
        final_df.to_parquet(out_file)
        log.info(f"Saved {len(final_df)} bars to {out_file}")
    else:
        log.error("Failed to build any valid flow bars.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Build Flow Bars")
    parser.add_argument("--freq", type=str, default="1S", help="Resampling frequency (e.g., 1S, 5S)")
    args = parser.parse_args()
    
    raw_dir = Path("data/raw/binance_aggtrades")
    out_dir = Path("data/processed/flow_bars")
    
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        build_dataset(sym, args.freq, raw_dir, out_dir)
