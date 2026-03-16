"""Binance Public Data Ingestion (Execution-Bearing & Alpha-Bearing)
Targets: tick-level trades, aggTrades, and L2 snapshots via data.binance.vision
"""
import os
import requests
import zipfile
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

log = logging.getLogger("binance_archive")
logging.basicConfig(level=logging.INFO)

BINANCE_DATA_URL = "https://data.binance.vision/data/spot/daily/aggTrades"

def download_aggtrades(symbol: str, date: str, output_dir: Path) -> pd.DataFrame:
    """Download daily aggTrades zip from Binance public archives, extract, and return as DataFrame."""
    output_dir.mkdir(parents=True, exist_ok=True)
    symbol_upper = symbol.replace("-", "").upper()
    
    file_name = f"{symbol_upper}-aggTrades-{date}.zip"
    url = f"{BINANCE_DATA_URL}/{symbol_upper}/{file_name}"
    zip_path = output_dir / file_name
    
    if not zip_path.exists():
        log.info(f"Downloading {url} to {zip_path}")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            log.error(f"Failed to download data for {symbol} on {date}: HTTP {response.status_code}")
            return pd.DataFrame()

    csv_name = file_name.replace(".zip", ".csv")
    csv_path = output_dir / csv_name
    
    if not csv_path.exists():
        log.info(f"Extracting {zip_path}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
        except zipfile.BadZipFile:
            log.error(f"Bad zip file: {zip_path}")
            return pd.DataFrame()

    if csv_path.exists():
        log.info(f"Loading {csv_path} into DataFrame")
        # Binance aggTrades columns: agg_trade_id, price, quantity, first_trade_id, last_trade_id, timestamp, is_buyer_maker, is_best_match
        cols = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "timestamp", "is_buyer_maker", "is_best_match"]
        df = pd.read_csv(csv_path, names=cols, header=None)
        
        # Binance changed timestamps to microseconds in recent archives
        if not df.empty:
            ts_val = df['timestamp'].iloc[0]
            ts_unit = 'ms' if ts_val < 1e14 else 'us'
            df['datetime'] = pd.to_datetime(df['timestamp'], unit=ts_unit)
            
        return df
    return pd.DataFrame()

def process_microstructure(aggtrades_df: pd.DataFrame) -> dict:
    """Compile raw aggTrades into execution-bearing and alpha-bearing features (e.g. order flow imbalance)."""
    if aggtrades_df.empty:
        return {}
        
    buyer_maker_vol = aggtrades_df[aggtrades_df['is_buyer_maker']]['quantity'].sum()
    seller_maker_vol = aggtrades_df[~aggtrades_df['is_buyer_maker']]['quantity'].sum()
    
    total_vol = buyer_maker_vol + seller_maker_vol
    imbalance = (seller_maker_vol - buyer_maker_vol) / total_vol if total_vol > 0 else 0
    
    return {
        "total_volume": float(total_vol),
        "buyer_maker_volume": float(buyer_maker_vol),
        "seller_maker_volume": float(seller_maker_vol),
        "flow_imbalance": float(imbalance),
        "vwap": float((aggtrades_df['price'] * aggtrades_df['quantity']).sum() / total_vol) if total_vol > 0 else 0
    }

if __name__ == "__main__":
    out_dir = Path("data/raw/binance_aggtrades")
    yesterday = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d") # 2 days ago to ensure archive is fully available
    df = download_aggtrades("BTCUSDT", yesterday, out_dir)
    if not df.empty:
        features = process_microstructure(df)
        log.info(f"Microstructure features for {yesterday}: {features}")
