"""Download historical trades from Binance Vision public data archives.

Usage:
    python -m src.data.download_binance_trades configs/experiments/crypto_ticks_exp014.yaml
"""
import sys
import os
import zipfile
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta
import yaml
import pandas as pd

from src.utils.io import ensure_dir
from src.utils.logging import get_logger

log = get_logger("download_binance_trades")

def download_trades(symbol: str, date: str, raw_dir: Path) -> Path | None:
    """Download daily trades zip from Binance Vision and extract."""
    base_url = "https://data.binance.vision/data/spot/daily/trades"
    clean_symbol = symbol.replace("/", "").replace("-", "")
    filename = f"{clean_symbol}-trades-{date}.zip"
    url = f"{base_url}/{clean_symbol}/{filename}"
    
    zip_path = raw_dir / filename
    csv_path = raw_dir / f"{clean_symbol}-trades-{date}.csv"
    
    if csv_path.exists():
        log.info(f"Already have {csv_path.name}")
        return csv_path
        
    try:
        log.info(f"Downloading {url} to {zip_path}")
        urllib.request.urlretrieve(url, zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
            
        zip_path.unlink() # Cleanup zip
        return csv_path
    except urllib.error.HTTPError as e:
        log.error(f"HTTP Error {e.code} for {url} - Data might not exist for this date yet.")
        if zip_path.exists():
            zip_path.unlink()
        return None
    except Exception as e:
        log.error(f"Error downloading {url}: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return None

def run(cfg_path: str):
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["data"]

    start_date = datetime.strptime(cfg["start_date"], "%Y-%m-%d").date()
    end_date = datetime.strptime(cfg["end_date"], "%Y-%m-%d").date()
    # shift dates backward a bit because binance vision data is usually 1-2 days delayed
    # and today is Mar 15 2026. The config asks for 2026-03-01 to 2026-03-05. That's fine.
    
    raw_dir = Path(cfg["raw_dir"])
    ensure_dir(raw_dir)

    for symbol in cfg["asset_universe"]:
        log.info(f"Processing symbol: {symbol}")
        
        current_date = start_date
        asset_dfs = []
        
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            csv_path = download_trades(symbol, date_str, raw_dir)
            
            if csv_path:
                try:
                    # Binance trades CSV columns (no header in zip):
                    # id, price, qty, quoteQty, time, isBuyerMaker, isBestMatch
                    df = pd.read_csv(
                        csv_path, 
                        names=["trade_id", "price", "qty", "quote_qty", "timestamp_ms", "is_buyer_maker", "is_best_match"]
                    )
                    asset_dfs.append(df)
                except Exception as e:
                    log.error(f"Error reading {csv_path}: {e}")
            
            current_date += timedelta(days=1)
            
        if asset_dfs:
            combined = pd.concat(asset_dfs, ignore_index=True)
            combined = combined.sort_values("timestamp_ms").reset_index(drop=True)
            
            label = cfg.get("asset_labels", {}).get(symbol, symbol.replace("/", "-"))
            out_file = raw_dir / f"{label}_combined.parquet"
            # save as parquet to save space and read faster
            combined.to_parquet(out_file, index=False)
            log.info(f"Saved {len(combined)} trades for {symbol} to {out_file}")

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent.parent)) 
    run(sys.argv[1] if len(sys.argv) > 1 else "configs/experiments/crypto_ticks_exp014.yaml")
