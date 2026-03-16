"""Download historical funding rates from Binance Vision public data archives (monthly)."""
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

log = get_logger("download_funding_vision")

def download_funding(symbol: str, date: str, raw_dir: Path) -> Path | None:
    """Download monthly fundingRate zip from Binance Vision and extract."""
    base_url = "https://data.binance.vision/data/futures/um/monthly/fundingRate"
    clean_symbol = symbol.replace("/", "").replace("-", "")
    filename = f"{clean_symbol}-fundingRate-{date}.zip"
    url = f"{base_url}/{clean_symbol}/{filename}"
    
    zip_path = raw_dir / filename
    csv_path = raw_dir / f"{clean_symbol}-fundingRate-{date}.csv"
    
    if csv_path.exists():
        return csv_path
        
    try:
        urllib.request.urlretrieve(url, zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
            
        zip_path.unlink()
        return csv_path
    except urllib.error.HTTPError as e:
        if zip_path.exists():
            zip_path.unlink()
        return None
    except Exception as e:
        if zip_path.exists():
            zip_path.unlink()
        return None

def run(cfg_path: str):
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["data"]

    months = pd.date_range(start="2024-01-01", end="2026-03-01", freq="MS").strftime("%Y-%m").tolist()
    
    raw_dir = Path(cfg.get("funding_raw_dir", "data/raw/crypto_funding"))
    ensure_dir(raw_dir)

    target_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "APT/USDT", "SUI/USDT"]

    for symbol in target_symbols:
        log.info(f"Processing funding for: {symbol}")
        asset_dfs = []
        
        for date_str in months:
            csv_path = download_funding(symbol, date_str, raw_dir)
            if csv_path:
                try:
                    df = pd.read_csv(csv_path)
                    if df.shape[1] == 3:
                        df.columns = ["calc_time", "funding_rate", "symbol"]
                        asset_dfs.append(df)
                    elif df.shape[1] == 4:
                         df = df.iloc[:, [0, 1, 2]]
                         if "funding_rate" in df.columns or "fundingRate" in df.columns:
                            pass
                         else:
                            df.columns = ["calc_time", "funding_rate", "symbol"]
                         asset_dfs.append(df)
                except Exception as e:
                    pass
                    
        if asset_dfs:
            combined = pd.concat(asset_dfs, ignore_index=True)
            combined = combined.drop_duplicates(subset=["calc_time"])
            combined = combined.sort_values("calc_time").reset_index(drop=True)
            
            combined["timestamp_ms"] = combined["calc_time"]
            
            # Need to handle potential parsing issues for type safety
            combined["funding_rate"] = pd.to_numeric(combined["funding_rate"], errors="coerce")
            combined = combined.dropna(subset=["funding_rate"])
            
            label = symbol.split("/")[0]
            out_file = raw_dir / f"{label}_funding.parquet"
            combined[["timestamp_ms", "funding_rate"]].to_parquet(out_file, index=False)
            log.info(f"Saved {len(combined)} funding periods for {symbol} to {out_file}")

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent.parent)) 
    run(sys.argv[1] if len(sys.argv) > 1 else "configs/experiments/crypto_1h_exp016.yaml")
