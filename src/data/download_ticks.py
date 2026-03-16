"""Download 1s OHLCV data from Binance via CCXT for microstructure research.

Usage:
    python -m src.data.download_ticks configs/experiments/crypto_ticks_exp014.yaml
"""
from __future__ import annotations
import time
from pathlib import Path
from datetime import datetime, timezone
import os

import ccxt
import pandas as pd
import yaml

from src.utils.io import ensure_dir, save_csv
from src.utils.logging import get_logger

log = get_logger("download_ticks")

_BATCH = 1000

def _ms(dt_str: str) -> int:
    """ISO date string -> epoch milliseconds."""
    return int(datetime.strptime(dt_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

def download_ohlcv(
    exchange_id: str,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
) -> pd.DataFrame:
    """Fetch all bars for symbol between two epoch-ms timestamps."""
    exchange_cls = getattr(ccxt, exchange_id)
    exchange = exchange_cls({"enableRateLimit": True})
    exchange.load_markets()

    all_rows = []
    cursor = since_ms

    log.info(f"Starting download for {symbol} {timeframe} from {since_ms} to {until_ms}")
    while cursor < until_ms:
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=_BATCH)
            if not bars:
                break
                
            all_rows.extend(bars)
            cursor = bars[-1][0] + 1
            
            if len(all_rows) % 10000 == 0:
                log.info(f"Downloaded {len(all_rows)} bars so far for {symbol}...")
                
            if len(bars) < _BATCH:
                break
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            log.error(f"Error fetching data: {e}")
            time.sleep(2) # Backoff
            
    df = pd.DataFrame(all_rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df[df["timestamp_ms"] < until_ms].copy()
    df = df.drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms").reset_index(drop=True)
    return df

def run(cfg_path: str) -> dict[str, Path]:
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["data"]

    since_ms = _ms(cfg["start_date"])
    until_ms = _ms(cfg["end_date"])
    raw_dir = Path(cfg["raw_dir"])
    ensure_dir(raw_dir)

    labels = cfg.get("asset_labels", {})
    paths = {}

    for symbol in cfg["asset_universe"]:
        label = labels.get(symbol, symbol.replace("/", "-"))
        log.info(f"Downloading {symbol} ({label}) | {cfg['start_date']} -> {cfg['end_date']}")
        
        out = raw_dir / f"{label}.csv"
        if out.exists():
            log.info(f"File {out} already exists, skipping download.")
            paths[label] = out
            continue
            
        df = download_ohlcv(cfg["exchange_id"], symbol, cfg["bar_size"], since_ms, until_ms)
        save_csv(df, out)
        log.info(f"  -> {len(df)} bars saved to {out}")
        paths[label] = out

    return paths

if __name__ == "__main__":
    import sys
    # For relative imports to work when run as script
    sys.path.append(str(Path(__file__).parent.parent.parent)) 
    run(sys.argv[1] if len(sys.argv) > 1 else "configs/experiments/crypto_ticks_exp014.yaml")
