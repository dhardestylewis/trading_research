"""Download historical funding rates from Binance USD(S)-M Futures via CCXT."""
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

log = get_logger("download_funding")

_BATCH = 1000

def _ms(dt_str: str) -> int:
    """ISO date string -> epoch milliseconds."""
    return int(datetime.strptime(dt_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

def download_funding_history(
    exchange_id: str,
    symbol: str,
    since_ms: int,
    until_ms: int,
) -> pd.DataFrame:
    """Fetch funding rate history for a perpetual symbol."""
    exchange_cls = getattr(ccxt, exchange_id)
    # Funding implies we want the swaps market
    exchange = exchange_cls({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"}
    })
    exchange.load_markets()

    all_rows = []
    cursor = since_ms

    perp_symbol = symbol.replace("/USDT", "/USDT:USDT")
    if perp_symbol not in exchange.markets:
        # Try generic formatting
        perp_symbol = symbol.replace("/", "")
        
    log.info(f"Starting funding download for {perp_symbol} from {since_ms} to {until_ms}")
    
    while cursor < until_ms:
        try:
            # fetchFundingRateHistory returns:
            # [{'info': {...}, 'symbol': 'BTC/USDT:USDT', 'fundingRate': 0.0001, 'timestamp': 1610000000000, 'datetime': '...'}]
            rates = exchange.fetch_funding_rate_history(symbol, since=cursor, limit=_BATCH)
            if not rates:
                break
                
            all_rows.extend(rates)
            cursor = rates[-1]['timestamp'] + 1
            
            if len(rates) < _BATCH:
                break
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            log.error(f"Error fetching funding data: {e}")
            time.sleep(2)
            
    if not all_rows:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_rows)
    df = df[["timestamp", "fundingRate"]]
    df = df.rename(columns={"timestamp": "timestamp_ms", "fundingRate": "funding_rate"})
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df[df["timestamp_ms"] < until_ms].copy()
    df = df.drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms").reset_index(drop=True)
    return df

def run(cfg_path: str) -> dict[str, Path]:
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["data"]

    since_ms = _ms("2024-01-01") # Pull wide window for basis calculations
    until_ms = _ms("2026-03-15")
    
    raw_dir = Path(cfg.get("funding_raw_dir", "data/raw/crypto_funding"))
    ensure_dir(raw_dir)

    paths = {}
    # We will just pull for the base assets needed in RV and the main panel
    # The config doesn't explicitly name a universe, but implies SOL, BTC, APT, SUI, ETH
    target_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "APT/USDT", "SUI/USDT"]

    for symbol in target_symbols:
        label = symbol.split("/")[0]
        log.info(f"Downloading Funding for {symbol}")
        
        out = raw_dir / f"{label}_funding.csv"
        if out.exists():
            log.info(f"File {out} exists, skipping.")
            paths[label] = out
            continue
            
        df = download_funding_history("binanceusdm", symbol, since_ms, until_ms)
        if not df.empty:
            save_csv(df, out)
            log.info(f"  -> {len(df)} funding intervals saved to {out}")
            paths[label] = out

    return paths

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent)) 
    run(sys.argv[1] if len(sys.argv) > 1 else "configs/experiments/crypto_1h_exp016.yaml")
