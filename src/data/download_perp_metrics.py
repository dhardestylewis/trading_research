"""Download historical Open Interest and Mark Price from Binance USD(S)-M Futures."""
from __future__ import annotations
import time
from pathlib import Path
from datetime import datetime, timezone
import requests
import pandas as pd
import yaml

from src.utils.io import ensure_dir, save_csv
from src.utils.logging import get_logger

log = get_logger("download_perp_metrics")

_BATCH_OI = 500
_BATCH_MARK = 1500

def _ms(dt_str: str) -> int:
    """ISO date string -> epoch milliseconds."""
    return int(datetime.strptime(dt_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

def fetch_open_interest(symbol: str, since_ms: int, until_ms: int, period: str = "15m") -> pd.DataFrame:
    """
    Fetch open interest history from Binance.
    Period options: "5m","15m","30m","1h","2h","4h","6h","12h","1d"
    """
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    all_rows = []
    cursor = since_ms
    
    clean_symbol = symbol.replace("/", "").replace("-", "")
    
    while cursor < until_ms:
        params = {
            "symbol": clean_symbol,
            "period": period,
            "limit": _BATCH_OI,
            "startTime": cursor,
            "endTime": until_ms
        }
        res = requests.get(url, params=params)
        
        if res.status_code != 200:
            log.error(f"Error fetching OI for {clean_symbol}: {res.text}")
            time.sleep(2)
            continue
            
        data = res.json()
        if not data:
            break
            
        all_rows.extend(data)
        
        last_ts = data[-1]["timestamp"]
        if last_ts == cursor:
            # Prevent infinite loop if API returns same timestamp
            cursor += _ms_for_period(period)
        else:
            cursor = last_ts + 1
            
        if len(data) < _BATCH_OI:
            break
        time.sleep(0.5)
        
    if not all_rows:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_rows)
    df = df[["timestamp", "sumOpenInterestValue"]]
    df = df.rename(columns={"timestamp": "timestamp_ms", "sumOpenInterestValue": "open_interest_notional"})
    df["open_interest_notional"] = df["open_interest_notional"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms").reset_index(drop=True)
    return df

def fetch_mark_price(symbol: str, since_ms: int, until_ms: int, interval: str = "15m") -> pd.DataFrame:
    """Fetch mark price klines from Binance."""
    url = "https://fapi.binance.com/fapi/v1/markPriceKlines"
    all_rows = []
    cursor = since_ms
    
    clean_symbol = symbol.replace("/", "").replace("-", "")
    
    while cursor < until_ms:
        params = {
            "symbol": clean_symbol,
            "interval": interval,
            "limit": _BATCH_MARK,
            "startTime": cursor,
            "endTime": until_ms
        }
        res = requests.get(url, params=params)
        
        if res.status_code != 200:
            log.error(f"Error fetching Mark Price for {clean_symbol}: {res.text}")
            time.sleep(2)
            continue
            
        data = res.json()
        if not data:
            break
            
        # Structure: [Open time, Open, High, Low, Close, Ignore, Close time, ...]
        all_rows.extend(data)
        
        last_ts = data[-1][0]
        if last_ts == cursor:
            cursor += _ms_for_period(interval)
        else:
            cursor = last_ts + 1
            
        if len(data) < _BATCH_MARK:
            break
        time.sleep(0.2)
        
    if not all_rows:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_rows, columns=["timestamp_ms", "open", "high", "low", "close", "ignore", "close_time_ms"] + [f"ign_{i}" for i in range(5)])
    df = df[["timestamp_ms", "close"]]
    df = df.rename(columns={"close": "mark_price"})
    df["mark_price"] = df["mark_price"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms").reset_index(drop=True)
    return df

def _ms_for_period(period: str) -> int:
    if period.endswith("m"): return int(period[:-1]) * 60 * 1000
    if period.endswith("h"): return int(period[:-1]) * 3600 * 1000
    if period.endswith("d"): return int(period[:-1]) * 86400 * 1000
    return 60 * 1000

def run(cfg_path: str):
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["data"]

    # Let's pull 1h and 15m to be safe? We will focus on 15m to match our potential resolution.
    # Open interest limits history to ~30 days on lower timeframes sometimes? No, Binance futures holds more.
    since_ms = _ms("2024-01-01")
    until_ms = _ms("2026-03-15")
    
    raw_dir = Path("data/raw/crypto_perp_metrics")
    ensure_dir(raw_dir)

    target_symbols = cfg.get("asset_universe", [])
    if not target_symbols:
        target_symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]

    period = "15m"

    for symbol in target_symbols:
        label = symbol.replace("/USDT", "").replace("-USD", "")
        log.info(f"Downloading Metrics for {symbol}")
        
        oi_out = raw_dir / f"{label}_open_interest_{period}.csv"
        if not oi_out.exists():
            oi_df = fetch_open_interest(symbol, since_ms, until_ms, period=period)
            if not oi_df.empty:
                save_csv(oi_df, oi_out)
                log.info(f"  -> {len(oi_df)} OI intervals saved to {oi_out}")
        
        mp_out = raw_dir / f"{label}_mark_price_{period}.csv"
        if not mp_out.exists():
            mp_df = fetch_mark_price(symbol, since_ms, until_ms, interval=period)
            if not mp_df.empty:
                save_csv(mp_df, mp_out)
                log.info(f"  -> {len(mp_df)} Mark Price intervals saved to {mp_out}")

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent)) 
    run(sys.argv[1] if len(sys.argv) > 1 else "configs/experiments/crypto_exp025.yaml")
