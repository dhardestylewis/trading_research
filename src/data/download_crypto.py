"""Download historical OHLCV data from Binance via CCXT.

Usage:
    python -m src.data.download_crypto configs/data/crypto_1h.yaml
"""
from __future__ import annotations
import time
from pathlib import Path
from datetime import datetime, timezone

import ccxt
import pandas as pd
import yaml

from src.utils.io import ensure_dir, save_csv
from src.utils.logging import get_logger

log = get_logger("download_crypto")

# Binance returns max 1000 candles per request
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
    """Fetch all OHLCV bars for *symbol* between two epoch-ms timestamps."""
    exchange_cls = getattr(ccxt, exchange_id)
    exchange = exchange_cls({"enableRateLimit": True})
    exchange.load_markets()

    all_rows: list[list] = []
    cursor = since_ms

    while cursor < until_ms:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=_BATCH)
        if not bars:
            break
        all_rows.extend(bars)
        # advance cursor past last bar
        cursor = bars[-1][0] + 1
        if len(bars) < _BATCH:
            break
        time.sleep(exchange.rateLimit / 1000)  # respect rate limit

    df = pd.DataFrame(all_rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df[df["timestamp_ms"] < until_ms].copy()
    df = df.drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms").reset_index(drop=True)
    return df


def run(cfg_path: str) -> dict[str, Path]:
    """Download OHLCV for all assets in the data config. Returns {label: csv_path}."""
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    since_ms = _ms(cfg["start_date"])
    until_ms = _ms(cfg["end_date"])
    raw_dir = Path(cfg["raw_dir"])
    ensure_dir(raw_dir)

    labels = cfg.get("asset_labels", {})
    paths: dict[str, Path] = {}

    for symbol in cfg["asset_universe"]:
        label = labels.get(symbol, symbol.replace("/", "-"))
        log.info("Downloading %s (%s) | %s → %s", symbol, label, cfg["start_date"], cfg["end_date"])
        df = download_ohlcv(cfg["exchange_id"], symbol, cfg["bar_size"], since_ms, until_ms)
        out = raw_dir / f"{label}.csv"
        save_csv(df, out)
        log.info("  → %d bars saved to %s", len(df), out)
        paths[label] = out

    return paths


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "configs/data/crypto_1h.yaml")
