"""Normalize raw OHLCV CSVs to canonical panel schema."""
from __future__ import annotations
from pathlib import Path

import pandas as pd
import yaml

from src.utils.io import load_csv, save_parquet, ensure_dir
from src.utils.logging import get_logger

log = get_logger("normalize_bars")


def normalize(raw_csv: Path, asset_label: str) -> pd.DataFrame:
    """Read a raw CSV and return a canonical DataFrame."""
    df = load_csv(raw_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    out = pd.DataFrame({
        "asset": asset_label,
        "timestamp": df["timestamp"],
        "open": df["open"].astype(float),
        "high": df["high"].astype(float),
        "low": df["low"].astype(float),
        "close": df["close"].astype(float),
        "volume": df["volume"].astype(float),
    })
    # dollar volume proxy = close * volume
    out["dollar_volume"] = out["close"] * out["volume"]
    return out
