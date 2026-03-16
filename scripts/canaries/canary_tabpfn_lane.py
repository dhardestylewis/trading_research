"""
canary_tabpfn_lane.py — TabPFN Foundation Model Lane for SOL-USD 8h Canary

This module provides a self-contained TabPFN inference lane that runs alongside 
the existing LightGBM canary_tick system. It:
  1. Loads the latest OHLCV data for SOL-USD
  2. Builds rich perp-state features + PCA reduction  
  3. Runs TabPFN classification inference
  4. Only fires a signal when the prediction lands in the top decile (D9)
  5. Logs all signals to a dedicated CSV

Usage:
    python canary_tabpfn_lane.py                # score latest bar
    python canary_tabpfn_lane.py --dry-run      # log only, no orders
"""
from __future__ import annotations
import argparse
import time
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("canary_tabpfn")

# ── Constants ──────────────────────────────────────────────────────
LOOKBACK_BARS = 300
PCA_COMPONENTS = 47  # Matched to exp027 fitted value
MAX_TRAIN_SAMPLES = 1000
TABPFN_BINS = 10
TARGET_ASSET = "SOL-USD"
COST_BPS = 14
LOG_PATH = Path("reports/canary/tabpfn_sol_8h_log.csv")
STATE_PATH = Path("data/artifacts/canary_tabpfn_state.json")
DECILE_THRESHOLD = 9  # Only fire when in top decile (D9 out of D0-D9)

# ── Monkey Patches (required for TabPFN 0.1.9 + modern PyTorch/sklearn) ──
import torch
import torch.nn.modules.transformer
import typing
import sklearn.utils.validation

torch.nn.modules.transformer.Optional = typing.Optional
torch.nn.modules.transformer.Tensor = torch.Tensor
torch.nn.modules.transformer.Module = torch.nn.Module
torch.nn.modules.transformer.Linear = torch.nn.Linear
torch.nn.modules.transformer.Dropout = torch.nn.Dropout
torch.nn.modules.transformer.LayerNorm = torch.nn.LayerNorm
torch.nn.modules.transformer.MultiheadAttention = torch.nn.MultiheadAttention

if not hasattr(torch.nn.modules.transformer, '_get_activation_fn'):
    def _get_activation_fn(activation: str):
        if activation == "relu":
            return torch.nn.functional.relu
        elif activation == "gelu":
            return torch.nn.functional.gelu
        raise RuntimeError(f"activation should be relu/gelu, not {activation}")
    torch.nn.modules.transformer._get_activation_fn = _get_activation_fn

_orig_check_X_y = sklearn.utils.validation.check_X_y
def _patched_check_X_y(*args, **kwargs):
    if 'force_all_finite' in kwargs:
        kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
    return _orig_check_X_y(*args, **kwargs)
sklearn.utils.validation.check_X_y = _patched_check_X_y

_orig_check_array = sklearn.utils.validation.check_array
def _patched_check_array(*args, **kwargs):
    if 'force_all_finite' in kwargs:
        kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
    return _orig_check_array(*args, **kwargs)
sklearn.utils.validation.check_array = _patched_check_array

from tabpfn import TabPFNClassifier


def fetch_recent_bars(limit: int = LOOKBACK_BARS) -> pd.DataFrame:
    """Fetch recent OHLCV bars for SOL/USDT via CCXT."""
    import ccxt
    exchange = ccxt.coinbase({"enableRateLimit": True})
    exchange.load_markets()
    
    symbol = "SOL/USD"
    log.info("Fetching %d bars for %s", limit, symbol)
    
    bars = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=limit)
    if not bars:
        raise RuntimeError("No bars returned")
    
    df = pd.DataFrame(bars, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df["asset"] = TARGET_ASSET
    df["dollar_volume"] = df["close"] * df["volume"]
    df = df.drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms").reset_index(drop=True)
    
    log.info("Fetched %d bars for %s", len(df), symbol)
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build rich perp-state features from raw OHLCV."""
    from src.data.build_rich_perp_state_features import build_rich_perp_state_features
    
    rich_df = build_rich_perp_state_features(df, horizons=[8])
    
    exclude_cols = ['timestamp', 'asset', 'symbol', 'timestamp_ms', 'open', 'high', 'low', 'close', 'volume', 'dollar_volume']
    for h in [8]:
        exclude_cols.extend([
            f'fwd_ret_{h}', f'gross_move_bps_{h}',
            f'prob_tail_25_{h}', f'prob_tail_50_{h}', f'prob_tail_100_{h}'
        ])
    
    feat_cols = [c for c in rich_df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(rich_df[c])]
    log.info("Built %d features", len(feat_cols))
    return rich_df, feat_cols


def fit_and_predict(rich_df: pd.DataFrame, feat_cols: list[str]) -> dict:
    """
    Train TabPFN on the historical data and predict the latest bar.
    Returns prediction metadata for the most recent observation.
    """
    # Split: use all but the last 8 bars as training, last bar as inference
    rich_df = rich_df.sort_values('timestamp').reset_index(drop=True)
    
    target_col = 'fwd_ret_8'
    if target_col not in rich_df.columns:
        log.warning("No target column '%s' found — cannot train", target_col)
        return {"status": "no_target"}
    
    # Valid rows only
    valid = rich_df[rich_df[target_col].notna()].copy()
    if len(valid) < 50:
        log.warning("Insufficient valid rows (%d) for training", len(valid))
        return {"status": "insufficient_data"}
    
    # Train/inference split
    train_df = valid.iloc[:-1].copy()
    latest = valid.iloc[-1:].copy()
    
    # Subsample training
    if len(train_df) > MAX_TRAIN_SAMPLES:
        recent_quota = int(MAX_TRAIN_SAMPLES * 0.5)
        older_quota = MAX_TRAIN_SAMPLES - recent_quota
        recent = train_df.iloc[-recent_quota:]
        older = train_df.iloc[:-recent_quota].sample(n=older_quota, random_state=42)
        train_df = pd.concat([older, recent]).sort_index()
    
    # Clean features
    X_train = train_df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test = latest[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = train_df[target_col].values * 10000  # Convert to bps
    
    # PCA
    n_components = min(PCA_COMPONENTS, X_train.shape[1], X_train.shape[0])
    scaler = StandardScaler()
    pca = PCA(n_components=n_components, random_state=42)
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Bin targets for TabPFN classifier
    bins = pd.qcut(y_train, TABPFN_BINS, labels=False, duplicates='drop')
    bin_edges = np.percentile(y_train, np.linspace(0, 100, TABPFN_BINS + 1))
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges) - 1)]
    
    # Fit TabPFN
    clf = TabPFNClassifier(device='cpu', N_ensemble_configurations=16)
    clf.fit(X_train_pca, bins)
    
    # Predict
    proba = clf.predict_proba(X_test_pca)[0]
    predicted_bps = sum(p * c for p, c in zip(proba, bin_centers[:len(proba)]))
    
    # Determine decile of this prediction relative to training distribution
    train_preds_proba = clf.predict_proba(X_train_pca)
    train_predicted_bps = np.array([
        sum(p * c for p, c in zip(row, bin_centers[:len(row)]))
        for row in train_preds_proba
    ])
    
    # What percentile is our prediction in?
    percentile = (train_predicted_bps < predicted_bps).mean()
    decile = min(int(percentile * 10), 9)
    
    return {
        "status": "ok",
        "timestamp": str(latest['timestamp'].iloc[0]),
        "predicted_bps": float(predicted_bps),
        "decile": decile,
        "in_top_decile": decile >= DECILE_THRESHOLD,
        "n_train": len(train_df),
        "n_pca_dims": n_components,
    }


def log_signal(result: dict, dry_run: bool = True):
    """Log the TabPFN signal to the dedicated CSV."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    record = {
        "tick_time": datetime.now(timezone.utc).isoformat(),
        "timestamp": result.get("timestamp"),
        "asset": TARGET_ASSET,
        "model": "TabPFNTopDecile",
        "predicted_bps": result.get("predicted_bps"),
        "decile": result.get("decile"),
        "in_top_decile": result.get("in_top_decile"),
        "signal_fires": result.get("in_top_decile", False),
        "dry_run": dry_run,
        "n_train": result.get("n_train"),
    }
    
    df = pd.DataFrame([record])
    write_header = not LOG_PATH.exists()
    df.to_csv(LOG_PATH, mode='a', header=write_header, index=False)
    log.info("Signal logged to %s", LOG_PATH)


def save_state(result: dict):
    """Persist state between ticks."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "last_tick": datetime.now(timezone.utc).isoformat(),
        "last_prediction": result,
    }
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def tick(dry_run: bool = True):
    """Execute one TabPFN canary tick."""
    log.info("=== TabPFN SOL-USD 8h Canary Tick ===")
    
    # 1. Fetch data
    panel = fetch_recent_bars()
    
    # 2. Build features
    rich_df, feat_cols = build_features(panel)
    
    # 3. Fit and predict
    result = fit_and_predict(rich_df, feat_cols)
    
    if result["status"] != "ok":
        log.warning("Prediction failed: %s", result["status"])
        save_state(result)
        return
    
    # 4. Gate: only fire on top-decile conviction
    if result["in_top_decile"]:
        log.info("🟢 TOP DECILE SIGNAL: %.1f bps predicted (D%d)", 
                 result["predicted_bps"], result["decile"])
    else:
        log.info("· No signal: %.1f bps predicted (D%d, need D%d+)", 
                 result["predicted_bps"], result["decile"], DECILE_THRESHOLD)
    
    # 5. Log
    log_signal(result, dry_run=dry_run)
    save_state(result)
    
    log.info("=== Tick complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TabPFN SOL-USD 8h Canary Lane")
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args()
    tick(dry_run=args.dry_run)
