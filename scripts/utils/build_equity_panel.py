"""
build_equity_panel.py — Equities/FX data pipeline for uncorrelated market testing

Downloads hourly OHLCV data for a diversified set of uncorrelated assets
using yfinance, then builds the same feature set used for crypto.

Assets: SPY, QQQ, GLD, TLT, DXY (UUP proxy), USO, FXI
These have low correlation with crypto and with each other.

Run with: python build_equity_panel.py
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("equity_panel")

# Diversified uncorrelated universe
EQUITIES = {
    "SPY": "S&P 500 ETF — US large-cap equities",
    "QQQ": "Nasdaq 100 ETF — US tech equities",
    "GLD": "Gold ETF — safe haven commodity",
    "TLT": "20+ Year Treasury ETF — US long bonds",
    "UUP": "US Dollar Index ETF — FX proxy for DXY",
    "USO": "US Oil Fund — crude oil commodity",
    "FXI": "China Large-Cap ETF — non-US equities",
}

OUTPUT_DIR = Path("data/processed/equity_panel")


def download_data(period: str = "2y", interval: str = "1h") -> pd.DataFrame:
    """Download hourly OHLCV for all assets."""
    import yfinance as yf

    all_dfs = []
    for ticker, desc in EQUITIES.items():
        log.info(f"Downloading {ticker} ({desc})...")
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            if data.empty:
                log.warning(f"No data for {ticker}")
                continue

            # Flatten multi-level columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            df = data.reset_index()
            # Handle both 'Datetime' and 'Date' column names
            ts_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
            df = df.rename(columns={
                ts_col: 'timestamp',
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            })
            df['asset'] = ticker
            df['dollar_volume'] = df['close'] * df['volume']
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

            all_dfs.append(df[['timestamp', 'asset', 'open', 'high', 'low', 'close', 'volume', 'dollar_volume']])
            log.info(f"  {ticker}: {len(df)} bars")
        except Exception as e:
            log.error(f"Failed to download {ticker}: {e}")

    if not all_dfs:
        raise RuntimeError("No data downloaded")

    panel = pd.concat(all_dfs, ignore_index=True)
    panel = panel.sort_values(['asset', 'timestamp']).reset_index(drop=True)
    log.info(f"Total panel: {len(panel)} bars across {panel['asset'].nunique()} assets")
    return panel


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Build rich features per asset, same as crypto pipeline."""
    from src.data.build_rich_perp_state_features import build_rich_perp_state_features

    all_rich = []
    for asset in panel['asset'].unique():
        asset_df = panel[panel['asset'] == asset].copy().sort_values('timestamp').reset_index(drop=True)
        if len(asset_df) < 100:
            log.warning(f"Skipping {asset}: only {len(asset_df)} bars")
            continue

        try:
            rich = build_rich_perp_state_features(asset_df, horizons=[4, 8])
            all_rich.append(rich)
            log.info(f"  {asset}: {len(rich)} rows, {rich.shape[1]} cols")
        except Exception as e:
            log.error(f"Feature build failed for {asset}: {e}")

    if not all_rich:
        raise RuntimeError("No features built")

    result = pd.concat(all_rich, ignore_index=True)
    log.info(f"Final panel: {len(result)} rows, {result.shape[1]} cols")
    return result


def compute_correlations(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise return correlations between all assets."""
    # Compute 8-bar returns per asset
    returns = {}
    for asset in panel['asset'].unique():
        asset_df = panel[panel['asset'] == asset].sort_values('timestamp')
        rets = asset_df['close'].pct_change(8).dropna()
        returns[asset] = rets.values[:min(len(rets), 5000)]  # Align lengths

    # Align by cutting to shortest length
    min_len = min(len(v) for v in returns.values())
    aligned = {k: v[:min_len] for k, v in returns.items()}

    corr_df = pd.DataFrame(aligned).corr()
    log.info(f"\n8-bar return correlations:\n{corr_df.round(2)}")
    return corr_df


def main():
    # Download
    panel = download_data()

    # Save raw
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUTPUT_DIR / "panel.parquet", index=False)
    log.info(f"Saved raw panel to {OUTPUT_DIR / 'panel.parquet'}")

    # Correlations (before features for diagnostics)
    corr = compute_correlations(panel)
    corr.to_csv(OUTPUT_DIR / "correlations.csv")

    # Features
    rich = build_features(panel)
    rich.to_parquet(OUTPUT_DIR / "panel_rich.parquet", index=False)
    log.info(f"Saved enriched panel to {OUTPUT_DIR / 'panel_rich.parquet'}")

    # Summary
    log.info("\n=== SUMMARY ===")
    for asset in panel['asset'].unique():
        n = len(panel[panel['asset'] == asset])
        log.info(f"  {asset}: {n} bars")
    log.info(f"Total: {len(panel)} bars, {len(rich)} enriched rows")
    mean_corr = corr.values[np.triu_indices_from(corr.values, k=1)].mean()
    log.info(f"Mean pairwise correlation: {mean_corr:.3f}")


if __name__ == "__main__":
    main()
