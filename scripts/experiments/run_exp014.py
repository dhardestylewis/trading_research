"""Run exp014: Microstructure Feasibility Map."""
import yaml
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from src.research.exp014.touch_to_fill import run_touch_simulation
from src.research.exp014.markout_analysis import build_markout_table
from src.reporting.exp014_report import generate_report
from src.utils.logging import get_logger

log = get_logger("run_exp014")

def generate_heuristic_signals(trades_df: pd.DataFrame, freq: str='1h') -> pd.DataFrame:
    """Generate hourly signals for testing queue position."""
    # Group trades by hourly bins to extract simple hourly signals
    start_ts = trades_df["timestamp_ms"].min()
    end_ts = trades_df["timestamp_ms"].max()
    
    # Create hourly targets
    ts_range = np.arange(start_ts, end_ts, 3600000) # 1h in ms
    
    # We will just alternate long and short to simulate passive provision
    directions = np.where((ts_range // 3600000) % 2 == 0, 1, -1)
    
    # Find the price AT that timestamp
    ts_arr = trades_df["timestamp_ms"].values
    prices = trades_df["price"].values
    
    target_prices = []
    
    for t in ts_range:
        idx = np.searchsorted(ts_arr, t)
        if idx < len(ts_arr):
            target_prices.append(prices[idx])
        else:
            target_prices.append(prices[-1])
            
    return pd.DataFrame({
        "timestamp_ms": ts_range,
        "direction": directions,
        "target_entry_price": target_prices
    })

def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
        
    data_cfg = cfg["data"]
    markout_cfg = cfg["markouts"]
    exec_cfg = cfg["execution"]
    
    raw_dir = Path(data_cfg["raw_dir"])
    out_dir = Path(data_cfg["processed_dir"])
    
    results = {}
    
    for symbol in data_cfg["asset_universe"]:
        label = data_cfg.get("asset_labels", {}).get(symbol, symbol.replace("/", "-"))
        parquet_path = raw_dir / f"{label}_combined.parquet"
        
        if not parquet_path.exists():
            log.warning(f"Data not found for {symbol} at {parquet_path}")
            continue
            
        log.info(f"Loading trades for {symbol}...")
        trades_df = pd.read_parquet(parquet_path)
        log.info(f"Loaded {len(trades_df)} trades.")
        
        # Generate some synthetic signals 
        sig_df = generate_heuristic_signals(trades_df)
        log.info(f"Testing fills on {len(sig_df)} hourly target levels.")
        
        # Run Touch to fill
        sim_res = run_touch_simulation(trades_df, sig_df, exec_cfg["queue_haircut_bps"])
        
        # Build Markout Table
        table_df = build_markout_table(trades_df, sig_df, sim_res, markout_cfg["horizons_sec"])
        
        results[symbol] = table_df
        log.info(f"\n{table_df.to_string()}")
        
    report_path = generate_report(results, out_dir)
    log.info(f"Report saved to {report_path}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else 'configs/experiments/crypto_ticks_exp014.yaml')
