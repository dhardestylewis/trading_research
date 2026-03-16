"""Runner for exp025: Event-Conditioned Perp-State Alpha."""
from __future__ import annotations
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.build_perp_features import build_perp_features
from src.models.multi_horizon_model import MultiHorizonModel
from src.reporting.exp025_report import generate_report
from src.utils.io import ensure_dir, load_parquet
from src.utils.logging import get_logger

log = get_logger("run_exp025")

def load_and_merge_data(cfg: dict) -> pd.DataFrame:
    """Load base panel and merge perp metrics."""
    panel_path = Path(cfg["data"]["panel_path"])
    raw_metrics_dir = Path("data/raw/crypto_perp_metrics")
    
    if not panel_path.exists():
        raise FileNotFoundError(f"Base panel not found at {panel_path}")
        
    log.info(f"Loading base panel from {panel_path}")
    panel = load_parquet(panel_path)
    
    # Identify resolution
    time_diffs = panel["timestamp"].diff().dropna()
    mode_diff = time_diffs.mode()[0]
    
    merged_assets = []
    
    for symbol in cfg["data"]["asset_universe"]:
        label = symbol.replace("/", "-")
        asset_panel = panel[panel["asset"] == label].copy()
        
        # Load perp metrics
        oi_path = raw_metrics_dir / f"{label.replace('-USD', '')}_open_interest_15m.csv"
        mp_path = raw_metrics_dir / f"{label.replace('-USD', '')}_mark_price_15m.csv"
        funding_path = Path("data/raw/crypto_funding") / f"{label.replace('-USD', '')}_funding.csv"
        
        # OI
        if oi_path.exists():
            oi_df = pd.read_csv(oi_path)
            oi_df["timestamp"] = pd.to_datetime(oi_df["timestamp"], utc=True)
            # Resample to align with panel by taking last value
            oi_df = oi_df.set_index("timestamp").resample(mode_diff).last().reset_index()
            asset_panel = pd.merge(asset_panel, oi_df[["timestamp", "open_interest_notional"]], on="timestamp", how="left")
        else:
            asset_panel["open_interest_notional"] = np.nan
            
        # Mark Price
        if mp_path.exists():
            mp_df = pd.read_csv(mp_path)
            mp_df["timestamp"] = pd.to_datetime(mp_df["timestamp"], utc=True)
            mp_df = mp_df.set_index("timestamp").resample(mode_diff).last().reset_index()
            asset_panel = pd.merge(asset_panel, mp_df[["timestamp", "mark_price"]], on="timestamp", how="left")
        else:
            asset_panel["mark_price"] = np.nan
            
        # Funding
        if funding_path.exists():
            f_df = pd.read_csv(funding_path)
            f_df["timestamp"] = pd.to_datetime(f_df["timestamp"], utc=True)
            # Resample taking last available funding
            f_df = f_df.set_index("timestamp").resample(mode_diff).last().ffill().reset_index()
            asset_panel = pd.merge(asset_panel, f_df[["timestamp", "funding_rate"]], on="timestamp", how="left")
        else:
            asset_panel["funding_rate"] = np.nan
            
        merged_assets.append(asset_panel)
        
    df = pd.concat(merged_assets, ignore_index=True)
    return df

def run_experiment(cfg_path: str):
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    log.info(f"Starting {cfg['experiment_name']}")
    
    # 1. Load Data
    panel = load_and_merge_data(cfg)
    
    # Extract horizons for feature building
    horizons = [h["bars"] for h in cfg["horizons"]]
    
    # 2. Build Features
    df = build_perp_features(panel, horizons)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 3. Time Series Walk Forward
    wf_cfg = cfg["model"]["walk_forward"]
    min_train = wf_cfg["min_train_bars"]
    val_bars = wf_cfg["val_bars"]
    step_bars = wf_cfg["step_bars"]
    
    times = sorted(df["timestamp"].unique())
    n_times = len(times)
    
    model = MultiHorizonModel(cfg)
    all_results = []
    
    cost_bps = cfg["execution"]["round_trip_bps"]
    
    for start_idx in range(min_train, n_times, step_bars):
        end_idx = min(start_idx + step_bars, n_times)
        if end_idx - start_idx < 10:
            break
            
        train_start = 0
        train_end = start_idx
        
        train_times = times[train_start:train_end]
        test_times = times[start_idx:end_idx]
        
        train_df = df[df["timestamp"].isin(train_times)]
        test_df = df[df["timestamp"].isin(test_times)]
        
        features = [c for c in train_df.columns if c not in ["timestamp", "asset"] and not c.startswith("ret_") and not c.startswith("fwd_ret") and not c.startswith("prob_tail")]
        features = [c for c in features if train_df[c].dtype in [float, int]]
        
        X_train = train_df[features]
        X_test = test_df[features]
        
        horizons_data = {}
        for h in cfg["horizons"]:
            horizons_data[h["label"]] = train_df
            
        log.info(f"Training up to {train_times[-1].date()} | Testing {test_times[0].date()} to {test_times[-1].date()}")
        model.fit(X_train, horizons_data)
        
        preds = model.predict(X_test)
        
        # Package results
        for h in cfg["horizons"]:
            h_str = h["label"]
            h_bars = h["bars"]
            
            res_df = test_df[["timestamp", "asset", f"fwd_ret_{h_bars}"]].copy()
            res_df = res_df.rename(columns={f"fwd_ret_{h_bars}": "realized_gross_bps"})
            res_df["realized_gross_bps"] *= 10000
            
            if h_str in preds:
                h_p = preds[h_str]
                res_df["pred_gross_bps"] = h_p["pred_gross_bps"] if "pred_gross_bps" in h_p else 0
                res_df["pred_linear_gross_bps"] = h_p["pred_linear_gross_bps"] if "pred_linear_gross_bps" in h_p else 0
                
            res_df["cost_bps"] = cost_bps
            # We predict long/short sign purely off pred_gross_bps
            res_df["position"] = 1
            res_df.loc[res_df["pred_gross_bps"] < 0, "position"] = -1
            
            # Simulated realized net
            res_df["realized_net_bps"] = res_df["realized_gross_bps"] * res_df["position"] - res_df["cost_bps"]
            
            res_df["baseline_position"] = 1
            res_df.loc[res_df["pred_linear_gross_bps"] < 0, "baseline_position"] = -1
            res_df["realized_baseline_net_bps"] = res_df["realized_gross_bps"] * res_df["baseline_position"] - res_df["cost_bps"]
            
            res_df["horizon"] = h_str
            res_df["fold"] = start_idx
            all_results.append(res_df)
            
    final_results = pd.concat(all_results, ignore_index=True)
    
    # 4. Reporting
    report_path = generate_report(final_results, cfg)
    log.info(f"exp025 pipeline complete. Gate report at {report_path}")

if __name__ == "__main__":
    run_experiment(sys.argv[1] if len(sys.argv) > 1 else "configs/experiments/crypto_exp025.yaml")
