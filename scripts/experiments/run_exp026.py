"""Runner for exp026: Rich-State Catalyst Alpha with Strong-Model Challengers."""
from __future__ import annotations
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys path
sys.path.append(str(Path(__file__).parent))

from src.data.build_rich_perp_state_features import build_rich_perp_state_features
from src.models.tree_baselines_exp026 import TreeBaselinesExp026
from src.models.foundation_challenger_exp026 import FoundationChallengerExp026
from tqdm import tqdm
from src.eval.economic_scorecard_exp026 import EconomicScorecardExp026
from src.reporting.exp026_report import generate_report
from src.utils.io import ensure_dir, load_parquet
from src.utils.logging import get_logger

log = get_logger("run_exp026")

def load_and_merge_data(cfg: dict) -> pd.DataFrame:
    """Load base panel and merge perp metrics."""
    panel_path = Path(cfg["data"]["panel_path"])
    raw_metrics_dir = Path("data/raw/crypto_perp_metrics")
    
    if not panel_path.exists():
        raise FileNotFoundError(f"Base panel not found at {panel_path}")
        
    log.info(f"Loading base panel from {panel_path}")
    panel = load_parquet(panel_path)
    
    if "timestamp" not in panel.columns:
        raise ValueError("Panel missing timestamp")
        
    time_diffs = panel["timestamp"].diff().dropna()
    mode_diff = time_diffs.mode()[0]
    
    merged_assets = []
    
    for symbol in cfg["data"]["asset_universe"]:
        label = symbol.replace("/", "-")
        asset_panel = panel[panel["asset"] == label].copy()
        if asset_panel.empty:
            continue
            
        # Load perp metrics
        oi_path = raw_metrics_dir / f"{label.replace('-USD', '')}_open_interest_15m.csv"
        mp_path = raw_metrics_dir / f"{label.replace('-USD', '')}_mark_price_15m.csv"
        funding_path = Path("data/raw/crypto_funding") / f"{label.replace('-USD', '')}_funding.csv"
        
        # OI
        if oi_path.exists():
            oi_df = pd.read_csv(oi_path)
            oi_df["timestamp"] = pd.to_datetime(oi_df["timestamp"], utc=True)
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
    
    # 2. Extract horizons
    horizons = [h["bars"] for h in cfg["horizons"]]
    seq_len = 6 
    
    # 3. Build Features
    df = build_rich_perp_state_features(panel, horizons, sequence_len=seq_len)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 4. Walk-Forward Setup
    wf_cfg = cfg["walk_forward"]
    # We will just use the first step for minimum viable reproduction
    # train_windows_days: [30, 60, 90]
    # For simplicity of exact replication logic, we'll convert days to bars.
    # We assume 1h bars based on the experiment spec.
    bars_per_day = 24
    
    min_train_bars = wf_cfg["train_windows_days"][0] * bars_per_day
    step_bars = wf_cfg["refresh_cadence_days"][-1] * bars_per_day
    
    times = sorted(df["timestamp"].unique())
    n_times = len(times)
    
    cost_bps = cfg["execution"]["round_trip_bps"]
    
    seq_cols = []
    if "ret_1" in df.columns: seq_cols.append("ret_1")
    if "oi_change_1" in df.columns: seq_cols.append("oi_change_1")
    if "funding_level" in df.columns: seq_cols.append("funding_level")
    if "mark_premium" in df.columns: seq_cols.append("mark_premium")
    if "vol_surge" in df.columns: seq_cols.append("vol_surge")
    if "rv_6" in df.columns: seq_cols.append("rv_6")
    
    # 5. Iterating Folds
    all_results = []
    
    for start_idx in tqdm(range(min_train_bars, n_times, step_bars), desc="Walk-forward Folds", leave=True):
        end_idx = min(start_idx + step_bars, n_times)
        if end_idx - start_idx < 10:
            break
            
        train_start = max(0, start_idx - (wf_cfg["train_windows_days"][-1] * bars_per_day))
        train_end = start_idx
        
        train_times = times[train_start:train_end]
        test_times = times[start_idx:end_idx]
        
        train_df = df[df["timestamp"].isin(train_times)]
        test_df = df[df["timestamp"].isin(test_times)]
        
        features = [c for c in train_df.columns if c not in ["timestamp", "asset"] and not c.startswith("ret_") and not c.startswith("fwd_ret") and not c.startswith("prob_tail") and not c.startswith("gross_move_bps")]
        features = [c for c in features if train_df[c].dtype in [float, int, np.float32, np.float64]]
        
        X_train = train_df[features]
        X_test = test_df[features]
        
        log.info(f"Training fold: {train_times[-1].date()} | Testing: {test_times[0].date()} to {test_times[-1].date()}")
        
        # Baselines
        baselines = TreeBaselinesExp026(horizons)
        baselines.fit(X_train, train_df)
        b_preds = baselines.predict(X_test)
        
        # Foundation Challenger
        challenger = FoundationChallengerExp026(horizons)
        challenger.fit(X_train, train_df)
        c_preds = challenger.predict(X_test)
        
        # Merge predictions into test_df
        fold_results = test_df[["timestamp", "asset"] + [f"fwd_ret_{h}" for h in horizons]].copy()
        
        for h in horizons:
            if h in b_preds:
                for col in b_preds[h].columns:
                    fold_results[f"{col}_{h}"] = b_preds[h][col]
            if h in c_preds:
                for col in c_preds[h].columns:
                    fold_results[f"{col}_{h}"] = c_preds[h][col]
                    
        all_results.append(fold_results)
        
    df_merged = pd.concat(all_results, ignore_index=True)
    
    # 6. Evaluation
    scorecard = EconomicScorecardExp026(cost_bps=cost_bps)
    model_names = cfg["models"]["baselines"] + cfg["models"]["challengers"]
    
    table1 = scorecard.evaluate_table1(df_merged, horizons, model_names)
    gates_result = scorecard.verify_gates(table1, cfg["gates"])
    
    # 7. Reporting
    report_path = generate_report(table1, gates_result, cfg)
    log.info(f"exp026 pipeline complete. Setup report generated at {report_path}")

if __name__ == "__main__":
    run_experiment(sys.argv[1] if len(sys.argv) > 1 else "configs/experiments/crypto_exp026.yaml")
