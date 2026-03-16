"""Run exp015: Catalyst Magnitude Atlas."""
import yaml
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from src.research.exp015.event_extraction import extract_events
from src.research.exp015.magnitude_distributions import compute_excursions
from src.reporting.exp015_report import generate_report
from src.utils.logging import get_logger

log = get_logger("run_exp015")

def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
        
    data_cfg = cfg["data"]
    events_cfg = cfg["events"]
    mag_cfg = cfg["magnitude"]
    gates_cfg = cfg["gates"]
    
    panel_path = Path(data_cfg["panel_path"])
    out_dir = Path(data_cfg["processed_dir"])
    
    if not panel_path.exists():
        log.error(f"Panel data not found at {panel_path}. Ensure exp008 or exp009 has run.")
        sys.exit(1)
        
    log.info(f"Loading panel from {panel_path}...")
    panel = pd.read_parquet(panel_path)
    log.info(f"Loaded {len(panel)} rows.")
    
    # Extract events
    df_with_events = extract_events(panel, events_cfg)
    
    # Identify the boolean event columns we just created
    event_cols = [c for c in df_with_events.columns if c.startswith("event_")]
    log.info(f"Found event states: {event_cols}")
    
    # Compute excursions for all bars
    # Using group_keys=False if grouped by asset to avoid multi-index bloat
    if "asset" in df_with_events.columns:
        df_exc = df_with_events.groupby("asset", group_keys=False).apply(lambda g: compute_excursions(g, mag_cfg["horizons_hours"]))
    else:
        df_exc = compute_excursions(df_with_events, mag_cfg["horizons_hours"])
        
    # Analyze by state
    tables = {}
    for event in event_cols:
        event_df = df_exc[df_exc[event] == True]
        obs = len(event_df)
        
        if obs < gates_cfg["min_observations"]:
            log.warning(f"Event {event} failed observations gate ({obs} < {gates_cfg['min_observations']})")
            continue
            
        metrics = {"Observations": obs}
        
        for h in mag_cfg["horizons_hours"]:
            col_up = f"mfe_up_{h}h_bps"
            col_down = f"mfe_down_{h}h_bps"
            
            # The maximum absolute excursion in EITHER direction
            max_abs_move = np.maximum(event_df[col_up].values, event_df[col_down].abs().values)
            max_abs_move = max_abs_move[~np.isnan(max_abs_move)]
            
            if len(max_abs_move) == 0:
                continue
                
            metrics[f"{h}h_median_bps"] = np.median(max_abs_move)
            metrics[f"{h}h_mean_bps"] = np.mean(max_abs_move)
            
            for t in mag_cfg["exceedance_bps_thresholds"]:
                exceedance_prob = np.mean(max_abs_move > t)
                metrics[f"{h}h_prob_>{t}bps"] = exceedance_prob
                
        # Check median friction gate on the longest horizon (e.g. 72h)
        longest_h = max(mag_cfg["horizons_hours"])
        if metrics.get(f"{longest_h}h_median_bps", 0) >= gates_cfg["min_median_move_bps"]:
            log.info(f"Event {event} passes median magnitude gate.")
        else:
            log.warning(f"Event {event} fails median magnitude gate ({metrics.get(f'{longest_h}h_median_bps', 0):.1f} < {gates_cfg['min_median_move_bps']})")
            
        tables[event] = metrics
        
    report_path = generate_report(tables, out_dir)
    log.info(f"Report saved to {report_path}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else 'configs/experiments/crypto_1h_exp015.yaml')
