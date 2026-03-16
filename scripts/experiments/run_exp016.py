"""Run exp016: Instrument and Relative-Value Screening."""
import yaml
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from src.research.exp015.event_extraction import add_atr
from src.research.exp016.relative_value_scanner import build_ratio_panel, evaluate_rv_excursions
from src.research.exp016.funding_basis_analyzer import evaluate_funding_opportunities
from src.reporting.exp016_report import generate_report
from src.utils.logging import get_logger

log = get_logger("run_exp016")

def extract_rv_events(panel: pd.DataFrame, events_cfg: dict) -> pd.DataFrame:
    """Extract events applied to the ratio panel."""
    def process_asset(g: pd.DataFrame) -> pd.DataFrame:
        df = g.copy()
        cfg = events_cfg["vol_expansion"]
        atr = add_atr(df, cfg["atr_window"])
        baseline = atr.rolling(cfg["baseline_window"]).mean()
        df["event_vol_expansion"] = atr > (cfg["multiplier"] * baseline)
        return df
        
    return panel.groupby("asset", group_keys=False).apply(process_asset)

def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
        
    data_cfg = cfg["data"]
    events_cfg = cfg["events"]
    mag_cfg = cfg["magnitude"]
    gates_cfg = cfg["gates"]
    
    panel_path = Path(data_cfg["panel_path"])
    out_dir = Path(data_cfg["processed_dir"])
    fund_raw_dir = Path(data_cfg["funding_raw_dir"])
    
    log.info(f"Loading panel from {panel_path}...")
    panel = pd.read_parquet(panel_path)
    
    # 1. RV Pairs
    ratio_panel = build_ratio_panel(panel, data_cfg["rv_pairs"])
    df_rv_events = extract_rv_events(ratio_panel, events_cfg)
    
    df_exc = df_rv_events.groupby("asset", group_keys=False).apply(
        lambda g: evaluate_rv_excursions(g, mag_cfg["horizons_hours"])
    )
    
    rows = []
    for asset, group in df_exc.groupby("asset"):
        event_df = group[group["event_vol_expansion"] == True]
        obs = len(event_df)
        
        if obs < gates_cfg["min_observations"]:
            continue
            
        row_dict = {"RV Pair": asset, "Observations": obs}
        
        for h in mag_cfg["horizons_hours"]:
            col_up = f"mfe_up_{h}h_bps"
            col_down = f"mfe_down_{h}h_bps"
            
            max_abs_move = np.maximum(event_df[col_up].values, event_df[col_down].abs().values)
            max_abs_move = max_abs_move[~np.isnan(max_abs_move)]
            
            if len(max_abs_move) > 0:
                row_dict[f"{h}h_median_bps"] = np.median(max_abs_move)
                row_dict[f"{h}h_prob_>100bps"] = np.mean(max_abs_move > 100)
                
        rows.append(row_dict)
        
    rv_results = pd.DataFrame(rows)
    
    # 2. Funding Basis
    funding_stats = {}
    target = gates_cfg["funding_target_annualized_percent"]
    
    for parquet_path in fund_raw_dir.glob("*_funding.parquet"):
        symbol = parquet_path.stem.split("_")[0]
        df_fund = pd.read_parquet(parquet_path)
        eval_df = evaluate_funding_opportunities(df_fund, target)
        
        if not eval_df.empty:
            total_obs = len(eval_df)
            valid_obs = eval_df["passes_target"].sum()
            avg_yield = eval_df["annualized_yield_pct"].mean()
            
            funding_stats[symbol] = {
                "Total 8h Intervals": total_obs,
                f"Intervals > {target}% Ann.": valid_obs,
                "% Time in Extreme Basis": (valid_obs / total_obs) * 100 if total_obs > 0 else 0,
                "Mean Ann. Yield %": avg_yield
            }
            
    report_path = generate_report(rv_results, funding_stats, out_dir)
    log.info(f"Exp016 Report generated: {report_path}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else 'configs/experiments/crypto_1h_exp016.yaml')
