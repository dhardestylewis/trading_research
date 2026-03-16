"""Report generator for exp026."""
from __future__ import annotations
import pandas as pd
from pathlib import Path
from src.utils.io import ensure_dir

def generate_report(df_table1: pd.DataFrame, gates_result: dict, cfg: dict) -> Path:
    out_dir = Path(cfg["reporting"]["output_dir"])
    ensure_dir(out_dir)
    
    report_path = out_dir / "exp026_report.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# exp026: {cfg['experiment_name']}\n\n")
        f.write("## 1. Objective and Setup\n")
        f.write(f"{cfg['objective']}\n\n")
        
        f.write("## 2. Hard Gates Verdict\n")
        f.write(f"- Gross plausibility (>25 bps, >100 trades): {'PASS' if gates_result['gate1_gross_plausibility'] else 'FAIL'}\n")
        f.write(f"- Net expectancy (>0 med, >0 trim mean): {'PASS' if gates_result['gate2_net_expectancy'] else 'FAIL'}\n")
        f.write(f"- Challenger value (beats baseline): {'PASS' if gates_result['gate3_challenger_value'] else 'FAIL'}\n")
        f.write(f"\n**OVERALL BRANCH VERDICT: {'PASS' if gates_result['overall_pass'] else 'KILL'}**\n\n")
        
        f.write("## 3. Table 1: Gross vs Cost vs Net by Asset × Horizon × Score Bucket\n")
        # Sort and write table 1
        # Priority to top buckets, positive net bps
        df_table1_sorted = df_table1.sort_values(
            by=["horizon", "model", "asset", "score_bucket"], 
            ascending=[True, True, True, False]
        )
        
        # Round numeric columns
        numeric_cols = ["predicted_gross_bps", "realized_gross_bps_mean", "estimated_cost_bps", "realized_net_bps_mean", "median_net_bps", "trimmed_mean_net_bps"]
        for col in numeric_cols:
            if col in df_table1_sorted.columns:
                df_table1_sorted[col] = df_table1_sorted[col].round(2)
                
        # Pick top buckets (e.g. bucket 9) or just write all if not too huge.
        # We'll just write the full Markdown table
        f.write(df_table1_sorted.to_markdown(index=False))
        f.write("\n\n")
        
    return report_path
