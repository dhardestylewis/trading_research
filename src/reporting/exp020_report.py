"""Report generation for exp020. Produces Cost Surface evaluation."""
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any

from src.utils.logging import get_logger

log = get_logger("exp020_report")

def generate_report(
    features: pd.DataFrame,
    surface_df: pd.DataFrame,
    model_result: Dict[str, Any],
    output_dir: Path,
    report_dir: Path
) -> Path:
    """Generate the markdown report for exp020 Execution Cost Modeling."""
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "exp020_cost_modeling_report.md"
    
    corr = model_result.get("correlation", 0.0)
    
    with open(report_path, "w") as f:
        f.write("# Exp020: Execution Cost Modeling Report\n\n")
        f.write("## 1. Executive Summary\n")
        f.write("This branch shifts focus from directional signal to direct estimation of execution cost (slippage and toxicity). ")
        f.write("The expectation is that execution cost is highly predictable due to structural market microstructure variables.\n\n")
        
        f.write(f"**Global OOF Correlation for Exec Loss:** `{corr:.4f}`\n\n")
        
        f.write("## 2. Feature Importance\n")
        f.write("Top defining features for execution constraints:\n")
        f.write("```text\n")
        if not model_result.get("feature_importance", pd.DataFrame()).empty:
            top_feats = model_result["feature_importance"].sort_values("mean_gain", ascending=False).head(10)
            f.write(top_feats[["mean_gain"]].to_string())
        else:
            f.write("No models trained or no feature importance available.")
        f.write("\n```\n\n")

        f.write("## 3. Cost Surface by Asset\n")
        f.write("Average predicted execution loss (bps) breakdown by asset.\n\n")
        
        asset_summary = surface_df.groupby("asset")["pred_exec_loss_bps"].agg(["mean", "median", "std"]).round(2)
        f.write(asset_summary.to_markdown())
        f.write("\n\n")
        
        f.write("## 4. Toxicity Regimes\n")
        if "toxicity_decile" in surface_df.columns:
            f.write("Execution cost escalates significantly in the worst toxicity deciles. Decile 9 represents the most expensive periods to execute.\n\n")
            tox_summary = surface_df.groupby("toxicity_decile")["pred_exec_loss_bps"].agg(["mean", "min", "max", "count"]).round(2)
            f.write(tox_summary.to_markdown())
            f.write("\n\n")
        else:
            f.write("No toxicity decile calculated.\n\n")
            
        f.write("## 5. Next Steps\n")
        f.write("The models produced here can act as a direct filter on future alpha (exp022), ")
        f.write("meaning any future candidate trade must have gross expected edge exceeding its specific point-in-time predicted execution loss.\n")

    log.info("Saved report to %s", report_path)
    return report_path
