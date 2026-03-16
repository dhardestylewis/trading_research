"""Report generator for Exp016 Instrument and Relative-Value Screening."""
from pathlib import Path
import pandas as pd
from typing import Dict
from src.utils.io import ensure_dir

def generate_report(rv_results: pd.DataFrame, funding_stats: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_lines = [
        "# Exp016 — Instrument / Relative-Value Screening",
        "",
        "## Summary",
        "This experiment evaluates whether the monetizable edge identified in purely directional OHLCV events",
        "can be captured via Cross-Asset Relative Value pairs or structured Yield/Basis trades, isolating the alpha from directional market beta and queue toxicity.",
        "",
        "### Relative-Value Event Magnitudes",
        "Measuring the maximum directional excursion of synthetic pair ratios (Asset A / Asset B)",
        "following volatility expansions on the ratio itself.",
        ""
    ]
    
    if not rv_results.empty:
        report_lines.append(rv_results.to_markdown(index=False))
        rv_results.to_csv(output_dir / "rv_magnitudes.csv", index=False)
    else:
        report_lines.append("No RV events found that pass criteria.")
        
    report_lines.extend([
        "",
        "### Funding Basis Edge Opportunities",
        "Measuring the percentage of historical hourly blocks where trailing annualized perp funding yield strictly exceeded the target threshold.",
        ""
    ])
    
    if funding_stats:
        fund_df = pd.DataFrame.from_dict(funding_stats, orient="index")
        fund_df.index.name = "Asset"
        fund_df = fund_df.reset_index()
        report_lines.append(fund_df.to_markdown(index=False))
        fund_df.to_csv(output_dir / "funding_stats.csv", index=False)
    else:
        report_lines.append("No funding data found.")
        
    report_path = output_dir / "exp016_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
        
    return report_path
