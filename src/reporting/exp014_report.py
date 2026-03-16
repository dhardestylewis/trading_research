"""Report generation for exp014 microstructure feasibility."""
from pathlib import Path
import pandas as pd
from typing import Dict
from src.utils.io import ensure_dir

def generate_report(results: Dict[str, pd.DataFrame], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_lines = [
        "# Exp014 — Microstructure Feasibility Map",
        "",
        "## Summary",
        "This experiment evaluates whether any execution style is viable by modeling fill probability and post-fill markouts directly using historical tick data.",
        ""
    ]
    
    for symbol, df in results.items():
        report_lines.append(f"### Asset: {symbol}")
        report_lines.append(df.to_markdown(index=False))
        report_lines.append("")
        
        # Save CSV for further analysis
        df.to_csv(output_dir / f"{symbol.replace('/', '_')}_markouts.csv", index=False)
        
    report_path = output_dir / "exp014_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
        
    return report_path
