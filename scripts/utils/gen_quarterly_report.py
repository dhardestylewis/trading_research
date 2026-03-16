"""Quick script to generate a full quarter-by-quarter PnL breakdown from dynamic_windows_results.csv"""
import pandas as pd
from pathlib import Path

df = pd.read_csv("reports/spike_tsfm/dynamic_windows_results.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
df['quarter'] = df['timestamp'].dt.to_period('Q').astype(str)

lines = ["# Dynamic Context — Full Quarterly Breakdown", ""]

for asset in sorted(df['asset'].unique()):
    lines.append(f"## {asset}")
    adf = df[df['asset'] == asset]
    quarters = sorted(adf['quarter'].unique())
    strategies = sorted(adf['strategy'].unique())
    
    # Header
    header = "| Quarter |"
    sep = "|---|"
    for s in strategies:
        header += f" {s} (trades/pnl) |"
        sep += "---|"
    lines.extend([header, sep])
    
    for q in quarters:
        row = f"| {q} |"
        for s in strategies:
            cell = adf[(adf['quarter'] == q) & (adf['strategy'] == s) & (adf['fires'])]
            if len(cell) > 0:
                row += f" {len(cell)} / {cell['net_bps'].sum():+.0f} |"
            else:
                row += " 0 / — |"
        lines.append(row)
    
    # Totals row
    row = "| **TOTAL** |"
    for s in strategies:
        cell = adf[(adf['strategy'] == s) & (adf['fires'])]
        if len(cell) > 0:
            row += f" **{len(cell)} / {cell['net_bps'].sum():+.0f}** |"
        else:
            row += " **0 / —** |"
    lines.append(row)
    lines.append("")

rpt = Path("reports/spike_tsfm/dynamic_context_quarterly.md")
rpt.write_text("\n".join(lines), encoding='utf-8')
print(f"Saved to {rpt}")
print("\n".join(lines))
