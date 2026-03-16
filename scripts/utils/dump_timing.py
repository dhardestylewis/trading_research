import pandas as pd
from analyze_pnl_timing import combined

with open("pnl_breakdown.txt", "w") as f:
    f.write("--- Q ---\n")
    f.write(combined.groupby('quarter')['net_bps'].sum().to_string())
    f.write("\n\n--- M (2024) ---\n")
    m24 = combined[combined['timestamp'].dt.year == 2024]
    f.write(m24.groupby('year_month')['net_bps'].sum().to_string())
