"""Generate reports for the exp015 Magnitude Atlas."""
from pathlib import Path
import pandas as pd
from typing import Dict
from src.utils.io import ensure_dir

def generate_report(metrics_by_event: Dict[str, dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert metrics dictionary to a DataFrame
    df = pd.DataFrame.from_dict(metrics_by_event, orient="index")
    # Clean up column ordering: base metrics first, then horizon stats
    if not df.empty:
        cols = list(df.columns)
        first_cols = ["Observations"]
        rest_cols = [c for c in cols if c not in first_cols]
        # Sort rest cols to group by horizon intuitively
        rest_cols = sorted(rest_cols)
        df = df[first_cols + rest_cols]
        df.index.name = "Event State"
        df = df.reset_index()
    
    report_lines = [
        "# Exp015 — Catalyst Magnitude Atlas",
        "",
        "## Summary",
        "This experiment evaluates intrinsic OHLCV events (Volatility Expansion, Range Breakouts, RSI Extremes)",
        "to determine if their conditional gross move distributions are large enough to survive standard execution friction.",
        "",
        "### Exceedance and Magnitude Table",
        "The table displays the maximum absolute excursion (in bps) in either direction within H hours following the event.",
        "Probabilities indicate the chance of the gross move exceeding X bps.",
        "",
        df.to_markdown(index=False) if not df.empty else "No events met criteria.",
        ""
    ]
    
    # Save CSV for further analysis
    if not df.empty:
        df.to_csv(output_dir / "event_magnitudes.csv", index=False)
        
    report_path = output_dir / "exp015_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
        
    return report_path
