import logging
import pandas as pd
from pathlib import Path

log = logging.getLogger("exp022_report")
logging.basicConfig(level=logging.INFO)

def generate_report(metrics: dict, output_dir: Path):
    """
    Generate the Exp022 Markdown report summarizing OOS performance of 
    short-horizon flow features on gross markouts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "exp022_short_horizon_flow_report.md"
    
    # metrics format: {"target": {"spearman": val, "sign_acc": val, "feature_importance": df}}
    
    with open(report_path, "w") as f:
        f.write("# Exp022: Short-Horizon Flow Pilot Results\n\n")
        f.write("This report summarizes the Out-Of-Sample (OOS) performance of "
                "trade-flow features on very short-term gross markouts.\n\n")
        
        f.write("## 1. Global OOS Performance\n\n")
        f.write("| Target | Spearman Corr | Sign Accuracy |\n")
        f.write("|---|---|---|\n")
        for target, res in metrics.items():
            f.write(f"| `{target}` | {res['spearman']:.4f} | {res['sign_acc']:.2%} |\n")
        f.write("\n")
        
        f.write("## 2. Feature Importance Analytics\n\n")
        for target, res in metrics.items():
            f.write(f"### Target: `{target}`\n\n")
            fi_df = res['feature_importance']
            if not fi_df.empty:
                f.write(fi_df.head(10).to_markdown(index=False))
            else:
                f.write("No feature importance available.\n")
            f.write("\n\n")
            
        f.write("## 3. Conclusion & Next Steps\n\n")
        all_corrs = [res['spearman'] for res in metrics.values()]
        if any(c > 0.02 for c in all_corrs):
            f.write("**PILOT SUCCESSFUL:** At least one short-horizon target exhibits meaningful OOS gross predictability from flow features alone. "
                    "Proceed with multi-venue backfills and enhanced flow model design.\n")
        else:
            f.write("**PILOT NEGATIVE:** Short-horizon gross moves are not consistently predictable using this limited feature set. "
                    "Review features or wait for enriched L2/DEX data.\n")
                    
    log.info(f"Generated Exp022 report at {report_path}")
    return report_path
