"""Reporting tools for exp025."""
from __future__ import annotations
import pandas as pd
from pathlib import Path

from src.utils.io import ensure_dir
from src.utils.logging import get_logger

log = get_logger("exp025_report")

def _safe_median(s: pd.Series) -> float:
    return float(s.median()) if not s.empty else 0.0

def _trimmed_mean(s: pd.Series, trim: float = 0.10) -> float:
    if len(s) < 10:
        return float(s.mean()) if not s.empty else 0.0
    q_low = s.quantile(trim)
    q_high = s.quantile(1 - trim)
    return float(s[(s >= q_low) & (s <= q_high)].mean())

def check_gates(results: pd.DataFrame, gates_cfg: dict, cost_floor: float) -> dict[str, bool]:
    """Check exp025 gates."""
    status = {}
    
    # Gate 1: At least two asset × horizon cells with 100+ trades & positive median > fee floor
    cells = results.groupby(["asset", "horizon"]).agg(
        trades=("pred_gross_bps", "count"),
        median_gross=("realized_gross_bps", "median")
    )
    passed_g1 = cells[(cells["trades"] >= 100) & (cells["median_gross"] > cost_floor)]
    status["Gate 1 (100+ trades, gross > floor)"] = len(passed_g1) >= 2
    
    # Gate 2: Positive net expectancy in score buckets
    # Look at top decile score
    if "is_top_decile" in results.columns:
        top_dec_results = results[results["is_top_decile"]]
        net_med = _safe_median(top_dec_results["realized_net_bps"])
        net_trim = _trimmed_mean(top_dec_results["realized_net_bps"], 0.10)
        status["Gate 2 (Positive net in top decile)"] = net_med > 0 and net_trim > 0
    else:
        status["Gate 2 (Positive net in top decile)"] = False
        
    # Gate 3: ML beats baselines
    base_med = _safe_median(results["realized_baseline_net_bps"]) if "realized_baseline_net_bps" in results.columns else 0.0
    ml_med = _safe_median(results["realized_net_bps"])
    status["Gate 3 (ML beats baseline)"] = ml_med > base_med
    
    # Gate 4: Walk-forward stability
    if "fold" in results.columns:
        fold_meds = results.groupby("fold")["realized_net_bps"].median()
        pos_folds = (fold_meds > 0).sum()
        status["Gate 4 (Walk-forward stability)"] = pos_folds >= (len(fold_meds) * 0.5)
    else:
        status["Gate 4 (Walk-forward stability)"] = False
        
    return status

def generate_report(results: pd.DataFrame, cfg: dict):
    """Generate markdown report for exp025."""
    out_dir = Path(cfg.get("reporting", {}).get("output_dir", "reports/exp025"))
    ensure_dir(out_dir)
    
    cost_floor = cfg["execution"]["round_trip_bps"]
    
    # First Table MUST BE: Gross Move vs Estimated Cost vs Net Expectancy by Asset x Horizon x Score
    # We bucket 'pred_gross_bps' into quintiles
    results["score_bucket"] = pd.qcut(results["pred_gross_bps"], q=5, labels=["low", "q2", "q3", "q4", "high"], duplicates="drop")
    
    table1 = results.groupby(["asset", "horizon", "score_bucket"]).agg(
        Count=("pred_gross_bps", "count"),
        Median_Gross_BPS=("realized_gross_bps", "median"),
        Est_Cost_BPS=("cost_bps", "mean"),
        Median_Net_BPS=("realized_net_bps", "median")
    ).dropna().reset_index()
    
    # Add top decile boolean for gate 2
    q90 = results["pred_gross_bps"].quantile(0.9)
    results["is_top_decile"] = results["pred_gross_bps"] >= q90
    
    gates = check_gates(results, cfg["gates"], cost_floor)
    
    md = [
        "# exp025: Event-Conditioned Perp-State Alpha",
        "",
        "## Core Economics",
        "The requirement to clear execution costs (estimated flat fee).",
        "",
        table1.to_markdown(index=False),
        "",
        "## Go/No-Go Gates",
    ]
    
    for g, passed in gates.items():
        icon = "✅ PASS" if passed else "❌ FAIL"
        md.append(f"- {icon} | {g}")
        
    md.append("\n## Verdict")
    if all(gates.values()):
        md.append("**PROCEED**. The feature set isolates states with gross moves large enough to clear perp execution costs.")
    else:
        md.append("**KILL**. The predicted signals do not reliably overcome realistic perp taker costs.")
        
    report_path = out_dir / "summary.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    log.info(f"Report written to {report_path}")
    return report_path
