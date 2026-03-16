"""Report generator for Exp017 — Direction-Locked RV Execution Study."""
from pathlib import Path
import pandas as pd
from src.utils.io import ensure_dir
from src.utils.logging import get_logger

log = get_logger("exp017_report")


def generate_report(trades_df: pd.DataFrame,
                    summary_df: pd.DataFrame,
                    comparison_df: pd.DataFrame,
                    kill_gates: dict,
                    output_dir: Path) -> Path:
    """Generate the exp017 markdown report.

    Produces:
    1. Kill-gate summary
    2. Side-by-side excursion vs realized PnL table
    3. Summary statistics by (pair × rule × spread_type)
    4. Sample trade log
    5. Best combinations and conclusion
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Exp017 — Direction-Locked RV Execution Study",
        "",
        "## Objective",
        "",
        "Test whether exp016's large RV excursions survive when converted into a",
        "**fully specified tradable rule** with direction locked ex ante, two-leg",
        "execution costs, and realistic holding horizons.",
        "",
        "---",
        "",
    ]

    # 1. Kill Gates
    lines.extend([
        "## Kill-Gate Evaluation",
        "",
    ])

    overall = kill_gates.get("overall_pass", False)
    verdict = "**PASS**" if overall else "**FAIL**"
    lines.append(f"**Overall Verdict: {verdict}**")
    lines.append("")

    gate_rows = []
    for key, val in kill_gates.items():
        if isinstance(val, dict) and "pass" in val:
            status = "PASS" if val["pass"] else "FAIL"
            gate_rows.append({
                "Gate": key,
                "Value": val["value"],
                "Threshold": val["threshold"],
                "Status": status,
            })

    if gate_rows:
        lines.append(pd.DataFrame(gate_rows).to_markdown(index=False))
    lines.append("")

    # 2. Side-by-side comparison
    lines.extend([
        "---",
        "",
        "## Excursion Opportunity vs Realized Rule-Locked PnL",
        "",
        "> This is the critical table. The left column shows what exp016 measured",
        "> (the best possible move in the optimal direction). The right column shows",
        "> what a direction-locked rule with full execution costs actually delivers.",
        "",
    ])

    if not comparison_df.empty:
        lines.append(comparison_df.to_markdown(index=False))
        comparison_df.to_csv(output_dir / "excursion_vs_realized.csv", index=False)
    else:
        lines.append("No comparison data available.")
    lines.append("")

    # 3. Summary statistics
    lines.extend([
        "---",
        "",
        "## Summary Statistics by (Pair × Rule × Spread × Horizon)",
        "",
    ])

    if not summary_df.empty:
        display_cols = ["pair", "rule", "spread_type", "horizon_h",
                        "trade_count", "hit_rate", "median_net_bps",
                        "mean_net_bps", "mean_gross_bps", "weekly_trade_count",
                        "max_drawdown_bps", "sharpe_of_spread"]
        cols = [c for c in display_cols if c in summary_df.columns]
        lines.append(summary_df[cols].to_markdown(index=False))
        summary_df.to_csv(output_dir / "summary_stats.csv", index=False)
    else:
        lines.append("No summary data available.")
    lines.append("")

    # 4. Best combinations
    lines.extend([
        "---",
        "",
        "## Best Performing Combinations",
        "",
    ])

    if not summary_df.empty:
        top5 = summary_df.head(5)
        lines.append(top5[["pair", "rule", "spread_type", "horizon_h",
                           "median_net_bps", "hit_rate", "trade_count"
                           ]].to_markdown(index=False))
    lines.append("")

    # 5. Sample trade log
    lines.extend([
        "---",
        "",
        "## Sample Trade Log (first 30 trades)",
        "",
    ])

    if not trades_df.empty:
        sample_cols = ["entry_time", "pair", "rule", "spread_type",
                       "direction_label", "horizon_h",
                       "gross_spread_bps", "net_spread_bps",
                       "total_friction_bps", "holding_bars", "exit_reason"]
        cols = [c for c in sample_cols if c in trades_df.columns]
        lines.append(trades_df.head(30)[cols].to_markdown(index=False))
        trades_df.to_csv(output_dir / "trade_log.csv", index=False)
    else:
        lines.append("No trades generated.")
    lines.append("")

    # 6. Conclusion
    lines.extend([
        "---",
        "",
        "## Conclusion",
        "",
    ])

    if overall:
        lines.extend([
            "Kill gates **passed**. At least 2 pairs show ≥100 trades with",
            "median net spread return > 40 bps after conservative two-leg friction.",
            "",
            "**Interpretation**: Event-conditioned market-neutral spread opportunity",
            "is confirmed as monetizable under direction-locked rules. Proceed to",
            "exp018 (instrument choice optimization).",
        ])
    else:
        lines.extend([
            "Kill gates **failed**. The large RV excursions from exp016 do not",
            "survive when converted into direction-locked, two-leg net PnL.",
            "",
            "**Interpretation**: The excursion statistics were measuring opportunity,",
            "not strategy economics. The gap between max excursion and realized PnL",
            "is too large to bridge with simple mechanical rules.",
            "",
            "See the side-by-side comparison table above for the capture ratio.",
        ])
    lines.append("")

    report_path = output_dir / "exp017_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info(f"Report written to {report_path}")
    return report_path
