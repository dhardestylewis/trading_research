"""Report generator for Exp018 — RV Rule Audit."""
from pathlib import Path
import pandas as pd
from src.utils.io import ensure_dir
from src.utils.logging import get_logger

log = get_logger("exp018_report")


def generate_report(dedup_result: dict,
                    horizon_result: dict,
                    tendency_result: dict,
                    gate_comparison: dict,
                    regression_deduped: pd.DataFrame,
                    final_verdict: str,
                    output_dir: Path) -> Path:
    """Generate the exp018 markdown report.

    Sections:
    1. Deduplication results
    2. Horizon sensitivity audit
    3. Central tendency audit
    4. Regression rule comparison
    5. Corrected pair-level gates
    6. Final verdict
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    # ── Header ──────────────────────────────────────────────────
    lines.extend([
        "# Exp018 — RV Rule Audit (Diagnostic Salvage)",
        "",
        "## Objective",
        "",
        "Determine whether exp017's negative results are caused by fixable",
        "artifacts (trade overlap, bad gates, crude direction rules) or by",
        "fundamentally untradeable RV opportunity.",
        "",
        "---",
        "",
    ])

    # ── 1. Deduplication ────────────────────────────────────────
    lines.extend([
        "## 1. Trade Deduplication",
        "",
        f"| Stage | Count |",
        f"|:------|------:|",
        f"| Original exp017 trades | {dedup_result['original_count']:,} |",
        f"| After cross-horizon collapse | {dedup_result['after_horizon_collapse']:,} |",
        f"| After temporal dedup (12h gap) | {dedup_result['after_temporal_dedup']:,} |",
        "",
    ])

    reduction_pct = (1 - dedup_result["after_temporal_dedup"] /
                     dedup_result["original_count"]) * 100
    lines.append(f"**Overall reduction: {reduction_pct:.1f}%**")
    lines.append("")

    # Horizon collapse detail
    hc_stats = dedup_result.get("horizon_collapse_stats", pd.DataFrame())
    if not hc_stats.empty:
        lines.extend([
            "### Cross-Horizon Collapse Detail",
            "",
            hc_stats.to_markdown(index=False),
            "",
        ])

    # Temporal dedup detail
    td_stats = dedup_result.get("temporal_dedup_stats", pd.DataFrame())
    if not td_stats.empty:
        lines.extend([
            "### Temporal Dedup Detail",
            "",
            td_stats.to_markdown(index=False),
            "",
        ])

    lines.extend(["---", ""])

    # ── 2. Horizon Audit ────────────────────────────────────────
    lines.extend([
        "## 2. Horizon Sensitivity Audit",
        "",
    ])

    hs = horizon_result.get("summary", {})
    if hs:
        lines.extend([
            f"- **{hs.get('pct_identical', '?')}%** of horizon pairs are "
            f"statistically identical (KS test p > 0.05)",
            f"- Exit breakdown: {hs.get('overall_pct_stop', '?')}% stop loss, "
            f"{hs.get('overall_pct_tp', '?')}% take profit, "
            f"{hs.get('overall_pct_horizon', '?')}% horizon expiry",
            "",
        ])

    # KS results
    ks_df = horizon_result.get("ks_results", pd.DataFrame())
    if not ks_df.empty:
        lines.extend([
            "### KS Test Results (horizon pairs)",
            "",
        ])
        # Show summary: how many identical per pair
        for pair in ks_df["pair"].unique():
            pair_ks = ks_df[ks_df["pair"] == pair]
            n_id = pair_ks["identical"].sum()
            n_tot = len(pair_ks)
            lines.append(f"- **{pair}**: {n_id}/{n_tot} horizon pairs identical")
        lines.append("")

    # Exit reason breakdown (compact — group by pair)
    breakdown_df = horizon_result.get("exit_breakdown", pd.DataFrame())
    if not breakdown_df.empty:
        lines.extend([
            "### Exit Reason Breakdown",
            "",
        ])
        # Aggregate to pair-level for readability
        pair_exit = breakdown_df.groupby("pair").agg({
            "pct_stop": "mean",
            "pct_tp": "mean",
            "pct_horizon": "mean",
        }).round(1).reset_index()
        lines.append(pair_exit.to_markdown(index=False))
        lines.append("")

    lines.extend(["---", ""])

    # ── 3. Central Tendency ─────────────────────────────────────
    lines.extend([
        "## 3. Central Tendency Audit",
        "",
    ])

    flags = tendency_result.get("flags", {})
    if flags:
        lines.extend([
            f"- Combos with median > 0 but mean < 0: "
            f"**{flags.get('combos_with_masked_mean', 0)}** / "
            f"{flags.get('total_combos', 0)}",
            f"- Combos with high skewness (> 2): "
            f"**{flags.get('combos_with_high_skew', 0)}**",
            f"- Pairs with positive median: "
            f"**{flags.get('pairs_with_positive_median', 0)}** / "
            f"{flags.get('total_pairs', 0)}",
            f"- Pairs with positive mean: "
            f"**{flags.get('pairs_with_positive_mean', 0)}** / "
            f"{flags.get('total_pairs', 0)}",
            "",
        ])

    # Pair-level summary (the decisive table)
    pair_stats = tendency_result.get("pair_stats", pd.DataFrame())
    if not pair_stats.empty:
        lines.extend([
            "### Pair-Level Central Tendency (Deduplicated Trades)",
            "",
            "> This is the decisive table.",
            "",
            pair_stats.to_markdown(index=False),
            "",
        ])

    # Combo-level detail
    combo_stats = tendency_result.get("combo_stats", pd.DataFrame())
    if not combo_stats.empty:
        lines.extend([
            "### Combo-Level Detail (Top 10 by Median)",
            "",
        ])
        display_cols = ["pair", "rule", "spread_type", "n_trades",
                        "median_net_bps", "mean_net_bps", "trimmed_mean_bps",
                        "skewness", "frac_positive",
                        "median_positive_mean_negative"]
        cols = [c for c in display_cols if c in combo_stats.columns]
        top10 = combo_stats.sort_values("median_net_bps", ascending=False).head(10)
        lines.append(top10[cols].to_markdown(index=False))
        lines.append("")

    lines.extend(["---", ""])

    # ── 4. Regression Rule Comparison ───────────────────────────
    lines.extend([
        "## 4. Regression Direction Rule vs Heuristic Rules",
        "",
    ])

    comparison = gate_comparison.get("comparison", {})
    pair_comp = comparison.get("pair_comparison", pd.DataFrame())

    if not pair_comp.empty:
        lines.extend([
            "### Pair-Level Comparison",
            "",
            pair_comp.to_markdown(index=False),
            "",
        ])

    # Regression pair stats if available
    if not regression_deduped.empty:
        lines.extend([
            "### Regression Rule Pair-Level Detail",
            "",
        ])
        for pair in regression_deduped["pair"].unique():
            pair_data = regression_deduped[regression_deduped["pair"] == pair]
            vals = pair_data["net_spread_bps"].values
            lines.append(
                f"- **{pair}**: n={len(vals)}, "
                f"median={float(pd.Series(vals).median()):.1f}, "
                f"mean={float(vals.mean()):.1f}, "
                f"frac_pos={float((vals > 0).mean()):.3f}"
            )
        lines.append("")

    lines.extend(["---", ""])

    # ── 5. Corrected Gates ──────────────────────────────────────
    lines.extend([
        "## 5. Corrected Pair-Level Gate Evaluation",
        "",
    ])

    for label, gates_key in [("Heuristic Rules (Deduplicated)", "heuristic_gates"),
                              ("Regression Rule (Deduplicated)", "regression_gates")]:
        gates = gate_comparison.get(gates_key, {})
        verdict = gates.get("verdict", "N/A")
        lines.extend([
            f"### {label}",
            "",
            f"**Verdict: {verdict}**",
            "",
        ])

        gate_detail = gates.get("gates", {})
        if gate_detail:
            gate_rows = []
            for gname, gval in gate_detail.items():
                gate_rows.append({
                    "Gate": gname,
                    "Value": gval["value"],
                    "Threshold": gval["threshold"],
                    "Status": "PASS" if gval["pass"] else "FAIL",
                })
            lines.append(pd.DataFrame(gate_rows).to_markdown(index=False))
            lines.append("")

        pair_detail = gates.get("pair_details", {})
        if pair_detail:
            detail_rows = []
            for pair, d in pair_detail.items():
                detail_rows.append({
                    "Pair": pair,
                    "N": d["n_trades"],
                    "Median": d["median_net_bps"],
                    "Mean": d["mean_net_bps"],
                    "Frac Pos": d["frac_positive"],
                    "Pass": "✓" if d["passes_all"] else "✗",
                })
            lines.append(pd.DataFrame(detail_rows).to_markdown(index=False))
            lines.append("")

    lines.extend(["---", ""])

    # ── 6. Final Verdict ────────────────────────────────────────
    lines.extend([
        "## 6. Final Verdict",
        "",
        f"**{final_verdict}**",
        "",
    ])

    # Write report
    report_path = output_dir / "exp018_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Save key tables
    if not pair_stats.empty:
        pair_stats.to_csv(output_dir / "pair_level_stats.csv", index=False)
    if not combo_stats.empty:
        combo_stats.to_csv(output_dir / "combo_stats.csv", index=False)
    if not pair_comp.empty:
        pair_comp.to_csv(output_dir / "rule_comparison.csv", index=False)

    log.info(f"Report written to {report_path}")
    return report_path
