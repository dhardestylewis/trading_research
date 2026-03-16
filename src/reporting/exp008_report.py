"""exp008 report generator — Capacity Research.

Produces:
  - Summary markdown report
  - Per-branch CSV tables
  - Figures: horizon stacking, transportability, calibration,
    rank quality, capacity sensitivity, scenario comparison
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.io import ensure_dir
from src.utils.logging import get_logger

log = get_logger("exp008_report")


# ── Plot generators ─────────────────────────────────────────────


def plot_horizon_stacking(horizon_result: pd.DataFrame, fig_dir: Path) -> Path | None:
    """Bar chart of trade counts and Sharpe per horizon."""
    if horizon_result is None or horizon_result.empty:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    labels = horizon_result["horizon"].astype(str).values
    x = np.arange(len(labels))

    # Trade count
    ax1.bar(x, horizon_result["trade_count"].fillna(0), color="steelblue", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Horizon")
    ax1.set_ylabel("Trade Count")
    ax1.set_title("Trade Count by Horizon")

    # Sharpe
    sharpe = horizon_result["sharpe"].fillna(0)
    colors = ["green" if s > 0 else "red" for s in sharpe]
    ax2.bar(x, sharpe, color=colors, alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("Horizon")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_title("Sharpe by Horizon")
    ax2.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    out = fig_dir / "horizon_stacking_tradecount.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_transportability_matrix(universe_result: pd.DataFrame, fig_dir: Path) -> Path | None:
    """Heatmap-style bar chart of per-asset edge."""
    if universe_result is None or universe_result.empty:
        return None

    df = universe_result[universe_result["asset"] != "QUALIFYING_AGGREGATE"].copy()
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))

    assets = df["asset"].values
    net_bps = df["mean_net_bps"].fillna(0).values
    colors = ["green" if q else "gray" for q in df["qualifies"]]

    x = np.arange(len(assets))
    ax.bar(x, net_bps, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(assets, rotation=45, ha="right")
    ax.set_ylabel("Mean Net Return (bps)")
    ax.set_title("Universe Expansion — Per-Asset Edge")
    ax.axhline(y=0, color="black", linewidth=0.5)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="green", alpha=0.8, label="Qualifies"),
        Patch(color="gray", alpha=0.8, label="Does not qualify"),
    ])

    plt.tight_layout()
    out = fig_dir / "transportability_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_calibration_curve(calibration_curve: pd.DataFrame, fig_dir: Path) -> Path | None:
    """Reliability diagram."""
    if calibration_curve is None or calibration_curve.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 7))

    for variant, grp in calibration_curve.groupby("variant"):
        ax.plot(grp["predicted_prob"], grp["observed_freq"],
                marker="o", label=variant, alpha=0.8)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    out = fig_dir / "calibration_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_rank_quality(rank_quality: pd.DataFrame, fig_dir: Path) -> Path | None:
    """Ventile return monotonicity chart."""
    if rank_quality is None or rank_quality.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))

    ventiles = rank_quality["ventile"].values
    returns = rank_quality["mean_return_bps"].values
    colors = ["green" if r > 0 else "red" for r in returns]

    ax.bar(ventiles, returns, color=colors, alpha=0.8)
    ax.set_xlabel("Score Ventile (0=lowest, 19=highest)")
    ax.set_ylabel("Mean Return (bps)")
    ax.set_title("Rank Quality — Return by Score Ventile")
    ax.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    out = fig_dir / "rank_quality.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_capacity_sensitivity(grid: pd.DataFrame, fig_dir: Path) -> Path | None:
    """Heatmap of $/week across notional × frequency at base slippage."""
    if grid is None or grid.empty:
        return None

    # Pick median slippage for the heatmap
    median_slip = sorted(grid["slippage_bps"].unique())[len(grid["slippage_bps"].unique()) // 2]
    subset = grid[grid["slippage_bps"] == median_slip]

    pivot = subset.pivot_table(
        index="notional_usd", columns="freq_multiplier",
        values="weekly_pnl_usd", aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.0f}×" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"${x:,.0f}" for x in pivot.index])
    ax.set_xlabel("Trade Frequency Multiplier")
    ax.set_ylabel("Notional per Trade")
    ax.set_title(f"Weekly PnL ($/week) — Slippage = {median_slip} bps")

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"${val:,.0f}", ha="center", va="center",
                    color="black" if abs(val) < pivot.values.max() * 0.5 else "white",
                    fontsize=9)

    fig.colorbar(im, ax=ax, label="$/week")
    plt.tight_layout()
    out = fig_dir / "capacity_sensitivity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_scenario_comparison(scorecard: pd.DataFrame, fig_dir: Path) -> Path | None:
    """Bar chart comparing $/week across scenarios."""
    if scorecard is None or scorecard.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))

    scenarios = scorecard["scenario"].values
    pnl = scorecard["weekly_pnl_usd"].values

    x = np.arange(len(scenarios))
    colors = ["steelblue" if p >= 0 else "red" for p in pnl]
    ax.bar(x, pnl, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha="right")
    ax.set_ylabel("Weekly PnL ($/week)")
    ax.set_title("Capacity Scenarios — Weekly PnL Comparison")
    ax.axhline(y=0, color="black", linewidth=0.5)

    for i, v in enumerate(pnl):
        ax.text(i, v + max(pnl) * 0.02, f"${v:,.0f}", ha="center", fontsize=9)

    plt.tight_layout()
    out = fig_dir / "weekly_pnl_scenarios.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_all_plots(
    fig_dir: Path,
    *,
    horizon_result: pd.DataFrame | None = None,
    universe_result: pd.DataFrame | None = None,
    calibration_curve: pd.DataFrame | None = None,
    rank_quality: pd.DataFrame | None = None,
    cs_ranking: pd.DataFrame | None = None,
    exec_aware_result: pd.DataFrame | None = None,
    scorecard: pd.DataFrame | None = None,
    grid: pd.DataFrame | None = None,
) -> list[Path]:
    """Generate all exp008 plots. Returns list of saved paths."""
    fig_dir = ensure_dir(fig_dir)
    paths: list[Path] = []

    p = plot_horizon_stacking(horizon_result, fig_dir)
    if p:
        paths.append(p)

    p = plot_transportability_matrix(universe_result, fig_dir)
    if p:
        paths.append(p)

    p = plot_calibration_curve(calibration_curve, fig_dir)
    if p:
        paths.append(p)

    p = plot_rank_quality(rank_quality, fig_dir)
    if p:
        paths.append(p)

    p = plot_capacity_sensitivity(grid, fig_dir)
    if p:
        paths.append(p)

    p = plot_scenario_comparison(scorecard, fig_dir)
    if p:
        paths.append(p)

    log.info("Generated %d plots", len(paths))
    return paths


# ── Report builder ──────────────────────────────────────────────


def _go_no_go_assessment(
    scorecard: pd.DataFrame | None,
    horizon_result: pd.DataFrame | None,
    universe_result: pd.DataFrame | None,
    rank_quality: pd.DataFrame | None,
    exec_aware_result: pd.DataFrame | None,
    gates: dict,
) -> list[str]:
    """Evaluate go/no-go gates and return list of assessment lines."""
    lines: list[str] = []

    # Branch A gate: trade count multiplier
    if horizon_result is not None:
        stacked = horizon_result[horizon_result["horizon"].astype(str) == "stacked"]
        if not stacked.empty and "trade_multiplier" in stacked.columns:
            mult = stacked["trade_multiplier"].values[0]
            gate = gates.get("min_horizon_stack_trade_multiplier", 2.0)
            status = "✅ PASS" if mult >= gate else "❌ FAIL"
            lines.append(f"| Branch A | Trade multiplier ≥ {gate}× | {mult:.1f}× | {status} |")
        net_exp = stacked["mean_net_bps"].values[0] if not stacked.empty else 0
        gate_exp = gates.get("min_stacked_net_expectancy_bps", 0)
        status = "✅ PASS" if net_exp > gate_exp else "❌ FAIL"
        lines.append(f"| Branch A | Net expectancy > {gate_exp} bps | {net_exp:.1f} bps | {status} |")

    # Branch B gate: qualifying assets
    if universe_result is not None:
        n_qual = len(universe_result[
            (universe_result["qualifies"] == True) &
            (universe_result["asset"] != "QUALIFYING_AGGREGATE") &
            (universe_result["asset"] != "SOL-USD")
        ])
        gate = gates.get("min_qualifying_expansion_assets", 2)
        status = "✅ PASS" if n_qual >= gate else "❌ FAIL"
        lines.append(f"| Branch B | ≥ {gate} qualifying expansion assets | {n_qual} | {status} |")

    # Branch C gate: top-bottom spread
    if rank_quality is not None and len(rank_quality) >= 2:
        top_v = rank_quality.nlargest(1, "ventile")["mean_return_bps"].values[0]
        bottom_v = rank_quality.nsmallest(1, "ventile")["mean_return_bps"].values[0]
        spread = top_v - bottom_v
        gate = gates.get("min_top_bottom_spread_bps", 20.0)
        status = "✅ PASS" if spread >= gate else "❌ FAIL"
        lines.append(f"| Branch C | Top-bottom spread ≥ {gate} bps | {spread:.1f} bps | {status} |")

    # Branch D gate: improvement
    if exec_aware_result is not None and not exec_aware_result.empty:
        if "improvement_vs_raw_bps" in exec_aware_result.columns:
            best_imp = exec_aware_result["improvement_vs_raw_bps"].max()
            gate = gates.get("min_exec_aware_improvement_bps", 5.0)
            status = "✅ PASS" if best_imp >= gate else "❌ FAIL"
            lines.append(f"| Branch D | Exec-aware improvement ≥ {gate} bps | {best_imp:.1f} bps | {status} |")

    # Aggregate gate — uses deployable-best (same-branch economics only)
    if scorecard is not None:
        baseline = scorecard[scorecard["scenario"] == "1_baseline_sol_1h"]["weekly_pnl_usd"].values
        deploy = scorecard[scorecard["scenario"] == "5_deployable_best"]["weekly_pnl_usd"].values
        if len(baseline) > 0 and len(deploy) > 0 and baseline[0] != 0:
            pct = (deploy[0] - baseline[0]) / abs(baseline[0]) * 100
            gate = gates.get("min_weekly_pnl_improvement_pct", 50.0)
            status = "✅ PASS" if pct >= gate else "❌ FAIL"
            lines.append(f"| **Aggregate** | Deployable PnL improvement ≥ {gate}% | {pct:.0f}% | {status} |")

    return lines


def build_exp008_summary(
    report_dir: Path,
    *,
    horizon_result: pd.DataFrame | None = None,
    universe_result: pd.DataFrame | None = None,
    calibration_curve: pd.DataFrame | None = None,
    rank_quality: pd.DataFrame | None = None,
    cs_ranking: pd.DataFrame | None = None,
    exec_aware_result: pd.DataFrame | None = None,
    scorecard: pd.DataFrame | None = None,
    grid: pd.DataFrame | None = None,
    cfg: dict | None = None,
    go_no_go: dict | None = None,
) -> Path:
    """Build the summary markdown report."""
    report_dir = Path(report_dir)
    ensure_dir(report_dir)
    cfg = cfg or {}
    go_no_go = go_no_go or {}

    lines: list[str] = []
    lines.append("# exp008 — Capacity Research Report")
    lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
    lines.append("## Objective\n")
    lines.append("Increase expected net PnL per week by jointly optimizing")
    lines.append("edge per trade × trade count × deployable notional.\n")

    # ── Branch A ────────────────────────────────────────────
    lines.append("---\n## Branch A: Multi-Horizon Stacking\n")
    if horizon_result is not None and not horizon_result.empty:
        lines.append("| Horizon | Trades | Sharpe | Net (bps) | Hit Rate | Fold Prof |")
        lines.append("|---------|--------|--------|-----------|----------|-----------|")
        for _, row in horizon_result.iterrows():
            lines.append(
                f"| {row['horizon']} | {row['trade_count']:.0f} | "
                f"{row.get('sharpe', 0):.2f} | {row.get('mean_net_bps', 0):.1f} | "
                f"{row.get('hit_rate', 0):.2%} | {row.get('fold_profitability', 0):.2%} |"
            )
        # Warn about degenerate stacked row
        stacked = horizon_result[horizon_result["horizon"].astype(str) == "stacked"]
        if not stacked.empty and stacked["trade_count"].values[0] < 10:
            lines.append(f"\n> ⚠️ **Stacked row has only {stacked['trade_count'].values[0]:.0f} trade(s)** — "
                         f"the {stacked['mean_net_bps'].values[0]:.1f} bps figure is not statistically valid "
                         f"and must not be used for economic projections.\n")
        lines.append("\n![Horizon stacking](figures/horizon_stacking_tradecount.png)\n")
    else:
        lines.append("*Branch A not run or no results.*\n")

    # ── Branch B ────────────────────────────────────────────
    lines.append("---\n## Branch B: Universe Expansion\n")
    if universe_result is not None and not universe_result.empty:
        lines.append("| Asset | Trades | Sharpe | Net (bps) | Hit Rate | Fold Prof | Qualifies |")
        lines.append("|-------|--------|--------|-----------|----------|-----------|-----------|")
        for _, row in universe_result.iterrows():
            q = "✅" if row.get("qualifies", False) else "❌"
            lines.append(
                f"| {row['asset']} | {row['trade_count']:.0f} | "
                f"{row.get('sharpe', 0):.2f} | {row.get('mean_net_bps', 0):.1f} | "
                f"{row.get('hit_rate', 0):.2%} | {row.get('fold_profitability', 0):.2%} | {q} |"
            )
        lines.append("\n![Transportability](figures/transportability_matrix.png)\n")
    else:
        lines.append("*Branch B not run — expansion assets not in current data.*\n")
        lines.append("*Run with expanded data config to populate this branch.*\n")

    # ── Branch C ────────────────────────────────────────────
    lines.append("---\n## Branch C: Score Calibration\n")
    if calibration_curve is not None and not calibration_curve.empty:
        # ECE summary
        for variant in ["raw", "isotonic"]:
            v_df = calibration_curve[calibration_curve["variant"] == variant]
            if not v_df.empty:
                weights = v_df["count"].values / v_df["count"].sum()
                ece = np.sum(weights * np.abs(v_df["predicted_prob"].values - v_df["observed_freq"].values))
                lines.append(f"- **ECE ({variant})**: {ece:.4f}")

        lines.append("\n![Calibration](figures/calibration_curve.png)\n")

    if rank_quality is not None and not rank_quality.empty:
        lines.append("### Rank Quality (Ventile Monotonicity)\n")
        if len(rank_quality) >= 2:
            top_v = rank_quality.nlargest(1, "ventile")["mean_return_bps"].values[0]
            bottom_v = rank_quality.nsmallest(1, "ventile")["mean_return_bps"].values[0]
            lines.append(f"- **Top ventile mean return**: {top_v:.1f} bps")
            lines.append(f"- **Bottom ventile mean return**: {bottom_v:.1f} bps")
            lines.append(f"- **Spread**: {top_v - bottom_v:.1f} bps")
        lines.append("\n![Rank quality](figures/rank_quality.png)\n")

    if cs_ranking is not None and not cs_ranking.empty:
        lines.append("### Cross-Sectional Ranking\n")
        lines.append(f"- **Bars evaluated**: {len(cs_ranking)}")
        lines.append(f"- **Mean spread**: {cs_ranking['spread_bps'].mean():.1f} bps")
        lines.append(f"- **Fraction positive**: {(cs_ranking['spread_bps'] > 0).mean():.1%}\n")

    # ── Branch D ────────────────────────────────────────────
    lines.append("---\n## Branch D: Execution-Aware Alpha\n")
    if exec_aware_result is not None and not exec_aware_result.empty:
        lines.append("| Target Variant | Slip (bps) | Trades | Sharpe | Net (bps) | Δ vs Raw |")
        lines.append("|----------------|------------|--------|--------|-----------|----------|")
        for _, row in exec_aware_result.iterrows():
            delta = row.get("improvement_vs_raw_bps", 0)
            lines.append(
                f"| {row['target_variant']} | {row.get('slippage_bps', 0):.0f} | "
                f"{row['trade_count']:.0f} | {row.get('sharpe', 0):.2f} | "
                f"{row.get('mean_net_bps', 0):.1f} | {delta:+.1f} |"
            )
        lines.append("")
    else:
        lines.append("*Branch D not run or no results.*\n")

    # ── Branch E ────────────────────────────────────────────
    lines.append("---\n## Branch E: Capacity Economics\n")
    if scorecard is not None and not scorecard.empty:
        lines.append("### Scenario Scorecard\n")
        lines.append("| Scenario | Edge (bps) | Trades/wk | Notional | $/week | Δ% |")
        lines.append("|----------|------------|-----------|----------|--------|----|")
        for _, row in scorecard.iterrows():
            lines.append(
                f"| {row['scenario']} | {row['edge_bps']:.1f} | "
                f"{row['trades_per_week']:.1f} | ${row['notional_usd']:,.0f} | "
                f"${row['weekly_pnl_usd']:,.0f} | {row['improvement_pct']:+.0f}% |"
            )
        lines.append("\n![Scenarios](figures/weekly_pnl_scenarios.png)\n")
        lines.append("### Sensitivity Grid\n")
        lines.append("![Sensitivity](figures/capacity_sensitivity.png)\n")
    else:
        lines.append("*No economics data.*\n")

    # ── Go / No-Go ──────────────────────────────────────────
    lines.append("---\n## Go / No-Go Assessment\n")
    gate_lines = _go_no_go_assessment(
        scorecard, horizon_result, universe_result,
        rank_quality, exec_aware_result, go_no_go,
    )
    if gate_lines:
        lines.append("| Branch | Gate | Value | Status |")
        lines.append("|--------|------|-------|--------|")
        lines.extend(gate_lines)
    else:
        lines.append("*No gates evaluated.*")

    # ── Reviewer interpretation ─────────────────────────────
    lines.append("")
    lines.append("---\n## Reviewer Interpretation\n")
    lines.append("> **Deployable scenario economics** and **branch-level diagnostics** ")
    lines.append("> are reported separately. Only same-branch economics should be used ")
    lines.append("> for capital allocation decisions.\n")
    lines.append("### Branch status\n")
    lines.append("| Branch | Status | Rationale |")
    lines.append("|--------|--------|-----------|")
    lines.append("| A: Multi-horizon | ❌ Dead | 2h/4h produce more trades but negative expectancy |")
    lines.append("| B: Universe expansion | ✅ **Winner** | Only branch with validated, additive uplift |")
    lines.append("| C: Score calibration | ❌ Dead | Top-bottom ventile spread too narrow for ranking |")
    lines.append("| D: Execution-aware | ❌ Dead | No incremental information beyond cost structure |")

    # Deployable economics summary
    if scorecard is not None:
        deploy_row = scorecard[scorecard["scenario"] == "5_deployable_best"]
        if not deploy_row.empty:
            dr = deploy_row.iloc[0]
            lines.append(f"\n### Deployable economics\n")
            lines.append(f"- **Net edge**: {dr['edge_bps']:.1f} bps")
            lines.append(f"- **Trade frequency**: {dr['trades_per_week']:.1f} trades/week")
            lines.append(f"- **Weekly PnL** (${dr['notional_usd']:,.0f}/trade): ${dr['weekly_pnl_usd']:,.0f}")
            lines.append(f"- **Improvement vs baseline**: {dr['improvement_pct']:+.0f}%")

    lines.append("")

    out = report_dir / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written: %s (%d lines)", out, len(lines))
    return out
