"""exp010 report generator — Family Canary Deployment Validation.

Produces:
  - Per-asset execution scorecard
  - Cross-asset fill quality comparison chart
  - Family weekly PnL time series
  - exp008/exp009 reconciliation appendix
  - Go/No-Go assessment
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

log = get_logger("exp010_report")


# ── Plot generators ─────────────────────────────────────────


def plot_execution_scorecard(scorecard: pd.DataFrame, fig_dir: Path) -> Path | None:
    """Grouped bar chart: net return bps + fill rate per asset."""
    if scorecard is None or scorecard.empty:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    assets = scorecard["asset"].values
    x = np.arange(len(assets))

    # Net return per trade
    net_bps = scorecard["mean_return_per_trade_bps"].values
    lane_types = scorecard["lane_type"].values if "lane_type" in scorecard.columns else ["unknown"] * len(assets)
    color_map = {"primary": "#2196F3", "secondary_shadow": "#FF9800", "research_shadow": "#9E9E9E"}
    colors = [color_map.get(lt, "#9E9E9E") for lt in lane_types]

    ax1.bar(x, net_bps, color=colors, alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(assets, rotation=30, ha="right")
    ax1.set_ylabel("Net Return per Filled Trade (bps)")
    ax1.set_title("Per-Asset Net Execution Return")
    ax1.axhline(y=0, color="black", linewidth=0.5)
    for i, v in enumerate(net_bps):
        if np.isfinite(v):
            ax1.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)

    # Fill rate
    fill_rates = scorecard["realized_fill_rate"].values * 100
    ax2.bar(x, fill_rates, color=colors, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(assets, rotation=30, ha="right")
    ax2.set_ylabel("Fill Rate (%)")
    ax2.set_title("Per-Asset Fill Rate")
    ax2.set_ylim(0, 105)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color="#2196F3", alpha=0.85, label="Primary"),
        Patch(color="#FF9800", alpha=0.85, label="Secondary Shadow"),
        Patch(color="#9E9E9E", alpha=0.85, label="Research Shadow"),
    ]
    ax2.legend(handles=legend_handles, loc="lower right")

    plt.tight_layout()
    out = fig_dir / "execution_scorecard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_fill_profile(fill_profile: pd.DataFrame, fig_dir: Path) -> Path | None:
    """KS test p-value heatmap for fill quality comparison."""
    if fill_profile is None or fill_profile.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, max(3, len(fill_profile) * 1.2)))

    metrics = ["slippage_ks_pvalue", "adverse_ks_pvalue", "pnl_ks_pvalue"]
    labels = ["Slippage", "Adverse Sel.", "Realized PnL"]
    available = [m for m in metrics if m in fill_profile.columns]
    avail_labels = [labels[metrics.index(m)] for m in available]

    if not available:
        plt.close(fig)
        return None

    data = fill_profile[available].values
    comparisons = fill_profile["comparison"].values

    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(avail_labels)))
    ax.set_xticklabels(avail_labels)
    ax.set_yticks(np.arange(len(comparisons)))
    ax.set_yticklabels(comparisons)
    ax.set_title(f"KS Test p-values vs SOL-USD (green = similar)")

    for i in range(len(comparisons)):
        for j in range(len(available)):
            val = data[i, j]
            text = f"{val:.3f}" if np.isfinite(val) else "N/A"
            color = "black" if val > 0.3 else "white"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    out = fig_dir / "fill_profile_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_weekly_pnl(weekly_pnl: pd.DataFrame, fig_dir: Path) -> Path | None:
    """Time series of per-asset + total weekly PnL."""
    if weekly_pnl is None or weekly_pnl.empty:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    week_col = "week" if "week" in weekly_pnl.columns else weekly_pnl.columns[0]
    weeks = weekly_pnl[week_col]

    # Per-asset weekly PnL
    asset_cols = [c for c in weekly_pnl.columns if c not in [week_col, "total"]]
    for col in asset_cols:
        ax1.plot(weeks, weekly_pnl[col], label=col, alpha=0.8)
    ax1.set_ylabel("Weekly PnL (return units)")
    ax1.set_title("Per-Asset Weekly PnL")
    ax1.legend()
    ax1.axhline(y=0, color="black", linewidth=0.5)

    # Total portfolio
    if "total" in weekly_pnl.columns:
        ax2.bar(weeks, weekly_pnl["total"], alpha=0.7, color="steelblue")
        cumulative = weekly_pnl["total"].cumsum()
        ax2_twin = ax2.twinx()
        ax2_twin.plot(weeks, cumulative, color="darkred", linewidth=2, label="Cumulative")
        ax2_twin.set_ylabel("Cumulative PnL", color="darkred")
        ax2_twin.legend(loc="upper left")

    ax2.set_ylabel("Weekly PnL (return units)")
    ax2.set_title("Family Portfolio Weekly PnL")
    ax2.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    out = fig_dir / "weekly_pnl.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_adverse_selection_by_asset(
    per_asset_logs: dict[str, pd.DataFrame],
    fig_dir: Path,
) -> Path | None:
    """Box plot of adverse selection distributions per asset."""
    if not per_asset_logs:
        return None

    data = {}
    for asset, log_df in per_asset_logs.items():
        filled = log_df[log_df["cancel_status"] == "filled"]
        valid = filled.dropna(subset=["fill_price", "midprice_after_15m"])
        if valid.empty:
            continue
        adv = -(valid["midprice_after_15m"] - valid["fill_price"]) / valid["fill_price"] * 10_000
        data[asset] = adv.values

    if not data:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(data.keys())
    ax.boxplot(
        [data[a] for a in labels],
        labels=labels,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", alpha=0.7),
    )
    ax.set_ylabel("Adverse Selection (bps)")
    ax.set_title("Adverse Selection Distribution by Asset (15m window)")
    ax.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    out = fig_dir / "adverse_selection_by_asset.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_all_plots(
    fig_dir: Path,
    *,
    execution_scorecard: pd.DataFrame | None = None,
    fill_profile: pd.DataFrame | None = None,
    weekly_pnl: pd.DataFrame | None = None,
    per_asset_logs: dict[str, pd.DataFrame] | None = None,
) -> list[Path]:
    """Generate all exp010 plots."""
    fig_dir = ensure_dir(fig_dir)
    paths: list[Path] = []

    p = plot_execution_scorecard(execution_scorecard, fig_dir)
    if p:
        paths.append(p)

    p = plot_fill_profile(fill_profile, fig_dir)
    if p:
        paths.append(p)

    p = plot_weekly_pnl(weekly_pnl, fig_dir)
    if p:
        paths.append(p)

    if per_asset_logs:
        p = plot_adverse_selection_by_asset(per_asset_logs, fig_dir)
        if p:
            paths.append(p)

    log.info("Generated %d plots", len(paths))
    return paths


# ── Go/No-Go evaluation ────────────────────────────────────────


def _evaluate_go_no_go(
    execution_scorecard: pd.DataFrame | None,
    fill_profile: pd.DataFrame | None,
    portfolio_stats: dict | None,
    gates: dict,
) -> list[str]:
    """Evaluate the four exp010 go/no-go gates."""
    lines: list[str] = []
    portfolio_stats = portfolio_stats or {}

    # Q1: SUI edge retention
    if execution_scorecard is not None and not execution_scorecard.empty:
        sui_rows = execution_scorecard[execution_scorecard["asset"] == "SUI-USD"]
        if not sui_rows.empty:
            sui_bps = sui_rows["mean_return_per_trade_bps"].values[0]
            sui_trades = sui_rows["submitted"].values[0] if "submitted" in sui_rows.columns else 0
            gate_bps = gates.get("sui_min_net_bps", 10.0)
            gate_trades = gates.get("sui_min_trade_count", 50)
            bps_pass = np.isfinite(sui_bps) and sui_bps >= gate_bps
            trades_pass = sui_trades >= gate_trades
            status = "✅ PASS" if (bps_pass and trades_pass) else "❌ FAIL"
            lines.append(f"| SUI edge | Net ≥ {gate_bps} bps, trades ≥ {gate_trades} | "
                         f"{sui_bps:.1f} bps, {sui_trades} trades | {status} |")

    # Q2: NEAR friction survival
    if execution_scorecard is not None and not execution_scorecard.empty:
        near_rows = execution_scorecard[execution_scorecard["asset"] == "NEAR-USD"]
        if not near_rows.empty:
            near_bps = near_rows["mean_return_per_trade_bps"].values[0]
            near_trades = near_rows["submitted"].values[0] if "submitted" in near_rows.columns else 0
            gate_bps = gates.get("near_min_net_bps", 5.0)
            gate_trades = gates.get("near_min_trade_count", 50)
            bps_pass = np.isfinite(near_bps) and near_bps >= gate_bps
            trades_pass = near_trades >= gate_trades
            status = "✅ PASS" if (bps_pass and trades_pass) else "❌ FAIL"
            lines.append(f"| NEAR friction | Net ≥ {gate_bps} bps, trades ≥ {gate_trades} | "
                         f"{near_bps:.1f} bps, {near_trades} trades | {status} |")

    # Q3: Family pool weekly PnL
    n_weeks = portfolio_stats.get("n_weeks", 0)
    n_positive = portfolio_stats.get("n_positive_weeks", 0)
    pct_positive = n_positive / n_weeks if n_weeks > 0 else 0
    gate_pct = gates.get("min_positive_week_pct", 0.50)
    status = "✅ PASS" if pct_positive >= gate_pct else "❌ FAIL"
    lines.append(f"| Weekly PnL | ≥ {gate_pct:.0%} positive weeks | "
                 f"{pct_positive:.0%} ({n_positive}/{n_weeks}) | {status} |")

    # Q4: Fill quality similarity
    if fill_profile is not None and not fill_profile.empty:
        alpha = gates.get("fill_quality_ks_alpha", 0.05)
        all_similar = fill_profile["profiles_similar"].all() if "profiles_similar" in fill_profile.columns else False
        status = "✅ PASS" if all_similar else "❌ FAIL"
        n_similar = fill_profile["profiles_similar"].sum() if "profiles_similar" in fill_profile.columns else 0
        lines.append(f"| Fill quality | All KS p > {alpha} | "
                     f"{n_similar}/{len(fill_profile)} pairs similar | {status} |")

    return lines


# ── Report builder ──────────────────────────────────────────


def build_exp010_summary(
    report_dir: Path,
    *,
    execution_scorecard: pd.DataFrame | None = None,
    fill_profile: pd.DataFrame | None = None,
    weekly_pnl: pd.DataFrame | None = None,
    portfolio_stats: dict | None = None,
    reconciliation: pd.DataFrame | None = None,
    fold_results: pd.DataFrame | None = None,
    cfg: dict | None = None,
    go_no_go: dict | None = None,
) -> Path:
    """Build the summary markdown report."""
    report_dir = Path(report_dir)
    ensure_dir(report_dir)
    cfg = cfg or {}
    go_no_go = go_no_go or {}
    portfolio_stats = portfolio_stats or {}

    lines: list[str] = []
    lines.append("# exp010 — Family Canary Deployment Validation")
    lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
    lines.append("## Objective\n")
    lines.append("Freeze the architecture from exp009 (family-pooled training,")
    lines.append("asset-specific deployment) and validate across the L1 family.")
    lines.append("Produce reconciliation appendix resolving exp008/exp009 SOL contradiction.\n")

    # ── Execution scorecard ─────────────────────────────────────
    lines.append("---\n## Per-Asset Execution Scorecard\n")
    if execution_scorecard is not None and not execution_scorecard.empty:
        lines.append("| Asset | Lane | Signals | Fill Rate | Net bps/Trade | Sharpe | Adv Sel (bps) |")
        lines.append("|-------|------|---------|-----------|---------------|--------|---------------|")
        for _, row in execution_scorecard.iterrows():
            lines.append(
                f"| {row['asset']} | {row.get('lane_type', '')} | "
                f"{row.get('submitted', 0):.0f} | "
                f"{row.get('realized_fill_rate', 0):.1%} | "
                f"{row.get('mean_return_per_trade_bps', np.nan):.1f} | "
                f"{row.get('sharpe', np.nan):.2f} | "
                f"{row.get('mean_adverse_selection_bps', np.nan):.1f} |"
            )
        lines.append("\n![Execution scorecard](figures/execution_scorecard.png)\n")
    else:
        lines.append("*No execution data available.*\n")

    # ── Cross-asset fill profile ────────────────────────────────
    lines.append("---\n## Cross-Asset Fill Quality (vs SOL-USD)\n")
    if fill_profile is not None and not fill_profile.empty:
        lines.append("| Comparison | Slippage p | Adverse p | PnL p | Similar? |")
        lines.append("|------------|-----------|-----------|-------|----------|")
        for _, row in fill_profile.iterrows():
            sim = "✅" if row.get("profiles_similar", False) else "❌"
            lines.append(
                f"| {row['comparison']} | "
                f"{row.get('slippage_ks_pvalue', np.nan):.3f} | "
                f"{row.get('adverse_ks_pvalue', np.nan):.3f} | "
                f"{row.get('pnl_ks_pvalue', np.nan):.3f} | {sim} |"
            )
        lines.append("\n![Fill profile](figures/fill_profile_comparison.png)\n")
        lines.append("![Adverse selection](figures/adverse_selection_by_asset.png)\n")
    else:
        lines.append("*No fill profile data available.*\n")

    # ── Family weekly PnL ───────────────────────────────────────
    lines.append("---\n## Family Portfolio Weekly PnL\n")
    if portfolio_stats:
        lines.append(f"- **Total PnL**: {portfolio_stats.get('total_pnl', 0):.6f}")
        lines.append(f"- **Mean weekly PnL**: {portfolio_stats.get('mean_weekly_pnl', 0):.6f}")
        lines.append(f"- **Weekly Sharpe**: {portfolio_stats.get('weekly_sharpe', 0):.2f}")
        lines.append(f"- **Max drawdown**: {portfolio_stats.get('max_drawdown', 0):.6f}")
        lines.append(f"- **Positive weeks**: {portfolio_stats.get('n_positive_weeks', 0)}/{portfolio_stats.get('n_weeks', 0)}")
        contrib = portfolio_stats.get("per_asset_contribution", {})
        if contrib:
            lines.append("\n**Per-asset contribution:**\n")
            for asset, pnl in sorted(contrib.items(), key=lambda x: -x[1]):
                lines.append(f"- {asset}: {pnl:.6f}")
        lines.append("\n![Weekly PnL](figures/weekly_pnl.png)\n")
    else:
        lines.append("*No weekly PnL data available.*\n")

    # ── Family fold results ─────────────────────────────────────
    if fold_results is not None and not fold_results.empty:
        lines.append("---\n## Family Fold-Level Summary\n")
        agg = fold_results.groupby("asset").agg(
            trades=("trade_count", "sum"),
            mean_net_bps=("mean_net_bps", "mean"),
            folds=("fold", "nunique"),
            fold_prof=("profitable", "mean"),
        ).reset_index()
        lines.append("| Asset | Trades | Net bps | Folds | Fold Prof |")
        lines.append("|-------|--------|---------|-------|-----------|")
        for _, row in agg.iterrows():
            lines.append(
                f"| {row['asset']} | {row['trades']:.0f} | "
                f"{row['mean_net_bps']:.1f} | {row['folds']:.0f} | "
                f"{row['fold_prof']:.1%} |"
            )
        lines.append("")

    # ── Reconciliation appendix ─────────────────────────────────
    lines.append("---\n## Appendix: exp008/exp009 Reconciliation\n")
    if reconciliation is not None and not reconciliation.empty:
        lines.append("> This table compares the **same policy parameters** across")
        lines.append("> exp008 Branch B and exp009 Branches A/B/C to diagnose the")
        lines.append("> SOL sign flip (exp008: −20.6 bps → exp009: +37.4 bps).\n")

        # Build markdown table
        display_cols = [
            "experiment_id", "branch", "training_pool", "deploy_asset",
            "threshold", "sep_gap", "regime_gate", "cost_bps",
            "trade_count", "fold_count", "mean_net_bps", "sharpe",
            "data_pipeline", "known_bugs_fixed",
        ]
        avail_cols = [c for c in display_cols if c in reconciliation.columns]

        lines.append("| " + " | ".join(avail_cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(avail_cols)) + " |")
        for _, row in reconciliation.iterrows():
            vals = []
            for c in avail_cols:
                v = row[c]
                if isinstance(v, float) and np.isfinite(v):
                    vals.append(f"{v:.1f}" if abs(v) < 1000 else f"{v:.0f}")
                elif isinstance(v, float):
                    vals.append("N/A")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")

        lines.append("\n**Interpretation**: If policy parameters are identical but results")
        lines.append("differ, the most likely explanation is a data pipeline fix applied")
        lines.append("between exp008 and exp009 (e.g. mixed history length handling).\n")
    else:
        lines.append("*No reconciliation data available.*\n")

    # ── Go / No-Go ──────────────────────────────────────────────
    lines.append("---\n## Go / No-Go Assessment\n")
    gate_lines = _evaluate_go_no_go(
        execution_scorecard, fill_profile, portfolio_stats, go_no_go,
    )
    if gate_lines:
        lines.append("| Gate | Criterion | Value | Status |")
        lines.append("|------|-----------|-------|--------|")
        lines.extend(gate_lines)
    else:
        lines.append("*No gates evaluated.*")

    lines.append("")

    out = report_dir / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written: %s (%d lines)", out, len(lines))
    return out
