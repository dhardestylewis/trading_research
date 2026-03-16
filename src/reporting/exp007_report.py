"""exp007 report builder.

Generates summary.md for the live paper canary study.

Report structure:
  1. Scope declaration (realized vs simulated execution fidelity)
  2. Metric dictionary
  3. Simulated vs Realized comparison table — side-by-side per lane
  4. Shortfall analysis — realized shortfall vs simulated entry, per lane
  5. Canary health scorecard — pass/fail per tolerance band
  6. Go/No-Go verdict — two-tier (Stage 2 canary or HOLD)
  7. Plots
  8. Stage gates status
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.io import ensure_dir
from src.utils.logging import get_logger

log = get_logger("exp007_report")


# ═══════════════════════════════════════════════════════════════════
#  Plots
# ═══════════════════════════════════════════════════════════════════

def plot_simulated_vs_realized_deltas(comparison: pd.DataFrame, fig_dir: Path):
    """Bar chart of metric deltas: realized − simulated, per lane."""
    fig_dir = ensure_dir(fig_dir)
    if comparison.empty:
        return

    plot_metrics = [
        "realized_fill_rate", "mean_slippage_bps", "cancel_rate",
        "mean_return_per_trade_bps", "sharpe", "shortfall_vs_simulated_bps",
    ]
    df = comparison[comparison["metric"].isin(plot_metrics)].copy()
    if df.empty:
        return

    lanes = df["lane"].unique()
    n_lanes = len(lanes)
    fig, axes = plt.subplots(1, n_lanes, figsize=(6 * n_lanes, 5), squeeze=False)

    for idx, lane in enumerate(sorted(lanes)):
        ax = axes[0, idx]
        lane_df = df[df["lane"] == lane]
        colors = ["#e74c3c" if d else "#27ae60" for d in lane_df["degradation_flag"]]
        bars = ax.barh(lane_df["metric"], lane_df["delta"].fillna(0), color=colors, alpha=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"Δ (realized − simulated)\n{lane}", fontsize=11)
        ax.set_xlabel("Delta")
        for i, v in enumerate(lane_df["delta"].fillna(0)):
            ax.text(v, i, f" {v:+.2f}", va="center", fontsize=8)

    fig.suptitle("Realized vs Simulated: Metric Deltas", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "simulated_vs_realized_deltas.png", dpi=150)
    plt.close(fig)


def plot_shortfall_analysis(log_df: pd.DataFrame, fig_dir: Path):
    """Distribution of shortfall vs simulated entry, by lane."""
    fig_dir = ensure_dir(fig_dir)
    filled = log_df[log_df["cancel_status"].isin(["filled", "partial_fill"])].copy()
    if filled.empty or "realized_shortfall_vs_simulated" not in filled.columns:
        return

    lane_col = "entry_mode" if "entry_mode" in filled.columns else None
    if lane_col:
        lanes = filled[lane_col].unique()
    else:
        lanes = ["all"]
        filled["_lane"] = "all"
        lane_col = "_lane"

    fig, axes = plt.subplots(1, len(lanes), figsize=(5 * len(lanes), 5), squeeze=False)

    for idx, lane in enumerate(sorted(lanes)):
        ax = axes[0, idx]
        lane_df = filled[filled[lane_col] == lane]
        shortfall = lane_df["realized_shortfall_vs_simulated"].dropna()

        ax.hist(shortfall, bins=25, color="#8e44ad", alpha=0.7, edgecolor="white")
        ax.axvline(shortfall.mean(), color="#e74c3c", linewidth=2, linestyle="--",
                   label=f"Mean = {shortfall.mean():.2f} bps")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Shortfall (bps)")
        ax.set_ylabel("Count")
        ax.set_title(f"Shortfall vs Simulated\n{lane}")
        ax.legend(fontsize=8)

    fig.suptitle("Realized Shortfall vs Simulated Entry Expectation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "shortfall_analysis.png", dpi=150)
    plt.close(fig)


def plot_weekly_stability(weekly: pd.DataFrame, fig_dir: Path):
    """Weekly execution error trend."""
    fig_dir = ensure_dir(fig_dir)
    if weekly.empty:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.bar(weekly["week"], weekly["mean_shortfall_bps"], color="#3498db", alpha=0.7)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("Mean Shortfall (bps)")
    ax1.set_title("Weekly Mean Shortfall vs Simulated")

    ax2.bar(weekly["week"], weekly["execution_error_bps"].fillna(0), color="#e67e22", alpha=0.7)
    ax2.set_ylabel("Execution Error (bps)")
    ax2.set_xlabel("ISO Week")
    ax2.set_title("Weekly Execution Error (Stability)")

    fig.suptitle("Execution Quality Stability Over Time", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "weekly_stability.png", dpi=150)
    plt.close(fig)


def plot_lane_overlay(
    simulated_summary: pd.DataFrame,
    realized_summary: pd.DataFrame,
    fig_dir: Path,
):
    """Side-by-side lane metrics: simulated vs realized."""
    fig_dir = ensure_dir(fig_dir)

    metrics_to_compare = [
        ("realized_fill_rate", "Fill Rate"),
        ("mean_slippage_bps", "Mean Slippage (bps)"),
        ("mean_return_per_trade_bps", "Return/Trade (bps)"),
        ("sharpe", "Sharpe"),
    ]

    sim_lane_col = "lane" if "lane" in simulated_summary.columns else None
    real_lane_col = "lane" if "lane" in realized_summary.columns else None
    if sim_lane_col is None or real_lane_col is None:
        return

    available = [(m, t) for m, t in metrics_to_compare
                 if m in simulated_summary.columns and m in realized_summary.columns]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, (metric, title) in zip(axes, available):
        sim_vals = simulated_summary.set_index(sim_lane_col)[metric]
        real_vals = realized_summary.set_index(real_lane_col)[metric]

        lanes = sorted(set(sim_vals.index) | set(real_vals.index))
        x = np.arange(len(lanes))
        w = 0.35

        ax.barh(x - w/2, [sim_vals.get(l, 0) for l in lanes], w, label="Simulated", color="#3498db", alpha=0.7)
        ax.barh(x + w/2, [real_vals.get(l, 0) for l in lanes], w, label="Realized", color="#e74c3c", alpha=0.7)
        ax.set_yticks(x)
        ax.set_yticklabels(lanes, fontsize=8)
        ax.set_xlabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle("Simulated vs Realized: Lane Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "lane_overlay.png", dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Summary Report
# ═══════════════════════════════════════════════════════════════════

def build_exp007_summary(
    report_dir: Path,
    simulated_summary: pd.DataFrame,
    realized_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    health_table: pd.DataFrame,
    weekly: pd.DataFrame,
    all_passed: bool,
    tolerance_bands: dict,
    policy_cfg: dict,
    metric_dict: dict,
    stage_gates: dict,
    mode: str,
    cfg: dict,
) -> Path:
    """Build exp007 summary.md."""
    lines: list[str] = []
    lines.append("# Experiment Report: crypto_1h_exp007\n")

    # ── Scope ────────────────────────────────────────────────────
    lines.append("## Scope\n")
    if mode == "simulate":
        lines.append(
            "**Self-comparison mode.** This report validates the canary pipeline "
            "by comparing exp006 simulated data against itself. All health checks "
            "should pass by construction. The pipeline is ready for live paper data.\n"
        )
    else:
        lines.append(
            "**Realized exchange execution fidelity.** This report compares live "
            "paper-trade fills against the simulated baseline from exp006. The key "
            "question: is the live environment merely noisier, or structurally worse "
            "than the backtest execution model?\n"
        )

    # ── Metric Dictionary ────────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Metric Dictionary\n")
    if metric_dict:
        lines.append("| Metric | Operational Definition |\n")
        lines.append("|:-------|:-----------------------|\n")
        for metric_name, defn in metric_dict.items():
            clean_defn = " ".join(defn.strip().split())
            lines.append(f"| **{metric_name}** | {clean_defn} |\n")
        lines.append("\n")

    # ── Policy ───────────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## Policy Specification\n")
    lines.append("```\n")
    lines.append(f"Training:     {policy_cfg.get('training_mode', 'pooled_train_sol_deploy')}\n")
    lines.append(f"Threshold:    τ = {policy_cfg.get('threshold', 0.55)}\n")
    lines.append(f"Sep gap:      {policy_cfg.get('sep_gap', 3)} bars\n")
    lines.append(f"Regime gate:  {policy_cfg.get('regime_gate', 'NOT_rebound')}\n")
    lines.append(f"Cost:         {policy_cfg.get('cost_bps', 15.0)} bps round-trip\n")
    lines.append("```\n")

    # ── Simulated vs Realized ────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Simulated vs Realized Comparison\n")

    if not comparison.empty:
        # Summary table showing key metrics per lane
        pivot_metrics = [
            "realized_fill_rate", "mean_slippage_bps", "cancel_rate",
            "mean_return_per_trade_bps", "sharpe", "shortfall_vs_simulated_bps",
        ]
        key_comp = comparison[comparison["metric"].isin(pivot_metrics)].copy()
        if not key_comp.empty:
            display = key_comp[["lane", "metric", "simulated", "realized", "delta", "degradation_flag"]].copy()
            display["status"] = display["degradation_flag"].map({True: "⚠️", False: "✅"})
            display = display.drop(columns=["degradation_flag"])
            lines.append(display.to_markdown(index=False))
            lines.append("\n\n")

            n_degraded = key_comp["degradation_flag"].sum()
            if n_degraded == 0:
                lines.append("**No structural degradation detected across any lane.**\n")
            else:
                lines.append(
                    f"**⚠️ {n_degraded} metric-lane pair(s) show degradation.**\n"
                )
    else:
        lines.append("⚠️ No comparison data available\n")

    # ── Shortfall Analysis ───────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Shortfall vs Simulated Entry\n")
    lines.append(
        "> **Key diagnostic**: distinguishes 'live is merely noisier' from "
        "'live is structurally worse than the backtest execution model.'\n\n"
    )

    if not realized_summary.empty and "shortfall_vs_simulated_bps" in realized_summary.columns:
        for _, row in realized_summary.iterrows():
            lane = row.get("lane", "unknown")
            sf = row.get("shortfall_vs_simulated_bps", np.nan)
            pct_worse = row.get("pct_structurally_worse", np.nan)
            if np.isfinite(sf):
                lines.append(f"- **{lane}**: mean shortfall = {sf:.2f} bps, "
                             f"{pct_worse:.0%} of trades structurally worse\n")
        lines.append("\n")

    # ── Canary Health Scorecard ──────────────────────────────────
    lines.append("---\n")
    lines.append("## Canary Health Scorecard\n")
    lines.append("> Tolerance bands from config. All bands must hold for Stage 2 advancement.\n\n")

    if not health_table.empty:
        ht = health_table.copy()
        ht["status"] = ht["passed"].map({True: "✅", False: "❌"})
        ht["value_fmt"] = ht["value"].map(lambda v: f"{v:.4f}" if np.isfinite(v) else "N/A")
        ht["threshold_fmt"] = ht.apply(
            lambda r: f"{r['direction']} {r['threshold']:.4f}", axis=1
        )
        display = ht[["lane", "metric", "value_fmt", "threshold_fmt", "status"]]
        display = display.rename(columns={
            "value_fmt": "Value",
            "threshold_fmt": "Threshold",
            "status": "Status",
        })
        lines.append(display.to_markdown(index=False))
        lines.append("\n\n")

    # ── Weekly Stability ─────────────────────────────────────────
    if not weekly.empty:
        lines.append("---\n")
        lines.append("## Weekly Execution Error\n")
        lines.append(weekly.to_markdown(index=False))
        lines.append("\n\n")

        max_weekly_err = tolerance_bands.get("max_weekly_execution_error_bps", 20.0)
        worst_week = weekly["execution_error_bps"].max()
        if np.isfinite(worst_week) and worst_week <= max_weekly_err:
            lines.append(f"✅ Worst-week execution error ({worst_week:.1f} bps) within "
                         f"tolerance ({max_weekly_err:.0f} bps)\n")
        elif np.isfinite(worst_week):
            lines.append(f"❌ Worst-week execution error ({worst_week:.1f} bps) exceeds "
                         f"tolerance ({max_weekly_err:.0f} bps)\n")

    # ── Go / No-Go Verdict ───────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Go / No-Go Verdict\n")
    lines.append("> Two-tier assessment: Stage 2 canary advancement vs HOLD.\n\n")

    if mode == "simulate":
        lines.append(
            "**ℹ️ Pipeline validation mode.** This report was generated in self-comparison "
            "mode against simulated data. Verdicts below will become meaningful when fed "
            "live paper-trade data.\n\n"
        )

    if all_passed:
        lines.append("**✅ GO — realized execution fidelity within tolerance bands.**\n\n")
        lines.append(
            "Stage 2 advancement criteria met. Recommend proceeding to tiny-capital canary "
            "on the primary marketable lane, keeping passive lanes as shadow monitors.\n"
        )
    else:
        lines.append("**⚠️ HOLD — one or more tolerance bands violated.**\n\n")
        # List failures
        if not health_table.empty:
            failures = health_table[~health_table["passed"]]
            for _, row in failures.iterrows():
                lines.append(
                    f"- ❌ **{row['lane']}** / {row['metric']}: "
                    f"{row['value']:.4f} vs {row['direction']} {row['threshold']:.4f}\n"
                )
        lines.append("\nDo not advance to Stage 2 until all bands hold.\n")

    # ── Stage Gates Status ───────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Stage Gates\n")
    for stage_name, stage_cfg in stage_gates.items():
        desc = stage_cfg.get("description", stage_name) if isinstance(stage_cfg, dict) else stage_name
        lines.append(f"\n### {stage_name}\n")
        lines.append(f"{desc}\n\n")
        reqs = stage_cfg.get("requirements", []) if isinstance(stage_cfg, dict) else []
        prereqs = stage_cfg.get("prerequisites", []) if isinstance(stage_cfg, dict) else []
        if prereqs:
            lines.append("**Prerequisites:**\n")
            for p in prereqs:
                lines.append(f"- {p}\n")
        if reqs:
            lines.append("**Requirements:**\n")
            for r in reqs:
                lines.append(f"- {r}\n")

    # ── Figures ──────────────────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Figures\n")
    fig_dir = report_dir / "figures"
    if fig_dir.exists():
        for fig_path in sorted(fig_dir.glob("*.png")):
            lines.append(f"![{fig_path.stem}]({fig_path.name})\n")

    # ── Next Steps ───────────────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Next Steps\n")
    if mode == "simulate":
        lines.append("1. Configure exchange sandbox credentials\n")
        lines.append("2. Run `python run_exp007.py --mode live` to begin canary\n")
        lines.append("3. Collect 7–14 days of live paper data\n")
        lines.append("4. Re-run this report on live data to get realized verdict\n")
    else:
        if all_passed:
            lines.append("1. Monitor canary health daily for remaining duration\n")
            lines.append("2. If all bands hold for 7+ days → advance to Stage 2 tiny-capital\n")
            lines.append("3. Start Stage 2 with primary marketable lane only\n")
            lines.append("4. Keep passive lanes as shadow monitors\n")
        else:
            lines.append("1. Investigate out-of-tolerance metrics\n")
            lines.append("2. Determine if degradation is structural or transient\n")
            lines.append("3. Re-run canary after addressing root cause\n")

    out_path = report_dir / "summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
