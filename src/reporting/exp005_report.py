"""exp005 report builder.

Generates summary.md for the live fill validation experiment.
Outputs a policy specification, entry mode × penalty scenario matrix,
go/no-go assessment, and recommended paper-deployment config.
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

log = get_logger("exp005_report")


# ═══════════════════════════════════════════════════════════════════
#  Plots
# ═══════════════════════════════════════════════════════════════════

def plot_penalty_heatmap(penalty_df: pd.DataFrame, entry_mode: str, fig_dir: Path):
    """Heatmap of Sharpe for fill_haircut × price_penalty for one entry mode."""
    fig_dir = ensure_dir(fig_dir)
    sub = penalty_df[penalty_df["entry_mode"] == entry_mode]
    if sub.empty:
        return

    piv = sub.pivot_table(
        index="fill_probability_haircut",
        columns="fill_price_penalty_bps",
        values="sharpe",
        aggfunc="mean",
    )
    if piv.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([f"+{v:.0f}bps" for v in piv.columns])
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([f"{v:.0%}" for v in piv.index])
    ax.set_xlabel("Fill Price Penalty (bps)")
    ax.set_ylabel("Fill Probability Haircut")
    ax.set_title(f"Stressed Sharpe — {entry_mode}")

    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if abs(val) > 3 else "black")

    fig.colorbar(im, label="Sharpe")
    fig.tight_layout()
    safe_name = entry_mode.replace(" ", "_").replace("-", "_")
    fig.savefig(fig_dir / f"penalty_heatmap_{safe_name}.png", dpi=150)
    plt.close(fig)


def plot_entry_mode_comparison(summary_df: pd.DataFrame, fig_dir: Path):
    """Bar chart comparing entry modes at baseline (no penalty)."""
    fig_dir = ensure_dir(fig_dir)
    if summary_df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Sharpe
    ax = axes[0]
    colors = ["#27ae60" if v > 0 else "#c0392b" for v in summary_df["sharpe"]]
    ax.barh(summary_df["entry_mode"], summary_df["sharpe"], color=colors)
    ax.set_xlabel("Sharpe Ratio")
    ax.set_title("Sharpe by Entry Mode")
    ax.axvline(0, color="black", linewidth=0.8)

    # Fill rate
    ax = axes[1]
    ax.barh(summary_df["entry_mode"], summary_df.get("fill_rate", 1.0),
            color="#3498db", alpha=0.7)
    ax.set_xlabel("Fill Rate")
    ax.set_title("Fill Rate by Entry Mode")
    ax.set_xlim(0, 1.1)

    # Fold profitability
    ax = axes[2]
    ax.barh(summary_df["entry_mode"], summary_df.get("fold_profitability", np.nan),
            color="#9b59b6", alpha=0.7)
    ax.set_xlabel("Fold Profitability")
    ax.set_title("Fold Profitability by Entry Mode")
    ax.axvline(0.7, color="red", linestyle="--", linewidth=0.8, label="Go/No-Go Gate")
    ax.set_xlim(0, 1.1)
    ax.legend()

    fig.tight_layout()
    fig.savefig(fig_dir / "entry_mode_comparison.png", dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Summary Report
# ═══════════════════════════════════════════════════════════════════

def build_exp005_summary(
    report_dir: Path,
    penalty_df: pd.DataFrame,
    entry_summary_df: pd.DataFrame,
    go_no_go_cfg: dict,
    policy_cfg: dict,
    cfg: dict,
) -> Path:
    """Build exp005 summary.md.

    Parameters
    ----------
    report_dir : Path
        Output directory for report files.
    penalty_df : pd.DataFrame
        Full penalty grid results (entry_mode × haircut × penalty).
    entry_summary_df : pd.DataFrame
        One row per entry mode at baseline (no penalty).
    go_no_go_cfg : dict
        Go/no-go gate thresholds.
    policy_cfg : dict
        Policy specification.
    cfg : dict
        Full experiment config.
    """
    lines: list[str] = []
    lines.append("# Experiment Report: crypto_1h_exp005\n")
    lines.append("## Objective\n")
    lines.append("**Validate whether the production-candidate policy survives "
                 "queue/priority penalty stress testing before paper deployment.**\n")

    # ── Policy Specification ─────────────────────────────────────
    lines.append("---\n")
    lines.append("## Policy Specification\n")
    lines.append("```\n")
    lines.append(f"Training:     {policy_cfg.get('training_mode', 'pooled_train_sol_deploy')}\n")
    lines.append(f"Threshold:    τ = {policy_cfg.get('threshold', 0.55)}\n")
    lines.append(f"Sep gap:      {policy_cfg.get('sep_gap', 3)} bars\n")
    lines.append(f"Regime gate:  {policy_cfg.get('regime_gate', 'NOT_rebound')}\n")
    lines.append(f"Cost:         {policy_cfg.get('cost_bps', 15.0)} bps round-trip\n")
    lines.append("```\n")

    # ── Entry Mode Baseline ──────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Entry Mode Baseline (No Penalty)\n")
    if not entry_summary_df.empty:
        lines.append(entry_summary_df.to_markdown(index=False))
        lines.append("\n\n")
    else:
        lines.append("⚠️ Entry mode summary not available\n")

    # ── Penalty Stress Test ──────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Queue / Priority Penalty Stress Test\n")
    lines.append("> Each cell shows the Sharpe ratio under a given combination of "
                 "fill-probability haircut and fill-price penalty.\n\n")
    if not penalty_df.empty:
        lines.append(penalty_df.to_markdown(index=False))
        lines.append("\n\n")
    else:
        lines.append("⚠️ Penalty grid not available\n")

    # ── Go / No-Go Assessment ────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Go / No-Go Assessment\n")

    gates = go_no_go_cfg or {}
    min_stressed_sharpe = gates.get("min_stressed_sharpe", 3.0)
    min_fill_rate = gates.get("min_fill_rate", 0.50)
    min_fold_prof = gates.get("min_fold_profitability", 0.70)

    lines.append(f"| Gate | Threshold | Status |\n")
    lines.append(f"|:-----|:----------|:-------|\n")

    # Check stressed Sharpe (80% haircut + 2bps penalty)
    stressed_rows = penalty_df[
        (penalty_df["fill_probability_haircut"] == 0.8) &
        (penalty_df["fill_price_penalty_bps"] == 2.0)
    ] if not penalty_df.empty else pd.DataFrame()

    if not stressed_rows.empty:
        best_stressed = stressed_rows.loc[stressed_rows["sharpe"].idxmax()]
        stressed_sharpe = best_stressed["sharpe"]
        sharpe_pass = stressed_sharpe >= min_stressed_sharpe
        lines.append(
            f"| Stressed Sharpe (80% fill, +2bps) | ≥ {min_stressed_sharpe:.1f} | "
            f"{'✅' if sharpe_pass else '❌'} {stressed_sharpe:.2f} |\n"
        )
    else:
        lines.append(f"| Stressed Sharpe | ≥ {min_stressed_sharpe:.1f} | ⚠️ N/A |\n")

    if not entry_summary_df.empty and "fill_rate" in entry_summary_df.columns:
        best_fill = entry_summary_df["fill_rate"].max()
        fill_pass = best_fill >= min_fill_rate
        lines.append(
            f"| Fill Rate | ≥ {min_fill_rate:.0%} | "
            f"{'✅' if fill_pass else '❌'} {best_fill:.0%} |\n"
        )

    if not entry_summary_df.empty and "fold_profitability" in entry_summary_df.columns:
        best_fp = entry_summary_df["fold_profitability"].max()
        fp_pass = best_fp >= min_fold_prof
        lines.append(
            f"| Fold Profitability | ≥ {min_fold_prof:.0%} | "
            f"{'✅' if fp_pass else '❌'} {best_fp:.0%} |\n"
        )

    # Overall verdict
    all_pass = True
    if not stressed_rows.empty:
        all_pass = all_pass and (stressed_rows["sharpe"].max() >= min_stressed_sharpe)
    if not entry_summary_df.empty and "fill_rate" in entry_summary_df.columns:
        all_pass = all_pass and (entry_summary_df["fill_rate"].max() >= min_fill_rate)
    if not entry_summary_df.empty and "fold_profitability" in entry_summary_df.columns:
        all_pass = all_pass and (entry_summary_df["fold_profitability"].max() >= min_fold_prof)

    lines.append("\n")
    if all_pass:
        lines.append("**✅ GO — proceed to paper deployment**\n")
    else:
        lines.append("**❌ NO-GO — edge does not survive stress test at required thresholds**\n")

    # ── Figures ──────────────────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Figures\n")
    fig_dir = report_dir / "figures"
    if fig_dir.exists():
        for fig_path in sorted(fig_dir.glob("*.png")):
            lines.append(f"![{fig_path.stem}]({fig_path.name})\n")

    # ── Recommended Config ───────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Recommended Paper-Deployment Configuration\n")
    if all_pass and not entry_summary_df.empty:
        best = entry_summary_df.loc[entry_summary_df["sharpe"].idxmax()]
        lines.append("```yaml\n")
        lines.append(f"policy:\n")
        lines.append(f"  training_mode: {policy_cfg.get('training_mode', 'pooled_train_sol_deploy')}\n")
        lines.append(f"  threshold: {policy_cfg.get('threshold', 0.55)}\n")
        lines.append(f"  sep_gap: {policy_cfg.get('sep_gap', 3)}\n")
        lines.append(f"  regime_gate: {policy_cfg.get('regime_gate', 'NOT_rebound')}\n")
        lines.append(f"  entry_mode: {best['entry_mode']}\n")
        lines.append(f"  cost_bps: {policy_cfg.get('cost_bps', 15.0)}\n")
        lines.append("```\n")
    else:
        lines.append("No deployment recommended at this time.\n")

    out_path = report_dir / "summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
