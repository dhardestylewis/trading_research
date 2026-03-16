"""exp009 report generator — Dilution Study.

Produces:
  - Dilution matrix heatmap (training config × deploy asset → net bps)
  - SOL preservation comparison
  - Family-pooled per-asset results
  - Summary markdown report
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

log = get_logger("exp009_report")


# ── Plot generators ─────────────────────────────────────────


def plot_dilution_comparison(dilution_cmp: pd.DataFrame, fig_dir: Path) -> Path | None:
    """Bar chart comparing SOL net bps across training pools."""
    if dilution_cmp is None or dilution_cmp.empty:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    pools = dilution_cmp["pool_name"].values
    net_bps = dilution_cmp["mean_net_bps"].values
    trades = dilution_cmp["trade_count"].values

    x = np.arange(len(pools))
    colors = ["green" if b > 0 else "red" for b in net_bps]

    ax1.bar(x, net_bps, color=colors, alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(pools)
    ax1.set_ylabel("Mean Net Return (bps)")
    ax1.set_title("SOL Edge by Training Pool")
    ax1.axhline(y=0, color="black", linewidth=0.5)
    for i, v in enumerate(net_bps):
        ax1.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=10)

    ax2.bar(x, trades, color="steelblue", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(pools)
    ax2.set_ylabel("Trade Count")
    ax2.set_title("SOL Trade Count by Training Pool")

    plt.tight_layout()
    out = fig_dir / "dilution_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_family_per_asset(family_result: pd.DataFrame, fig_dir: Path) -> Path | None:
    """Bar chart of per-asset net bps from family-pooled training."""
    if family_result is None or family_result.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))

    assets = family_result["asset"].values
    net_bps = family_result["mean_net_bps"].values
    qualifies = family_result.get("qualifies", pd.Series([False] * len(family_result)))
    colors = ["green" if q else "gray" for q in qualifies]

    x = np.arange(len(assets))
    ax.bar(x, net_bps, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(assets, rotation=45, ha="right")
    ax.set_ylabel("Mean Net Return (bps)")
    ax.set_title("Family-Pooled Training — Per-Asset Edge")
    ax.axhline(y=0, color="black", linewidth=0.5)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="green", alpha=0.8, label="Qualifies"),
        Patch(color="gray", alpha=0.8, label="Does not qualify"),
    ])

    plt.tight_layout()
    out = fig_dir / "family_per_asset.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_all_plots(
    fig_dir: Path,
    *,
    dilution_cmp: pd.DataFrame | None = None,
    family_result: pd.DataFrame | None = None,
) -> list[Path]:
    """Generate all exp009 plots."""
    fig_dir = ensure_dir(fig_dir)
    paths: list[Path] = []

    p = plot_dilution_comparison(dilution_cmp, fig_dir)
    if p:
        paths.append(p)

    p = plot_family_per_asset(family_result, fig_dir)
    if p:
        paths.append(p)

    log.info("Generated %d plots", len(paths))
    return paths


# ── Report builder ──────────────────────────────────────────


def _go_no_go(
    dilution_cmp: pd.DataFrame | None,
    family_result: pd.DataFrame | None,
    gates: dict,
) -> list[str]:
    """Evaluate go/no-go gates."""
    lines: list[str] = []

    # SOL preservation check
    if dilution_cmp is not None and not dilution_cmp.empty:
        orig = dilution_cmp[dilution_cmp["pool_name"] == "original_3"]
        full = dilution_cmp[dilution_cmp["pool_name"] == "full_8"]

        if not orig.empty and not full.empty:
            orig_bps = orig["mean_net_bps"].values[0]
            full_bps = full["mean_net_bps"].values[0]
            degradation = orig_bps - full_bps
            gate = gates.get("max_sol_degradation_bps", 5.0)
            status = "✅ PASS" if degradation <= gate else "❌ FAIL"
            lines.append(f"| SOL preservation | Degradation ≤ {gate} bps | {degradation:.1f} bps | {status} |")

        if not full.empty:
            sol_bps = full["mean_net_bps"].values[0]
            gate = gates.get("min_sol_net_bps", 0.0)
            status = "✅ PASS" if sol_bps >= gate else "❌ FAIL"
            lines.append(f"| SOL positivity | Net bps ≥ {gate} | {sol_bps:.1f} bps | {status} |")

    # Family pool check
    if family_result is not None and not family_result.empty:
        n_deployable = family_result.get("qualifies", pd.Series()).sum() if "qualifies" in family_result.columns else 0
        gate = gates.get("min_family_deployable_assets", 2)
        status = "✅ PASS" if n_deployable >= gate else "❌ FAIL"
        lines.append(f"| Family deployable | ≥ {gate} assets | {n_deployable} | {status} |")

    return lines


def build_exp009_summary(
    report_dir: Path,
    *,
    dilution_cmp: pd.DataFrame | None = None,
    family_result: pd.DataFrame | None = None,
    backbone_result: pd.DataFrame | None = None,
    cfg: dict | None = None,
    go_no_go: dict | None = None,
) -> Path:
    """Build the summary markdown report."""
    report_dir = Path(report_dir)
    ensure_dir(report_dir)
    cfg = cfg or {}
    go_no_go = go_no_go or {}

    lines: list[str] = []
    lines.append("# exp009 — Dilution Study Report")
    lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
    lines.append("## Objective\n")
    lines.append("Can we preserve SOL edge while adding transportable assets,")
    lines.append("or are we trading one away for the others?\n")

    # ── Branch A+B: Dilution comparison ─────────────────────
    lines.append("---\n## Branches A+B: Pooled Dilution Comparison\n")
    if dilution_cmp is not None and not dilution_cmp.empty:
        lines.append("| Pool | Asset | Trades | Sharpe | Net (bps) | Hit Rate | Fold Prof |")
        lines.append("|------|-------|--------|--------|-----------|----------|-----------|")
        for _, row in dilution_cmp.iterrows():
            lines.append(
                f"| {row['pool_name']} | {row['asset']} | {row['trade_count']:.0f} | "
                f"{row.get('sharpe', 0):.2f} | {row['mean_net_bps']:.1f} | "
                f"{row.get('mean_hit_rate', 0):.2%} | {row.get('fold_profitability', 0):.2%} |"
            )
        lines.append("\n![Dilution comparison](figures/dilution_comparison.png)\n")

        # Interpretation
        orig = dilution_cmp[dilution_cmp["pool_name"] == "original_3"]
        full = dilution_cmp[dilution_cmp["pool_name"] == "full_8"]
        if not orig.empty and not full.empty:
            delta = full["mean_net_bps"].values[0] - orig["mean_net_bps"].values[0]
            direction = "improved" if delta > 0 else "degraded"
            lines.append(f"> SOL edge {direction} by {abs(delta):.1f} bps when moving "
                         f"from 3-asset to 8-asset pool.\n")
    else:
        lines.append("*No dilution comparison data.*\n")

    # ── Branch C: Family-pooled ─────────────────────────────
    lines.append("---\n## Branch C: Family-Pooled Training\n")
    if family_result is not None and not family_result.empty:
        lines.append("| Asset | Trades | Sharpe | Net (bps) | Hit Rate | Fold Prof | Qualifies |")
        lines.append("|-------|--------|--------|-----------|----------|-----------|-----------|")
        for _, row in family_result.iterrows():
            q = "✅" if row.get("qualifies", False) else "❌"
            lines.append(
                f"| {row['asset']} | {row['trade_count']:.0f} | "
                f"{row.get('sharpe', 0):.2f} | {row['mean_net_bps']:.1f} | "
                f"{row.get('mean_hit_rate', 0):.2%} | {row.get('fold_profitability', 0):.2%} | {q} |"
            )
        lines.append("\n![Family per-asset](figures/family_per_asset.png)\n")
    else:
        lines.append("*No family-pooled data.*\n")

    # ── Branch D: Backbone + heads ──────────────────────────
    lines.append("---\n## Branch D: Per-Asset Heads on Shared Backbone\n")
    if backbone_result is not None and not backbone_result.empty:
        lines.append(f"Status: {backbone_result['status'].values[0]}\n")
        lines.append(f"Note: {backbone_result['note'].values[0]}\n")
    else:
        lines.append("*Not implemented — requires architecture change.*\n")

    # ── Go / No-Go ──────────────────────────────────────────
    lines.append("---\n## Go / No-Go Assessment\n")
    gate_lines = _go_no_go(dilution_cmp, family_result, go_no_go)
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
