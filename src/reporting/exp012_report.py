"""exp012 report generator — Passive Realism."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.logging import get_logger

log = get_logger("exp012_report")


def generate_passive_plots(fig_dir: Path, results: pd.DataFrame):
    """Generate passive realism visualisation plots."""
    if results.empty:
        return

    # ── 1. Net expectancy heatmap: offset × haircut ──────────────
    try:
        for metric in ["net_bps_mean", "gross_bps_mean"]:
            if metric not in results.columns:
                continue
            agg = results.groupby(["offset_bps", "queue_haircut"])[metric].mean().reset_index()
            pivot = agg.pivot(index="offset_bps", columns="queue_haircut", values=metric)

            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                          vmin=pivot.values.min(), vmax=max(pivot.values.max(), 1))
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{h:.0%}" for h in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{o:.0f}" for o in pivot.index])
            ax.set_xlabel("Queue Haircut")
            ax.set_ylabel("Limit Offset (bps below open)")
            plt.colorbar(im, ax=ax, label=f"Mean {metric.replace('_', ' ')}")
            ax.set_title(f"Passive Entry: {metric.replace('_', ' ').title()}")
            fig.tight_layout()
            fig.savefig(fig_dir / f"passive_{metric}_heatmap.png", dpi=120)
            plt.close(fig)
    except Exception as e:
        log.warning("Heatmap plot failed: %s", e)

    # ── 2. Per-asset adverse selection ───────────────────────────
    try:
        if "adverse_sel_bps" in results.columns:
            zero_haircut = results[results["queue_haircut"] == 0.0]
            if not zero_haircut.empty:
                asset_adv = zero_haircut.groupby("asset")["adverse_sel_bps"].mean().sort_values()
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ["#F44336" if v < 0 else "#4CAF50" for v in asset_adv.values]
                ax.barh(asset_adv.index, asset_adv.values, color=colors)
                ax.set_xlabel("Mean Adverse Selection (bps)")
                ax.set_title("Adverse Selection After Passive Fill (by Asset)")
                ax.axvline(0, color="black", linewidth=0.5)
                fig.tight_layout()
                fig.savefig(fig_dir / "adverse_selection_by_asset.png", dpi=120)
                plt.close(fig)
    except Exception as e:
        log.warning("Adverse selection plot failed: %s", e)

    # ── 3. Fill rate vs offset ───────────────────────────────────
    try:
        if "fill_rate" in results.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            for haircut in sorted(results["queue_haircut"].unique()):
                subset = results[results["queue_haircut"] == haircut]
                agg = subset.groupby("offset_bps")["fill_rate"].mean()
                ax.plot(agg.index, agg.values * 100, marker="o",
                       label=f"Haircut {haircut:.0%}")
            ax.set_xlabel("Limit Offset (bps below open)")
            ax.set_ylabel("Fill Rate (%)")
            ax.set_title("Fill Rate vs Passive Offset")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / "fill_rate_vs_offset.png", dpi=120)
            plt.close(fig)
    except Exception as e:
        log.warning("Fill rate plot failed: %s", e)


def build_exp012_summary(
    report_dir: Path,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    kill_gate_result: tuple[bool, str],
    cfg: dict,
) -> Path:
    """Write the exp012 summary report."""
    passes, reason = kill_gate_result

    lines: list[str] = []
    lines.append("# exp012 — Passive Realism\n")
    lines.append(f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*\n")

    lines.append("## Objective\n")
    lines.append("Test whether passive entry can reduce effective round-trip cost")
    lines.append("enough that a weak but real gross move becomes net-positive.\n")

    # ── Summary table ────────────────────────────────────────────
    lines.append("---")
    lines.append("## Passive Grid Summary\n")
    if not summary.empty:
        lines.append(summary.to_markdown(index=False))
        lines.append("")

    # ── Kill gate ────────────────────────────────────────────────
    lines.append("---")
    lines.append("## Kill Gate\n")
    emoji = "PASS" if passes else "FAIL"
    lines.append(f"**{emoji} {reason}**\n")

    if passes:
        lines.append("> Proceed to **exp013 (Magnitude Model)** on net-positive cells.\n")
    else:
        lines.append("> ⚠️ Passive entry does not rescue economics under conservative assumptions.")
        lines.append("> Consider: different assets, different entry structures, or accept that")
        lines.append("> this market segment cannot be monetized.\n")

    # ── Per-asset detail ─────────────────────────────────────────
    if not results.empty and "asset" in results.columns:
        lines.append("---")
        lines.append("## Per-Asset Results (at most conservative haircut)\n")

        haircut_max = results["queue_haircut"].max()
        conservative = results[results["queue_haircut"] == haircut_max]
        if not conservative.empty:
            detail_cols = [
                "asset", "offset_bps", "fill_rate", "gross_bps_mean",
                "net_bps_mean", "adverse_sel_bps", "hit_rate", "net_positive",
            ]
            avail = [c for c in detail_cols if c in conservative.columns]
            lines.append(conservative[avail].to_markdown(index=False))
            lines.append("")

    # ── Figures ──────────────────────────────────────────────────
    lines.append("---")
    lines.append("## Figures\n")
    lines.append("![Net BPS Heatmap](figures/passive_net_bps_mean_heatmap.png)\n")
    lines.append("![Adverse Selection](figures/adverse_selection_by_asset.png)\n")
    lines.append("![Fill Rate](figures/fill_rate_vs_offset.png)\n")

    out = report_dir / "summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written: %s", out)
    return out
