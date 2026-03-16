"""exp011 report generator — Gross-Move Atlas.

PROGRAM RULE: the gross bps distribution must be the FIRST table.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.logging import get_logger

log = get_logger("exp011_report")


def generate_atlas_plots(
    fig_dir: Path,
    atlas: pd.DataFrame,
):
    """Generate atlas visualisation plots."""
    if atlas.empty:
        return

    # ── 1. Gross bps heatmap: asset × regime (market entry, median) ──
    try:
        market = atlas[atlas["entry"] == "market_next_open"].copy()
        if not market.empty:
            pivot = market.pivot_table(
                index="asset", columns="regime",
                values="gross_bps_abs_median", aggfunc="first",
            )
            fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.4)))
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=9)
            plt.colorbar(im, ax=ax, label="Median |gross bps|")
            ax.set_title("Gross Move Magnitude: Asset × Regime (Market Entry)")
            fig.tight_layout()
            fig.savefig(fig_dir / "gross_bps_heatmap.png", dpi=120)
            plt.close(fig)
    except Exception as e:
        log.warning("Heatmap plot failed: %s", e)

    # ── 2. Exceedance bar chart (top 20 viable cells) ────────────
    try:
        viable = atlas[atlas.get("viable", False)].head(20) if "viable" in atlas.columns else atlas.head(20)
        if not viable.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            labels = viable.apply(lambda r: f"{r['asset']}\n{r['regime']}\n{r['entry']}", axis=1)
            x = range(len(viable))

            for thresh_col, color, label in [
                ("frac_abs_gt_20bps", "#4CAF50", ">20 bps"),
                ("frac_abs_gt_30bps", "#FF9800", ">30 bps"),
                ("frac_abs_gt_50bps", "#F44336", ">50 bps"),
            ]:
                if thresh_col in viable.columns:
                    ax.bar(x, viable[thresh_col] * 100, alpha=0.6, label=label, color=color)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=7, ha="center")
            ax.set_ylabel("% of trades exceeding threshold")
            ax.set_title("Gross Move Exceedance — Top Cells")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / "exceedance_chart.png", dpi=120)
            plt.close(fig)
    except Exception as e:
        log.warning("Exceedance plot failed: %s", e)

    # ── 3. Per-asset gross bps distribution box plot ──────────────
    try:
        market = atlas[atlas["entry"] == "market_next_open"].copy()
        if not market.empty:
            cols = ["gross_bps_median", "gross_bps_p75", "gross_bps_p90"]
            avail = [c for c in cols if c in market.columns]
            if avail:
                asset_order = market.groupby("asset")["gross_bps_p75"].max().sort_values(ascending=False).index
                fig, ax = plt.subplots(figsize=(12, 6))
                for i, col in enumerate(avail):
                    vals = [market[market["asset"] == a][col].values for a in asset_order]
                    positions = np.arange(len(asset_order)) + i * 0.25 - 0.25
                    bp = ax.boxplot(
                        vals, positions=positions, widths=0.2,
                        patch_artist=True, showfliers=False,
                    )
                    color = ["#2196F3", "#4CAF50", "#FF5722"][i]
                    for patch in bp["boxes"]:
                        patch.set_facecolor(color)
                        patch.set_alpha(0.6)

                ax.set_xticks(range(len(asset_order)))
                ax.set_xticklabels(asset_order, rotation=45, ha="right", fontsize=9)
                ax.set_ylabel("Gross bps")
                ax.set_title("Gross Move Distribution by Asset (Market Entry, All Regimes)")
                ax.axhline(30, color="red", linestyle="--", alpha=0.5, label="30 bps friction")
                ax.legend(avail + ["Friction hurdle"])
                fig.tight_layout()
                fig.savefig(fig_dir / "asset_gross_bps_boxplot.png", dpi=120)
                plt.close(fig)
    except Exception as e:
        log.warning("Box plot failed: %s", e)

    # ── 4. Entry convention comparison ───────────────────────────
    try:
        if not atlas.empty and "entry" in atlas.columns:
            entry_agg = atlas.groupby("entry").agg(
                mean_fill_rate=("fill_rate", "mean"),
                mean_gross_bps_median=("gross_bps_median", "mean"),
                mean_gross_bps_p75=("gross_bps_p75", "mean"),
            )
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].barh(entry_agg.index, entry_agg["mean_fill_rate"] * 100)
            axes[0].set_xlabel("Mean Fill Rate (%)")
            axes[0].set_title("Fill Rate by Entry Convention")

            axes[1].barh(entry_agg.index, entry_agg["mean_gross_bps_p75"])
            axes[1].set_xlabel("Mean P75 Gross bps")
            axes[1].set_title("P75 Gross Move by Entry Convention")
            axes[1].axvline(30, color="red", linestyle="--", alpha=0.5)

            fig.tight_layout()
            fig.savefig(fig_dir / "entry_convention_comparison.png", dpi=120)
            plt.close(fig)
    except Exception as e:
        log.warning("Entry comparison plot failed: %s", e)


def build_exp011_summary(
    report_dir: Path,
    atlas: pd.DataFrame,
    kill_gate_result: tuple[bool, str],
    cfg: dict,
) -> Path:
    """Write the exp011 summary report."""
    passes, reason = kill_gate_result
    friction_bps = cfg.get("friction", {}).get("round_trip_bps", 30.0)

    lines: list[str] = []
    lines.append("# exp011 — Gross-Move Atlas\n")
    lines.append(f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*\n")

    lines.append("## Objective\n")
    lines.append("Magnitude-first economic screening. Find cells where gross move")
    lines.append(f"distribution plausibly clears {friction_bps:.0f} bps friction.\n")
    lines.append("**No ML. Phase 1 only.**\n")

    # ── Table 1: Gross bps distribution (FIRST TABLE — program rule) ──
    lines.append("---")
    lines.append("## Table 1: Gross BPS Distribution Before Costs\n")
    lines.append("> **This is the first table per program rule.** All research")
    lines.append("> must confront the economic hurdle before any modeling narrative.\n")

    if not atlas.empty:
        # Show top 20 ranked cells
        cols = [
            "rank", "asset", "horizon", "regime", "entry",
            "fill_count", "fill_rate",
            "gross_bps_median", "gross_bps_p75", "gross_bps_p90",
            "frac_abs_gt_30bps", "frac_abs_gt_50bps",
            "viable",
        ]
        avail = [c for c in cols if c in atlas.columns]
        top = atlas[avail].head(20)

        lines.append("| " + " | ".join(avail) + " |")
        lines.append("| " + " | ".join(["---"] * len(avail)) + " |")
        for _, row in top.iterrows():
            vals = []
            for c in avail:
                v = row[c]
                if isinstance(v, float):
                    if "rate" in c or "frac" in c:
                        vals.append(f"{v:.1%}")
                    else:
                        vals.append(f"{v:.1f}")
                elif isinstance(v, bool):
                    vals.append("YES" if v else "NO")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    # ── Kill gate ────────────────────────────────────────────────
    lines.append("---")
    lines.append("## Kill Gate Assessment\n")
    emoji = "PASS" if passes else "FAIL"
    lines.append(f"**{emoji} {reason}**\n")

    if not passes:
        lines.append("> WARNING: The atlas did not find enough viable cells.")
        lines.append("> Per program rules: **do not build another model.**")
        lines.append("> Investigate alternative universes, horizons, or entry conventions before proceeding.\n")

    # ── Viable cells detail ──────────────────────────────────────
    if not atlas.empty and "viable" in atlas.columns:
        viable = atlas[atlas["viable"]]
        n_viable = len(viable)
        lines.append("---")
        lines.append(f"## Viable Cells ({n_viable} found)\n")

        if n_viable > 0:
            detail_cols = [
                "rank", "asset", "horizon", "regime", "entry",
                "fill_count", "gross_bps_median", "gross_bps_p75",
                "gross_bps_p90", "adverse_sel_mean_bps", "viable",
            ]
            avail = [c for c in detail_cols if c in viable.columns]
            lines.append("| " + " | ".join(avail) + " |")
            lines.append("| " + " | ".join(["---"] * len(avail)) + " |")
            for _, row in viable.iterrows():
                vals = []
                for c in avail:
                    v = row[c]
                    if isinstance(v, float):
                        vals.append(f"{v:.1f}")
                    elif isinstance(v, bool):
                        vals.append("YES" if v else "NO")
                    else:
                        vals.append(str(v))
                lines.append("| " + " | ".join(vals) + " |")
            lines.append("")
        else:
            lines.append("*No viable cells found.*\n")

    # ── Figures ──────────────────────────────────────────────────
    lines.append("---")
    lines.append("## Figures\n")
    lines.append("![Gross BPS Heatmap](figures/gross_bps_heatmap.png)\n")
    lines.append("![Exceedance Chart](figures/exceedance_chart.png)\n")
    lines.append("![Asset Box Plot](figures/asset_gross_bps_boxplot.png)\n")
    lines.append("![Entry Comparison](figures/entry_convention_comparison.png)\n")

    # ── Next steps ───────────────────────────────────────────────
    lines.append("---")
    lines.append("## Next Steps\n")
    if passes:
        lines.append("Kill gate PASSED. Proceed to **exp012 (Passive Realism)**")
        lines.append("on the identified viable cells.\n")
    else:
        lines.append("Kill gate FAILED. Options:\n")
        lines.append("1. Expand universe to higher-volatility names")
        lines.append("2. Add 15m / 4h horizons")
        lines.append("3. Investigate different event-driven regime definitions")
        lines.append("4. Accept that this market segment may not clear friction\n")

    out = report_dir / "summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written: %s", out)
    return out
