"""exp006 report builder.

Generates summary.md for the paper-trade validation study.

Report structure:
  1. Metric Dictionary (operational definitions)
  2. Per-lane summary table (primary + shadow)
  3. Execution quality scorecard vs go/no-go thresholds
  4. Plots: slippage, fill rate by hour, adverse selection, shortfall
  5. Go/no-go verdict for live capital advancement
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

log = get_logger("exp006_report")


# ═══════════════════════════════════════════════════════════════════
#  Plots
# ═══════════════════════════════════════════════════════════════════

def plot_slippage_distribution(log_df: pd.DataFrame, fig_dir: Path):
    """Histogram of realized slippage vs simulated in bps."""
    fig_dir = ensure_dir(fig_dir)
    filled = log_df[log_df["cancel_status"].isin(["filled", "partial_fill"])]
    if filled.empty or "simulated_fill_price" not in filled.columns:
        return

    slip = (
        (filled["fill_price"] - filled["simulated_fill_price"])
        / filled["simulated_fill_price"] * 10_000
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(slip.dropna(), bins=30, color="#3498db", alpha=0.7, edgecolor="white")
    ax.axvline(slip.mean(), color="#e74c3c", linewidth=2, linestyle="--",
               label=f"Mean = {slip.mean():.1f} bps")
    ax.set_xlabel("Slippage (bps)")
    ax.set_ylabel("Count")
    ax.set_title("Realized Slippage vs Simulated Fill Price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "slippage_distribution.png", dpi=150)
    plt.close(fig)


def plot_fill_rate_by_hour(log_df: pd.DataFrame, fig_dir: Path):
    """Fill rate broken down by hour of day."""
    fig_dir = ensure_dir(fig_dir)
    df = log_df.copy()
    if "signal_timestamp" not in df.columns:
        return
    df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"])
    df["hour"] = df["signal_timestamp"].dt.hour
    df["is_filled"] = df["cancel_status"].isin(["filled", "partial_fill"])

    hourly = df.groupby("hour").agg(
        fill_rate=("is_filled", "mean"),
        count=("is_filled", "count"),
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    bars = ax1.bar(hourly["hour"], hourly["fill_rate"], color="#27ae60", alpha=0.7,
                   label="Fill Rate")
    ax1.set_xlabel("Hour of Day (UTC)")
    ax1.set_ylabel("Fill Rate", color="#27ae60")
    ax1.set_ylim(0, 1.1)
    ax1.tick_params(axis="y", labelcolor="#27ae60")

    ax2 = ax1.twinx()
    ax2.plot(hourly["hour"], hourly["count"], color="#e67e22", linewidth=2,
             marker="o", label="Signal Count")
    ax2.set_ylabel("Signal Count", color="#e67e22")
    ax2.tick_params(axis="y", labelcolor="#e67e22")

    ax1.set_title("Fill Rate and Signal Count by Hour")
    fig.tight_layout()
    fig.savefig(fig_dir / "fill_rate_by_hour.png", dpi=150)
    plt.close(fig)


def plot_adverse_selection(log_df: pd.DataFrame, fig_dir: Path):
    """Histogram of adverse selection in bps."""
    fig_dir = ensure_dir(fig_dir)
    filled = log_df[log_df["cancel_status"].isin(["filled", "partial_fill"])]
    if filled.empty:
        return

    valid = filled.dropna(subset=["fill_price", "midprice_after_15m"])
    if valid.empty:
        return

    adverse = -(valid["midprice_after_15m"] - valid["fill_price"]) / valid["fill_price"] * 10_000

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c" if v > 0 else "#27ae60" for v in adverse]
    ax.hist(adverse.dropna(), bins=30, color="#9b59b6", alpha=0.7, edgecolor="white")
    ax.axvline(adverse.mean(), color="#e74c3c", linewidth=2, linestyle="--",
               label=f"Mean = {adverse.mean():.1f} bps")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Adverse Selection (bps, positive = adverse)")
    ax.set_ylabel("Count")
    ax.set_title("Post-Fill Adverse Selection (15min window)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "adverse_selection.png", dpi=150)
    plt.close(fig)


def plot_shortfall_scatter(log_df: pd.DataFrame, fig_dir: Path):
    """Scatter: shortfall vs simulated vs realized PnL."""
    fig_dir = ensure_dir(fig_dir)
    filled = log_df[log_df["cancel_status"].isin(["filled", "partial_fill"])]
    if filled.empty:
        return

    valid = filled.dropna(subset=["realized_shortfall_vs_simulated", "realized_pnl_at_horizon"])
    if valid.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        valid["realized_shortfall_vs_simulated"],
        valid["realized_pnl_at_horizon"] * 10_000,
        c=valid["realized_pnl_at_horizon"] * 10_000,
        cmap="RdYlGn", alpha=0.7, edgecolors="white", linewidth=0.5,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Realized Shortfall vs Simulated (bps)")
    ax.set_ylabel("Realized PnL (bps)")
    ax.set_title("Execution Shortfall vs Trade PnL")
    fig.colorbar(scatter, label="PnL (bps)")
    fig.tight_layout()
    fig.savefig(fig_dir / "shortfall_scatter.png", dpi=150)
    plt.close(fig)


def plot_lane_comparison(lane_summary: pd.DataFrame, fig_dir: Path):
    """Side-by-side comparison of primary vs shadow lanes."""
    fig_dir = ensure_dir(fig_dir)
    if lane_summary.empty:
        return

    metrics_to_plot = [
        ("realized_fill_rate", "Fill Rate", "#27ae60"),
        ("mean_slippage_bps", "Mean Slippage (bps)", "#e74c3c"),
        ("mean_adverse_selection_bps", "Adverse Selection (bps)", "#9b59b6"),
        ("mean_return_per_trade_bps", "Return/Trade (bps)", "#3498db"),
    ]

    available = [(m, t, c) for m, t, c in metrics_to_plot if m in lane_summary.columns]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, (metric, title, color) in zip(axes, available):
        vals = lane_summary[metric].fillna(0)
        ax.barh(lane_summary["lane"], vals, color=color, alpha=0.7)
        ax.set_xlabel(title)
        ax.set_title(title)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=9)

    fig.suptitle("Lane Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "lane_comparison.png", dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Summary Report
# ═══════════════════════════════════════════════════════════════════

def build_exp006_summary(
    report_dir: Path,
    lane_summary: pd.DataFrame,
    full_log: pd.DataFrame,
    go_no_go_cfg: dict,
    policy_cfg: dict,
    metric_dict: dict,
    cfg: dict,
) -> Path:
    """Build exp006 summary.md.

    Parameters
    ----------
    report_dir : Path
        Output directory for report files.
    lane_summary : pd.DataFrame
        One row per lane with execution quality metrics.
    full_log : pd.DataFrame
        Combined paper-trade log (all lanes).
    go_no_go_cfg : dict
        Go/no-go gate thresholds.
    policy_cfg : dict
        Policy specification.
    metric_dict : dict
        Metric dictionary definitions.
    cfg : dict
        Full experiment config.
    """
    lines: list[str] = []
    lines.append("# Experiment Report: crypto_1h_exp006\n")
    lines.append("## Objective\n")
    lines.append(
        "**Paper-trade validation study.** Bridge simulated fills from exp005 "
        "to live order lifecycle evaluation. This is NOT a model experiment — "
        "the model gates were already passed in exp005.\n"
    )

    # ── Metric Dictionary ────────────────────────────────────────
    lines.append("---\n")
    lines.append("## Metric Dictionary\n")
    lines.append(
        "> Operational definitions for all execution metrics used in this report. "
        "Ambiguity in execution metrics is the primary source of paper-validation "
        "failure in production quant workflows.\n\n"
    )

    if metric_dict:
        lines.append("| Metric | Operational Definition |\n")
        lines.append("|:-------|:-----------------------|\n")
        for metric_name, defn in metric_dict.items():
            clean_defn = " ".join(defn.strip().split())
            lines.append(f"| **{metric_name}** | {clean_defn} |\n")
        lines.append("\n")

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

    # ── Lane Summary ─────────────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Lane Summary (Primary + Shadow)\n")
    if not lane_summary.empty:
        display_cols = [c for c in lane_summary.columns
                        if not c.endswith("_values")]
        lines.append(lane_summary[display_cols].to_markdown(index=False))
        lines.append("\n\n")
    else:
        lines.append("⚠️ No lane summary available\n")

    # ── Per-Signal Log Stats ─────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Per-Signal Log Statistics\n")
    if not full_log.empty:
        total = len(full_log)
        filled_n = int((full_log["cancel_status"].isin(["filled", "partial_fill"])).sum())
        cancelled_n = int((full_log["cancel_status"] == "cancelled").sum())
        lines.append(f"- **Total signals logged:** {total}\n")
        lines.append(f"- **Filled:** {filled_n} ({filled_n/total:.1%})\n")
        lines.append(f"- **Cancelled:** {cancelled_n} ({cancelled_n/total:.1%})\n")

        if "realized_pnl_at_horizon" in full_log.columns:
            filled_pnl = full_log.loc[
                full_log["cancel_status"].isin(["filled", "partial_fill"]),
                "realized_pnl_at_horizon"
            ].dropna()
            if not filled_pnl.empty:
                lines.append(f"- **Median trade return:** {filled_pnl.median()*10000:.1f} bps\n")
                lines.append(f"- **Mean trade return:** {filled_pnl.mean()*10000:.1f} bps\n")

        # Log columns present
        lines.append(f"\n**Log columns:** {len(full_log.columns)} fields per signal\n")
    else:
        lines.append("⚠️ No signal log data\n")

    # ── Go / No-Go Assessment ────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Go / No-Go Assessment (Paper → Live Capital)\n")
    lines.append(
        "> These gates evaluate **execution fidelity**, not model quality. "
        "Model quality gates were passed in exp005.\n\n"
    )

    gates = go_no_go_cfg or {}
    min_fill = gates.get("min_realized_fill_rate", 0.60)
    max_slip = gates.get("max_mean_slippage_bps", 10.0)
    max_adv = gates.get("max_adverse_selection_bps", 15.0)
    max_cancel = gates.get("max_cancel_rate", 0.30)
    min_ret = gates.get("min_return_per_trade_bps", 5.0)

    lines.append("| Gate | Threshold | Status |\n")
    lines.append("|:-----|:----------|:-------|\n")

    all_pass = True

    if not lane_summary.empty:
        # Use primary lane for gates
        primary = lane_summary[lane_summary["lane_type"] == "primary"]
        if primary.empty:
            primary = lane_summary.iloc[[0]]

        row = primary.iloc[0]

        # Fill rate
        fr = row.get("realized_fill_rate", np.nan)
        fr_pass = fr >= min_fill if np.isfinite(fr) else False
        all_pass = all_pass and fr_pass
        lines.append(
            f"| Realized Fill Rate | ≥ {min_fill:.0%} | "
            f"{'✅' if fr_pass else '❌'} {fr:.1%} |\n"
        )

        # Slippage
        sl = row.get("mean_slippage_bps", np.nan)
        sl_pass = abs(sl) <= max_slip if np.isfinite(sl) else False
        all_pass = all_pass and sl_pass
        lines.append(
            f"| Mean Slippage | ≤ {max_slip:.0f} bps | "
            f"{'✅' if sl_pass else '❌'} {sl:.1f} bps |\n"
        )

        # Adverse selection
        ad = row.get("mean_adverse_selection_bps", np.nan)
        ad_pass = ad <= max_adv if np.isfinite(ad) else False
        all_pass = all_pass and ad_pass
        lines.append(
            f"| Adverse Selection (15m) | ≤ {max_adv:.0f} bps | "
            f"{'✅' if ad_pass else '❌'} {ad:.1f} bps |\n"
        )

        # Cancel rate
        cr = row.get("cancel_rate", np.nan)
        cr_pass = cr <= max_cancel if np.isfinite(cr) else False
        all_pass = all_pass and cr_pass
        lines.append(
            f"| Cancel Rate | ≤ {max_cancel:.0%} | "
            f"{'✅' if cr_pass else '❌'} {cr:.1%} |\n"
        )

        # Return per trade
        rt = row.get("mean_return_per_trade_bps", np.nan)
        rt_pass = rt >= min_ret if np.isfinite(rt) else False
        all_pass = all_pass and rt_pass
        lines.append(
            f"| Return per Trade | ≥ {min_ret:.0f} bps | "
            f"{'✅' if rt_pass else '❌'} {rt:.1f} bps |\n"
        )
    else:
        all_pass = False
        lines.append("| All gates | — | ⚠️ No data |\n")

    lines.append("\n")
    if all_pass:
        lines.append("**✅ GO — simulated execution fidelity validated. Advance to live paper.**\n\n")
        lines.append(
            "**⬜ NOT YET — live capital requires realized exchange execution data (exp007).**\n"
        )
    else:
        lines.append(
            "**⚠️ HOLD — execution fidelity requires live paper-trade data to evaluate. "
            "Simulated scaffold is ready; feed live order logs to validate.**\n"
        )

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
    lines.append("1. Connect live paper-trade order logs to `PaperTradeLogger`\n")
    lines.append("2. Re-run this report on 7+ days of live data\n")
    lines.append("3. Compare realized vs simulated metrics per lane\n")
    lines.append("4. If all gates pass on live data → GO to live capital\n")
    lines.append("5. Judge early results by: realized fill quality, signed slippage, "
                 "median trade expectancy, weekly profitability, stability of execution error\n")

    out_path = report_dir / "summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
