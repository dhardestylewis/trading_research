"""exp003 report builder.

Generates summary.md with three top-level verdicts:
  1. Statistical signal — corrected metrics still positive?
  2. Policy extractability — sparse policy with sufficient trades and fold stability?
  3. Execution survivability — 1-bar delay near break-even?

Plus 5 go/no-go gates.
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

log = get_logger("exp003_report")


# ═══════════════════════════════════════════════════════════════════
#  Plots
# ═══════════════════════════════════════════════════════════════════

def plot_fill_comparison(fill_df: pd.DataFrame, fig_dir: Path):
    """Bar chart of Sharpe by fill type at threshold 0.55."""
    fig_dir = ensure_dir(fig_dir)
    t55 = fill_df[fill_df["threshold"] == 0.55] if "threshold" in fill_df.columns else fill_df
    if t55.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in t55["sharpe"]]
    ax.barh(t55["fill_type"], t55["sharpe"], color=colors)
    ax.set_xlabel("Sharpe Ratio")
    ax.set_title("SOL-Only LightGBM Sharpe by Fill Type (τ=0.55)")
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fill_comparison.png")
    plt.close(fig)


def plot_horizon_delay_heatmap(horizon_df: pd.DataFrame, fig_dir: Path):
    """Heatmap of Sharpe across horizon × delay."""
    fig_dir = ensure_dir(fig_dir)
    t55 = horizon_df[horizon_df["threshold"] == 0.55] if "threshold" in horizon_df.columns else horizon_df
    if t55.empty:
        return

    piv = t55.pivot_table(index="horizon_h", columns="delay_bars", values="sharpe", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([f"{h}h" for h in piv.index])
    ax.set_xlabel("Delay (bars)")
    ax.set_ylabel("Horizon")
    ax.set_title("SOL-Only LightGBM Sharpe — Horizon × Delay")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=11,
                    color="white" if abs(piv.values[i, j]) > 0.5 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(fig_dir / "horizon_delay_heatmap.png")
    plt.close(fig)


def plot_cost_sensitivity(cost_df: pd.DataFrame, fig_dir: Path):
    """Line chart of Sharpe vs cost for each delay."""
    fig_dir = ensure_dir(fig_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    for delay, grp in cost_df.groupby("delay_bars"):
        grp_s = grp.sort_values("cost_bps")
        ax.plot(grp_s["cost_bps"], grp_s["sharpe"], marker="o", label=f"delay={delay}")
    ax.set_xlabel("One-Way Cost (bps)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("SOL-Only Cost Sensitivity")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "cost_sensitivity.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Go/No-Go Gates
# ═══════════════════════════════════════════════════════════════════

def evaluate_gates(
    corrected_regime_df: pd.DataFrame,
    delay_grid_df: pd.DataFrame,
    sparse_df: pd.DataFrame,
    cfg: dict,
) -> dict[str, tuple[bool, str]]:
    """Evaluate 5 go/no-go gates.

    Returns dict mapping gate name → (passed, detail_string).
    """
    criteria = cfg.get("success_criteria", {})
    gates: dict[str, tuple[bool, str]] = {}

    # Gate 1: SOL-only corrected metrics positive
    if not corrected_regime_df.empty:
        lgb = corrected_regime_df[corrected_regime_df["model_name"] == "lightgbm"]
        if not lgb.empty:
            mean_sharpe = lgb["conditional_sharpe"].mean()
            gates["Corrected Signal Positive"] = (
                mean_sharpe > 0,
                f"mean conditional Sharpe={mean_sharpe:.3f}"
            )

    # Gate 2: 1-bar delay near break-even
    if not delay_grid_df.empty:
        d1 = delay_grid_df[
            (delay_grid_df["delay_bars"] == 1) &
            (delay_grid_df.get("cost_regime", pd.Series(["base"] * len(delay_grid_df))) == "base")
        ]
        # Fall back to model_name filter if present
        if "model_name" in d1.columns:
            d1 = d1[d1["model_name"] == "lightgbm"]
        if not d1.empty:
            delay_sharpe = d1.iloc[0]["sharpe"]
            min_sharpe = criteria.get("one_bar_delay_sharpe_min", -0.5)
            gates["1-Bar Delay Survivable"] = (
                delay_sharpe >= min_sharpe,
                f"1-bar delay Sharpe={delay_sharpe:.3f} (min={min_sharpe})"
            )

    # Gate 3: Trade sufficiency (from delay=0 baseline)
    if not delay_grid_df.empty:
        d0 = delay_grid_df[delay_grid_df["delay_bars"] == 0]
        if "model_name" in d0.columns:
            d0 = d0[d0["model_name"] == "lightgbm"]
        if not d0.empty:
            tc = int(d0.iloc[0].get("trade_count", 0))
            min_tc = criteria.get("min_trade_count", 100)
            gates["Trade Sufficiency"] = (
                tc >= min_tc,
                f"trade count={tc} (min={min_tc})"
            )

    # Gate 4: Fold profitability ≥ 65%
    # Prefer sparse policy data; fall back to delay grid
    best_fold_prof = 0.0
    source = "none"
    if not sparse_df.empty and "fold_profitability" in sparse_df.columns:
        best_sparse = sparse_df[sparse_df["delay_bars"] == 0]
        if not best_sparse.empty:
            best_fold_prof = best_sparse["fold_profitability"].max()
            source = "sparse_policy"
    if best_fold_prof == 0 and not delay_grid_df.empty:
        d0 = delay_grid_df[delay_grid_df["delay_bars"] == 0]
        if "fold_profitability" in d0.columns:
            best_fold_prof = d0["fold_profitability"].max() if not d0.empty else 0
            source = "delay_grid"
    min_fp = criteria.get("fold_profitability_min", 0.65)
    gates["Fold Profitability"] = (
        best_fold_prof >= min_fp,
        f"best={best_fold_prof:.0%} from {source} (min={min_fp:.0%})"
    )

    # Gate 5: Non-trivial tail (≥50 trades)
    min_tail = criteria.get("min_tail_trades", 50)
    if not sparse_df.empty:
        best_trades = sparse_df[sparse_df["delay_bars"] == 0]["trade_count"].max() if not sparse_df.empty else 0
        gates["Non-Trivial Tail"] = (
            best_trades >= min_tail,
            f"best sparse trade count={best_trades} (min={min_tail})"
        )

    return gates


# ═══════════════════════════════════════════════════════════════════
#  Summary Report
# ═══════════════════════════════════════════════════════════════════

def build_summary(
    report_dir: Path,
    corrected_regime_df: pd.DataFrame,
    corrected_fold_df: pd.DataFrame,
    fill_grid_df: pd.DataFrame,
    delay_grid_df: pd.DataFrame,
    horizon_df: pd.DataFrame,
    sparse_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    cfg: dict,
) -> Path:
    """Build exp003 summary.md."""
    lines: list[str] = []
    lines.append("# Experiment Report: crypto_1h_exp003\n")
    lines.append("## Objective\n")
    lines.append("**Determine whether the SOL-only LightGBM signal at 1h remains positive "
                 "under corrected conditioning, realistic timing, and policy sparsification.**\n")

    # ── Three Top-Level Verdicts ─────────────────────────────────
    lines.append("---\n")
    lines.append("## Top-Level Verdicts\n")

    # Verdict 1: Statistical Signal
    lines.append("### 1. Statistical Signal\n")
    if not corrected_regime_df.empty:
        lgb = corrected_regime_df[corrected_regime_df["model_name"] == "lightgbm"]
        if not lgb.empty:
            mean_s = lgb["conditional_sharpe"].mean()
            verdict = "✅ POSITIVE" if mean_s > 0 else "❌ NEGATIVE"
            lines.append(f"**{verdict}** — Mean corrected conditional Sharpe = {mean_s:.3f}\n")
        else:
            lines.append("⚠️ No LightGBM data in corrected regime metrics\n")
    else:
        lines.append("⚠️ Corrected regime metrics not available\n")

    # Verdict 2: Policy Extractability
    lines.append("### 2. Policy Extractability\n")
    if not sparse_df.empty:
        d0 = sparse_df[sparse_df["delay_bars"] == 0]
        best = d0.loc[d0["sharpe"].idxmax()] if not d0.empty else None
        if best is not None:
            verdict = "✅ EXTRACTABLE" if best["trade_count"] >= 100 and best["fold_profitability"] >= 0.65 else "⚠️ MARGINAL"
            lines.append(f"**{verdict}** — Best sparse policy: {best['policy']}, "
                        f"Sharpe={best['sharpe']:.2f}, trades={int(best['trade_count'])}, "
                        f"fold prof={best['fold_profitability']:.0%}\n")
        else:
            lines.append("⚠️ No delay=0 sparse policy results\n")
    else:
        lines.append("⚠️ Sparse policy metrics not available\n")

    # Verdict 3: Execution Survivability
    lines.append("### 3. Execution Survivability\n")
    if not fill_grid_df.empty:
        t55 = fill_grid_df[fill_grid_df["threshold"] == 0.55] if "threshold" in fill_grid_df.columns else fill_grid_df
        if not t55.empty:
            lines.append("| Fill Type | Sharpe | Trades | Hit Rate | Fold Prof |\n")
            lines.append("|---|---|---|---|---|\n")
            for _, r in t55.iterrows():
                lines.append(f"| {r['fill_type']} | {r['sharpe']:.2f} | {int(r['trade_count'])} "
                            f"| {r['hit_rate']:.1%} | {r['fold_profitability']:.0%} |\n")
            lines.append("\n")
    else:
        lines.append("⚠️ Fill simulation results not available\n")

    # ── Branch Results ───────────────────────────────────────────
    lines.append("---\n")
    lines.append("## Branch 1 — Corrected Diagnostics\n")

    lines.append("### Corrected Regime Metrics (Active Trades Only)\n")
    if not corrected_regime_df.empty:
        lines.append(corrected_regime_df.to_markdown(index=False))
        lines.append("\n\n")

    lines.append("### Corrected Fold Attribution\n")
    if not corrected_fold_df.empty:
        # Show summary stats
        for model, mg in corrected_fold_df.groupby("model_name"):
            lines.append(f"**{model}**: {len(mg)} folds, "
                        f"mean net PnL={mg['fold_net_pnl'].mean():.6f}, "
                        f"fold profitability={( mg['fold_net_pnl'] > 0).mean():.0%}\n")
    lines.append("\n")

    lines.append("---\n")
    lines.append("## Branch 2 — Execution Timing Audit\n")

    lines.append("### Fill Simulation Grid\n")
    if not fill_grid_df.empty:
        lines.append(fill_grid_df.to_markdown(index=False))
        lines.append("\n\n")

    lines.append("### SOL-Only Delay × Cost Grid\n")
    if not delay_grid_df.empty:
        lines.append(delay_grid_df.to_markdown(index=False))
        lines.append("\n\n")

    lines.append("---\n")
    lines.append("## Branch 3 — Horizon Extension\n")
    if not horizon_df.empty:
        lines.append(horizon_df.to_markdown(index=False))
        lines.append("\n\n")

    lines.append("---\n")
    lines.append("## Branch 4 — Sparse Event Policies\n")
    if not sparse_df.empty:
        lines.append(sparse_df.sort_values("sharpe", ascending=False).to_markdown(index=False))
        lines.append("\n\n")

    lines.append("---\n")
    lines.append("## Branch 5 — Cost Sensitivity\n")
    if not cost_df.empty:
        lines.append(cost_df.to_markdown(index=False))
        lines.append("\n\n")

    # ── Figures ──────────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## Figures\n")
    fig_dir = report_dir / "figures"
    if fig_dir.exists():
        for fig in sorted(fig_dir.glob("*.png")):
            lines.append(f"![{fig.stem}]({fig.name})\n")

    # ── Go/No-Go ─────────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## Go / No-Go Assessment\n")
    gates = evaluate_gates(corrected_regime_df, delay_grid_df, sparse_df, cfg)
    for gate_name, (passed, detail) in gates.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        lines.append(f"- **{gate_name}**: {status} — {detail}\n")
    lines.append("\n")

    # Recommendation
    all_pass = all(p for p, _ in gates.values())
    delay_ok = gates.get("1-Bar Delay Survivable", (False, ""))[0]
    fold_ok = gates.get("Fold Profitability", (False, ""))[0]
    signal_ok = gates.get("Corrected Signal Positive", (False, ""))[0]

    if all_pass:
        rec = "✅ Proceed to SOL-only paper trading."
    elif signal_ok and fold_ok and not delay_ok:
        rec = "⚠️ Signal and policy are valid, but execution timing needs work. Consider longer horizons or maker execution."
    elif signal_ok and not fold_ok:
        rec = "❌ Signal exists but policy extraction fails stability tests. Redesign policy."
    else:
        rec = "❌ No-go. The 1h SOL signal is statistically interesting but operationally too fragile."

    lines.append(f"\n### Recommendation: **{rec}**\n")

    out_path = report_dir / "summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
