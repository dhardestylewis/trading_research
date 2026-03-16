"""Diagnostic plots for exp002 reporting.

Four plot groups:
  1. Score ordering (decile returns, hit rate, calibration)
  2. Asset portability (equity curves per asset mode)
  3. Regime sensitivity (Sharpe by regime, fold PnL vs vol)
  4. Fragility surfaces (delay×cost heatmap, threshold×tail heatmap)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.utils.io import ensure_dir
from src.utils.logging import get_logger

log = get_logger("diagnostic_plots")

# ── Style defaults ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ═══════════════════════════════════════════════════════════════════
#  Group 1 — Score ordering
# ═══════════════════════════════════════════════════════════════════

def plot_decile_returns(decile_df: pd.DataFrame, fig_dir: Path):
    """Bar chart of mean net return by score decile for each model."""
    fig_dir = ensure_dir(fig_dir)
    for model, grp in decile_df.groupby("model_name"):
        fig, ax = plt.subplots()
        sorted_g = grp.sort_values("score_decile")
        colors = ["#c0392b" if v < 0 else "#27ae60" for v in sorted_g["mean_net_return"]]
        ax.bar(sorted_g["score_decile"].astype(str), sorted_g["mean_net_return"], color=colors)
        ax.set_xlabel("Score Decile")
        ax.set_ylabel("Mean Net Return")
        ax.set_title(f"Net Return by Score Decile — {model}")
        ax.axhline(0, color="black", linewidth=0.8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"score_decile_returns_{model}.png")
        plt.close(fig)


def plot_decile_hit_rate(decile_df: pd.DataFrame, fig_dir: Path):
    """Bar chart of hit rate by score decile for each model."""
    fig_dir = ensure_dir(fig_dir)
    for model, grp in decile_df.groupby("model_name"):
        fig, ax = plt.subplots()
        sorted_g = grp.sort_values("score_decile")
        ax.bar(sorted_g["score_decile"].astype(str), sorted_g["hit_rate"], color="#3498db")
        ax.set_xlabel("Score Decile")
        ax.set_ylabel("Hit Rate")
        ax.set_title(f"Hit Rate by Score Decile — {model}")
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(fig_dir / f"score_decile_hit_rate_{model}.png")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Group 2 — Asset portability
# ═══════════════════════════════════════════════════════════════════

def plot_asset_equity_curves(
    preds: pd.DataFrame,
    fig_dir: Path,
    model_name: str = "lightgbm",
    threshold: float = 0.55,
    cost_bps: float = 15.0,
):
    """Equity curves for each asset (+ pooled) for a given model."""
    fig_dir = ensure_dir(fig_dir)
    cost = cost_bps / 10_000.0
    mgrp = preds[preds["model_name"] == model_name].sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Per-asset curves
    for asset in sorted(mgrp["asset"].unique()):
        agrp = mgrp[mgrp["asset"] == asset]
        above = agrp[agrp["y_pred_prob"] > threshold]
        if len(above) < 5:
            continue
        net = above["fwd_ret_1h"].values - 2 * cost
        cum = (1 + net).cumprod()
        ax.plot(above["timestamp"].values, cum, label=asset, alpha=0.8)

    # Pooled curve
    above_all = mgrp[mgrp["y_pred_prob"] > threshold]
    if len(above_all) > 5:
        net_all = above_all["fwd_ret_1h"].values - 2 * cost
        cum_all = (1 + net_all).cumprod()
        ax.plot(above_all["timestamp"].values, cum_all, label="Pooled",
                color="black", linewidth=2)

    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(f"Asset Equity Curves — {model_name} (τ={threshold})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "asset_equity_curves.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Group 3 — Regime sensitivity
# ═══════════════════════════════════════════════════════════════════

def plot_regime_sharpe(regime_df: pd.DataFrame, fig_dir: Path):
    """Horizontal bar chart of conditional Sharpe by regime for each model."""
    fig_dir = ensure_dir(fig_dir)
    for model, grp in regime_df.groupby("model_name"):
        grp = grp.sort_values("conditional_sharpe")
        fig, ax = plt.subplots(figsize=(10, max(4, len(grp) * 0.4)))
        colors = ["#c0392b" if v < 0 else "#27ae60" for v in grp["conditional_sharpe"]]
        ax.barh(grp["regime"], grp["conditional_sharpe"], color=colors)
        ax.set_xlabel("Conditional Sharpe")
        ax.set_title(f"Sharpe by Regime — {model}")
        ax.axvline(0, color="black", linewidth=0.8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"regime_sharpe_{model}.png")
        plt.close(fig)


def plot_fold_pnl_vs_vol(fold_desc: pd.DataFrame, fig_dir: Path):
    """Scatter of fold PnL vs fold average realized vol."""
    fig_dir = ensure_dir(fig_dir)
    if "realized_vol_mean" not in fold_desc.columns:
        return

    for model, grp in fold_desc.groupby("model_name"):
        fig, ax = plt.subplots()
        ax.scatter(grp["realized_vol_mean"], grp["fold_net_pnl"], alpha=0.7, s=40)
        ax.set_xlabel("Fold Avg Realized Vol")
        ax.set_ylabel("Fold Net PnL")
        ax.set_title(f"Fold PnL vs Volatility — {model}")
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"fold_pnl_vs_vol_{model}.png")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Group 4 — Fragility surfaces
# ═══════════════════════════════════════════════════════════════════

def plot_delay_cost_heatmap(robust_df: pd.DataFrame, fig_dir: Path):
    """Heatmap of Sharpe across delay × cost regime (taker fill only)."""
    fig_dir = ensure_dir(fig_dir)

    taker = robust_df[robust_df["fill_mode"] == "taker"]
    for model, grp in taker.groupby("model_name"):
        piv = grp.pivot_table(
            index="cost_regime", columns="delay_bars", values="sharpe", aggfunc="mean"
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn", origin="lower")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels(piv.index)
        ax.set_xlabel("Delay (bars)")
        ax.set_ylabel("Cost Regime")
        ax.set_title(f"Sharpe — Delay × Cost — {model}")
        # Annotate cells
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.2f}",
                        ha="center", va="center", fontsize=10,
                        color="white" if abs(piv.values[i, j]) > 0.5 else "black")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(fig_dir / f"delay_cost_heatmap_{model}.png")
        plt.close(fig)


def plot_threshold_tail_heatmap(policy_df: pd.DataFrame, fig_dir: Path):
    """Heatmap of Sharpe across threshold × tail quantile (tail-only policy)."""
    fig_dir = ensure_dir(fig_dir)

    tail = policy_df[policy_df["policy"] == "tail_only"]
    for model, grp in tail.groupby("model_name"):
        grp = grp.dropna(subset=["tail_quantile"])
        if grp.empty:
            continue
        piv = grp.pivot_table(
            index="tail_quantile", columns="threshold", values="sharpe", aggfunc="mean"
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn", origin="lower")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([f"{t:.2f}" for t in piv.columns])
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels([f"{q:.3f}" for q in piv.index])
        ax.set_xlabel("Threshold (τ)")
        ax.set_ylabel("Tail Quantile")
        ax.set_title(f"Sharpe — Threshold × Tail — {model}")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.2f}",
                        ha="center", va="center", fontsize=10,
                        color="white" if abs(piv.values[i, j]) > 0.5 else "black")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(fig_dir / f"threshold_tail_heatmap_{model}.png")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════════════

def save_all_diagnostic_plots(
    preds: pd.DataFrame,
    decile_df: pd.DataFrame | None = None,
    regime_df: pd.DataFrame | None = None,
    fold_desc: pd.DataFrame | None = None,
    robustness_df: pd.DataFrame | None = None,
    policy_df: pd.DataFrame | None = None,
    fig_dir: str | Path = "reports/exp002/figures",
):
    """Generate and save all exp002 diagnostic plots."""
    fig_dir = ensure_dir(Path(fig_dir))

    if decile_df is not None:
        log.info("  Plotting score-decile charts…")
        plot_decile_returns(decile_df, fig_dir)
        plot_decile_hit_rate(decile_df, fig_dir)

    log.info("  Plotting asset equity curves…")
    plot_asset_equity_curves(preds, fig_dir)

    if regime_df is not None:
        log.info("  Plotting regime Sharpe…")
        plot_regime_sharpe(regime_df, fig_dir)

    if fold_desc is not None:
        log.info("  Plotting fold PnL vs vol…")
        plot_fold_pnl_vs_vol(fold_desc, fig_dir)

    if robustness_df is not None:
        log.info("  Plotting delay-cost heatmap…")
        plot_delay_cost_heatmap(robustness_df, fig_dir)

    if policy_df is not None:
        log.info("  Plotting threshold-tail heatmap…")
        plot_threshold_tail_heatmap(policy_df, fig_dir)

    log.info("  Diagnostic plots saved to %s", fig_dir)
