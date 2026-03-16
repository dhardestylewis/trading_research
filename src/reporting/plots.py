"""Generate plots for the experiment report."""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from src.utils.io import ensure_dir

sns.set_theme(style="whitegrid", palette="muted")


def plot_equity_curves(sim_df: pd.DataFrame, out_dir: str | Path, model_name: str = "") -> Path:
    """Plot cumulative net return by asset and cost regime."""
    out_dir = ensure_dir(out_dir)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, (regime, rg) in enumerate(sim_df.groupby("cost_regime")):
        ax = axes[i]
        for asset, ag in rg.groupby("asset"):
            ag = ag.sort_values("timestamp")
            ax.plot(ag["timestamp"], ag["cumulative_net"], label=asset, linewidth=1.2)
        ax.set_title(f"{regime} cost")
        ax.legend(fontsize=8)
        ax.set_xlabel("")
    fig.suptitle(f"Equity Curves — {model_name}", fontsize=13)
    fig.tight_layout()
    path = out_dir / "equity_curve.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_drawdown(sim_df: pd.DataFrame, out_dir: str | Path, model_name: str = "") -> Path:
    """Plot drawdown curve for base cost regime."""
    out_dir = ensure_dir(out_dir)
    base = sim_df[sim_df["cost_regime"] == "base"]

    fig, ax = plt.subplots(figsize=(12, 4))
    for asset, ag in base.groupby("asset"):
        ag = ag.sort_values("timestamp")
        cum = ag["cumulative_net"]
        dd = (cum - cum.cummax()) / cum.cummax()
        ax.fill_between(ag["timestamp"], dd, 0, alpha=0.3, label=asset)
    ax.set_title(f"Drawdown (base cost) — {model_name}")
    ax.set_ylabel("Drawdown")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "drawdown_curve.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, out_dir: str | Path, model_name: str = "") -> Path:
    """Plot calibration curve (reliability diagram)."""
    out_dir = ensure_dir(out_dir)
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(prob_pred, prob_true, "s-", label=model_name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "calibration.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_feature_importance(model, feature_names: list[str], out_dir: str | Path, top_n: int = 20) -> Path:
    """Plot LightGBM feature importance."""
    out_dir = ensure_dir(out_dir)
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(idx)), importances[idx])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx], fontsize=8)
    ax.set_title("Feature Importance (LightGBM)")
    fig.tight_layout()
    path = out_dir / "feature_importance.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_threshold_sensitivity(metrics_df: pd.DataFrame, out_dir: str | Path) -> Path:
    """Plot Sharpe and cumulative return vs threshold."""
    out_dir = ensure_dir(out_dir)
    base = metrics_df[metrics_df["cost_regime"] == "base"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model, mg in base.groupby("model_name"):
        mg = mg.sort_values("threshold")
        axes[0].plot(mg["threshold"], mg["sharpe"], "o-", label=model)
        axes[1].plot(mg["threshold"], mg["cumulative_return"], "o-", label=model)
    axes[0].set_title("Sharpe vs Threshold (base cost)")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Sharpe")
    axes[0].legend(fontsize=8)
    axes[1].set_title("Cumulative Return vs Threshold (base cost)")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Cumulative Return")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "threshold_sensitivity.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
