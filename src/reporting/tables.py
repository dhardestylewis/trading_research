"""Generate summary tables (CSVs) for the experiment report."""
from __future__ import annotations
from pathlib import Path

import pandas as pd

from src.backtest.metrics import trading_metrics, forecast_metrics
from src.utils.io import save_csv, ensure_dir


def model_comparison_table(
    all_sim: pd.DataFrame,
    y_true_col: str = "y_true",
    y_prob_col: str = "y_pred_prob",
) -> pd.DataFrame:
    """Produce a comparison table across models, thresholds, and cost regimes."""
    rows: list[dict] = []

    for (model, threshold, cost_regime), grp in all_sim.groupby(
        ["model_name", "threshold", "cost_regime"], sort=False
    ):
        tm = trading_metrics(grp)
        # forecast metrics only meaningful once per (model, fold) but we compute aggregate here
        mask = grp[y_true_col].notna() & grp[y_prob_col].notna()
        if mask.sum() > 0:
            fm = forecast_metrics(grp.loc[mask, y_true_col].values, grp.loc[mask, y_prob_col].values)
        else:
            fm = {}
        row = {"model_name": model, "threshold": threshold, "cost_regime": cost_regime}
        row.update(tm)
        row.update(fm)
        rows.append(row)

    return pd.DataFrame(rows)


def fold_metrics_table(
    all_sim: pd.DataFrame,
) -> pd.DataFrame:
    """Produce per-fold trading metrics for stability analysis."""
    rows: list[dict] = []

    for (model, threshold, cost_regime, fold_id), grp in all_sim.groupby(
        ["model_name", "threshold", "cost_regime", "fold_id"], sort=False
    ):
        tm = trading_metrics(grp)
        row = {"model_name": model, "threshold": threshold, "cost_regime": cost_regime, "fold_id": fold_id}
        row.update(tm)
        rows.append(row)

    return pd.DataFrame(rows)


def asset_metrics_table(all_sim: pd.DataFrame) -> pd.DataFrame:
    """Per-asset trading metrics (base cost, threshold 0.55 as default)."""
    base = all_sim[(all_sim["cost_regime"] == "base") & (all_sim["threshold"] == 0.55)]
    rows: list[dict] = []
    for (model, asset), grp in base.groupby(["model_name", "asset"], sort=False):
        tm = trading_metrics(grp)
        row = {"model_name": model, "asset": asset}
        row.update(tm)
        rows.append(row)
    return pd.DataFrame(rows)


def save_all_tables(all_sim: pd.DataFrame, out_dir: str | Path, y_true_col: str = "y_true", y_prob_col: str = "y_pred_prob") -> dict[str, Path]:
    """Save all summary tables and return paths."""
    out = ensure_dir(Path(out_dir) / "tables")
    paths = {}

    mc = model_comparison_table(all_sim, y_true_col, y_prob_col)
    paths["model_comparison"] = save_csv(mc, out / "model_comparison.csv")

    fm = fold_metrics_table(all_sim)
    paths["fold_metrics"] = save_csv(fm, out / "fold_metrics.csv")

    am = asset_metrics_table(all_sim)
    paths["asset_metrics"] = save_csv(am, out / "asset_metrics.csv")

    # Threshold sensitivity = model_comparison filtered to base cost
    ts = mc[mc["cost_regime"] == "base"][["model_name", "threshold", "sharpe", "cumulative_return", "max_drawdown", "num_trades", "exposure_fraction"]]
    paths["threshold_sensitivity"] = save_csv(ts, out / "threshold_sensitivity.csv")

    # Cost sensitivity = model_comparison filtered to threshold 0.55
    cs = mc[mc["threshold"] == 0.55][["model_name", "cost_regime", "sharpe", "cumulative_return", "max_drawdown", "avg_return_per_bar"]]
    paths["cost_sensitivity"] = save_csv(cs, out / "cost_sensitivity.csv")

    return paths
