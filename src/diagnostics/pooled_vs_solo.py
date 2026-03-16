"""Branch E — Pooled training vs SOL-only deployment comparison.

Compares two training regimes under identical deployment conditions:
  1. Train on ALL assets (pooled), deploy SOL only
  2. Train on SOL only, deploy SOL only

Both use the same: features, threshold, spacing rule (sep_3bar_t0.55),
fill rule (close_to_next_open), cost rule (15 bps).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.train_lightgbm import train as train_lightgbm
from src.backtest.sparse_policy import threshold_separation_policy
from src.utils.logging import get_logger

log = get_logger("pooled_vs_solo")


def _evaluate_deployment(
    preds: pd.DataFrame,
    *,
    label: str,
    threshold: float,
    sep_gap: int,
    cost_bps: float,
    ret_col: str,
    bars_per_year: float,
) -> dict:
    """Evaluate a deployment under sparse policy."""
    cost = cost_bps / 10_000.0
    af = bars_per_year

    # Sort and apply sparse policy
    sorted_preds = preds.sort_values("timestamp")
    probs = sorted_preds["y_pred_prob"].values
    positions = threshold_separation_policy(probs, threshold, sep_gap)
    active_mask = positions > 0

    if active_mask.sum() < 5:
        return {
            "training_mode": label,
            "prediction_count": len(preds),
            "active_trades": 0,
            "sharpe": np.nan,
            "cumulative_return": np.nan,
            "max_drawdown": np.nan,
            "mean_net_return": np.nan,
            "hit_rate": np.nan,
            "fold_profitability": np.nan,
            "fold_count": 0,
            "mean_score": np.nan,
        }

    gross = sorted_preds[ret_col].values[active_mask]
    net = gross - 2 * cost

    mean_net = np.nanmean(net)
    std_net = np.nanstd(net)
    sharpe = (mean_net / std_net * np.sqrt(af)) if std_net > 0 else 0.0

    cum = (1 + net).prod() - 1
    cum_series = pd.Series((1 + net).cumprod())
    dd = ((cum_series - cum_series.cummax()) / cum_series.cummax()).min()

    # Fold profitability
    if "fold_id" in sorted_preds.columns:
        fold_ids = sorted_preds["fold_id"].values[active_mask]
        fold_rets = pd.DataFrame({"fold_id": fold_ids, "ret": gross}).groupby("fold_id")["ret"].mean()
        fold_prof = (fold_rets > 0).mean() if len(fold_rets) > 0 else 0.0
        fold_count = len(fold_rets)
    else:
        fold_prof = np.nan
        fold_count = 0

    return {
        "training_mode": label,
        "prediction_count": len(preds),
        "active_trades": int(active_mask.sum()),
        "sharpe": sharpe,
        "cumulative_return": cum,
        "max_drawdown": dd,
        "mean_net_return": mean_net,
        "hit_rate": (gross > 0).mean(),
        "fold_profitability": fold_prof,
        "fold_count": fold_count,
        "mean_score": sorted_preds["y_pred_prob"].values[active_mask].mean(),
    }


def pooled_vs_solo_comparison(
    panel: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    fold_df: pd.DataFrame,
    pooled_preds: pd.DataFrame,
    feat_cols: list[str],
    *,
    target_asset: str = "SOL-USD",
    threshold: float = 0.55,
    sep_gap: int = 3,
    cost_bps: float = 15.0,
    ret_col_pooled: str = "fwd_ret_1h",
    bars_per_year: float = 365.25 * 24,
) -> pd.DataFrame:
    """Compare pooled-train/SOL-deploy vs SOL-train/SOL-deploy.

    Parameters
    ----------
    panel : full OHLCV panel.
    features : feature DataFrame with asset + timestamp columns.
    labels : labels DataFrame.
    fold_df : fold definitions.
    pooled_preds : exp001 predictions (trained on all assets).
    feat_cols : feature column names.
    target_asset : asset to deploy on.
    threshold, sep_gap, cost_bps : policy parameters.

    Returns
    -------
    DataFrame with one row per training mode and comparison metrics.
    """
    log.info("═══ Pooled vs Solo comparison ═══")

    # ── Mode 1: Pooled train, SOL deploy ──
    pooled_sol = pooled_preds[
        (pooled_preds["asset"] == target_asset) &
        (pooled_preds["model_name"] == "lightgbm")
    ].copy()
    log.info("Pooled→SOL: %d predictions", len(pooled_sol))

    row_pooled = _evaluate_deployment(
        pooled_sol,
        label="pooled_train_sol_deploy",
        threshold=threshold,
        sep_gap=sep_gap,
        cost_bps=cost_bps,
        ret_col=ret_col_pooled,
        bars_per_year=bars_per_year,
    )

    # ── Mode 2: SOL-only train, SOL deploy ──
    log.info("Training SOL-only models per fold…")

    # Build merged SOL dataset
    merged = features.copy()
    merged["asset"] = panel["asset"].values
    ts_vals = panel["timestamp"].values
    merged["timestamp"] = pd.DatetimeIndex(ts_vals).tz_localize(None)
    for col in labels.columns:
        if col not in merged.columns:
            merged[col] = labels[col].values

    sol_data = merged[merged["asset"] == target_asset].copy()
    target_col = "fwd_profitable_1h"
    ret_col_solo = "fwd_ret_1h"

    if target_col not in sol_data.columns or ret_col_solo not in sol_data.columns:
        log.warning("Missing target/return columns for SOL-only training")
        return pd.DataFrame([row_pooled])

    # Ensure fold boundaries are tz-naive
    fold_df = fold_df.copy()
    for col in ("start", "end"):
        if fold_df[col].dt.tz is not None:
            fold_df[col] = fold_df[col].dt.tz_localize(None)

    sol_preds_parts: list[pd.DataFrame] = []
    for fold_id in sorted(fold_df["fold_id"].unique()):
        fold_rows = fold_df[fold_df["fold_id"] == fold_id]
        train_r = fold_rows[fold_rows["split"] == "train"]
        test_r = fold_rows[fold_rows["split"] == "test"]
        if train_r.empty or test_r.empty:
            continue
        train_r = train_r.iloc[0]
        test_r = test_r.iloc[0]

        ts = sol_data["timestamp"]
        train_mask = (ts >= train_r["start"]) & (ts < train_r["end"])
        test_mask = (ts >= test_r["start"]) & (ts < test_r["end"])

        train_df = sol_data[train_mask].copy()
        test_df = sol_data[test_mask].copy()

        required = feat_cols + [target_col, ret_col_solo]
        for df_ in (train_df, test_df):
            valid = df_[required].notna().all(axis=1)
            df_.drop(df_[~valid].index, inplace=True)

        if len(train_df) < 50 or len(test_df) < 5:
            continue

        tm = train_lightgbm(
            train_df[feat_cols], train_df[target_col],
            config_path="configs/models/lightgbm_v1.yaml",
            feature_names=feat_cols,
        )
        probs = tm.predict_proba(test_df)

        pred_df = test_df[["asset", "timestamp", ret_col_solo]].copy()
        pred_df["y_pred_prob"] = probs
        pred_df["fold_id"] = fold_id
        pred_df["model_name"] = "lightgbm"
        sol_preds_parts.append(pred_df)

    if not sol_preds_parts:
        log.warning("No SOL-only predictions produced")
        return pd.DataFrame([row_pooled])

    sol_only_preds = pd.concat(sol_preds_parts, ignore_index=True)
    log.info("SOL-only: %d predictions across %d folds",
             len(sol_only_preds), sol_only_preds["fold_id"].nunique())

    row_solo = _evaluate_deployment(
        sol_only_preds,
        label="sol_train_sol_deploy",
        threshold=threshold,
        sep_gap=sep_gap,
        cost_bps=cost_bps,
        ret_col=ret_col_solo,
        bars_per_year=bars_per_year,
    )

    df = pd.DataFrame([row_pooled, row_solo])
    log.info("Pooled vs Solo:\n%s", df.to_string(index=False))
    return df
