"""SOL-only horizon extension study (Branch 3).

For each horizon in [1, 2, 4] bars:
  - Filter panel to SOL-USD only
  - Train LightGBM per fold on SOL-only data
  - Evaluate at thresholds with delay sweep
  - Compare horizon × delay metrics
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from src.models.train_lightgbm import train as train_lightgbm
from src.validation.fold_builder import build_folds
from src.diagnostics.robustness_grid import shift_predictions
from src.utils.logging import get_logger

log = get_logger("sol_horizon_study")


def run_sol_horizon_study(
    panel: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    fold_df: pd.DataFrame,
    feat_cols: list[str],
    horizons: list[int] = (1, 2, 4),
    thresholds: list[float] = (0.50, 0.55, 0.60),
    delays: list[int] = (0, 1, 2),
    cost_bps: float = 15.0,
    bars_per_year: float = 365.25 * 24,
) -> pd.DataFrame:
    """Run SOL-only LightGBM across multiple horizons.

    Returns one row per (horizon, threshold, delay) with full metrics.
    """
    cost = cost_bps / 10_000.0
    af = bars_per_year

    # Build merged dataset
    merged = features.copy()
    merged["asset"] = panel["asset"].values
    merged["timestamp"] = pd.DatetimeIndex(panel["timestamp"].values).tz_localize(None)

    # Attach all horizon labels
    for col in labels.columns:
        if col not in merged.columns:
            merged[col] = labels[col].values

    # Filter to SOL-USD
    sol_mask = merged["asset"] == "SOL-USD"
    merged_sol = merged[sol_mask].copy()
    log.info("SOL-only dataset: %d rows", len(merged_sol))

    # Ensure fold boundaries are tz-naive
    fold_df = fold_df.copy()
    for col in ("start", "end"):
        if fold_df[col].dt.tz is not None:
            fold_df[col] = fold_df[col].dt.tz_localize(None)

    rows: list[dict] = []

    for h in horizons:
        target_col = f"fwd_profitable_{h}h"
        ret_col = f"fwd_ret_{h}h"

        if target_col not in merged_sol.columns:
            log.warning("Horizon %dh: target %s not found — skipping", h, target_col)
            continue

        log.info("── Horizon %dh ──", h)

        all_preds: list[pd.DataFrame] = []

        for fold_id in sorted(fold_df["fold_id"].unique()):
            fold_rows = fold_df[fold_df["fold_id"] == fold_id]
            train_r = fold_rows[fold_rows["split"] == "train"]
            test_r = fold_rows[fold_rows["split"] == "test"]
            if train_r.empty or test_r.empty:
                continue
            train_r = train_r.iloc[0]
            test_r = test_r.iloc[0]

            ts = merged_sol["timestamp"]
            train_mask = (ts >= train_r["start"]) & (ts < train_r["end"])
            test_mask = (ts >= test_r["start"]) & (ts < test_r["end"])

            train_df = merged_sol[train_mask].copy()
            test_df = merged_sol[test_mask].copy()

            # Drop NaN rows
            required = feat_cols + [target_col, ret_col]
            for df_ in (train_df, test_df):
                valid = df_[required].notna().all(axis=1)
                df_.drop(df_[~valid].index, inplace=True)

            if len(train_df) < 50 or len(test_df) < 5:
                continue

            # Train
            tm = train_lightgbm(
                train_df[feat_cols], train_df[target_col],
                config_path="configs/models/lightgbm_v1.yaml",
                feature_names=feat_cols,
            )
            probs = tm.predict_proba(test_df)

            pred_df = test_df[["asset", "timestamp", ret_col]].copy()
            pred_df = pred_df.rename(columns={ret_col: "fwd_ret"})
            pred_df["y_pred_prob"] = probs
            pred_df["fold_id"] = fold_id
            pred_df["model_name"] = "lightgbm"
            all_preds.append(pred_df)

        if not all_preds:
            log.warning("Horizon %dh: no predictions produced", h)
            continue

        preds = pd.concat(all_preds, ignore_index=True)
        log.info("  Horizon %dh: %d prediction rows", h, len(preds))

        # Evaluate across delays and thresholds
        for delay in delays:
            shifted = shift_predictions(preds, delay)

            for threshold in thresholds:
                above = shifted[shifted["y_pred_prob"] > threshold]
                if len(above) < 5:
                    continue

                gross = above["fwd_ret"].values
                net = gross - 2 * cost
                mean_net = np.nanmean(net)
                std_net = np.nanstd(net)
                sharpe = (mean_net / std_net * np.sqrt(af)) if std_net > 0 else 0.0

                cum = (1 + net).prod() - 1
                cum_series = pd.Series((1 + net).cumprod())
                dd = ((cum_series - cum_series.cummax()) / cum_series.cummax()).min()

                fold_rets = above.groupby("fold_id")["fwd_ret"].mean()
                fold_prof = (fold_rets > 0).mean() if len(fold_rets) > 0 else 0.0

                rows.append({
                    "horizon_h": h,
                    "threshold": threshold,
                    "delay_bars": delay,
                    "sharpe": sharpe,
                    "cumulative_return": cum,
                    "max_drawdown": dd,
                    "mean_net_return": mean_net,
                    "hit_rate": (gross > 0).mean(),
                    "trade_count": len(above),
                    "fold_profitability": fold_prof,
                    "fold_count": above["fold_id"].nunique(),
                })

    return pd.DataFrame(rows)
