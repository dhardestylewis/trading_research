"""Branch A — Multi-horizon stacking for increased trade count.

Train parallel LightGBM models at 1h, 2h, and 4h horizons, run the
reference sparse policy independently on each, then combine into a
composite stream with deduplication.

Goal: 2–3× trade count while keeping net expectancy positive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.sparse_policy import threshold_separation_policy
from src.diagnostics.regime_labeller import label_regimes
from src.models.train_lightgbm import train as train_lightgbm
from src.utils.logging import get_logger

log = get_logger("multi_horizon_stacking")


def _evaluate_horizon(
    preds: pd.DataFrame,
    *,
    horizon: int,
    threshold: float,
    sep_gap: int,
    cost_bps: float,
    bars_per_year: float = 365.25 * 24,
) -> dict:
    """Evaluate a single horizon under the sparse policy."""
    ret_col = f"fwd_ret_{horizon}h"
    cost = cost_bps / 10_000.0

    sorted_p = preds.sort_values("timestamp")
    probs = sorted_p["y_pred_prob"].values
    positions = threshold_separation_policy(probs, threshold, sep_gap)
    active = positions > 0

    if active.sum() < 3:
        return {
            "horizon": horizon,
            "trade_count": 0,
            "sharpe": np.nan,
            "cumulative_return": np.nan,
            "mean_net_bps": np.nan,
            "hit_rate": np.nan,
            "fold_profitability": np.nan,
        }

    gross = sorted_p[ret_col].values[active]
    net = gross - 2 * cost
    mean_net = np.nanmean(net)
    std_net = np.nanstd(net)
    sharpe = (mean_net / std_net * np.sqrt(bars_per_year)) if std_net > 0 else 0.0
    cum = (1 + net).prod() - 1

    fold_ids = sorted_p["fold_id"].values[active] if "fold_id" in sorted_p.columns else np.zeros(active.sum())
    fold_rets = pd.DataFrame({"fold_id": fold_ids, "ret": gross}).groupby("fold_id")["ret"].mean()
    fold_prof = (fold_rets > 0).mean() if len(fold_rets) > 0 else 0.0

    return {
        "horizon": horizon,
        "trade_count": int(active.sum()),
        "sharpe": sharpe,
        "cumulative_return": cum,
        "mean_net_bps": mean_net * 10_000,
        "hit_rate": (gross > 0).mean(),
        "fold_profitability": fold_prof,
    }


def train_horizon_models(
    merged: pd.DataFrame,
    fold_df: pd.DataFrame,
    feat_cols: list[str],
    horizons: list[int],
    *,
    target_asset: str = "SOL-USD",
    training_mode: str = "pooled",
) -> dict[int, pd.DataFrame]:
    """Train per-horizon models and return predictions keyed by horizon.

    Parameters
    ----------
    merged : DataFrame with features + labels + asset + timestamp columns.
    fold_df : Walk-forward fold definitions.
    feat_cols : Feature column names.
    horizons : List of horizon integers (e.g. [1, 2, 4]).
    target_asset : Asset to deploy on.
    training_mode : 'pooled' trains on all assets, 'solo' trains on target only.

    Returns
    -------
    Dict mapping horizon -> DataFrame of predictions for the target asset.
    """
    # Normalize fold boundaries to tz-naive
    fold_df = fold_df.copy()
    for col in ("start", "end"):
        if fold_df[col].dt.tz is not None:
            fold_df[col] = fold_df[col].dt.tz_localize(None)

    horizon_preds: dict[int, pd.DataFrame] = {}

    for h in horizons:
        target_col = f"fwd_profitable_{h}h"
        ret_col = f"fwd_ret_{h}h"

        if target_col not in merged.columns or ret_col not in merged.columns:
            log.warning("Missing columns for horizon %dh: %s / %s", h, target_col, ret_col)
            continue

        log.info("── Training %dh horizon model ──", h)
        all_preds: list[pd.DataFrame] = []

        for fold_id in sorted(fold_df["fold_id"].unique()):
            fold_rows = fold_df[fold_df["fold_id"] == fold_id]
            train_r = fold_rows[fold_rows["split"] == "train"]
            test_r = fold_rows[fold_rows["split"] == "test"]
            if train_r.empty or test_r.empty:
                continue
            train_r = train_r.iloc[0]
            test_r = test_r.iloc[0]

            ts = merged["timestamp"]
            train_mask = (ts >= train_r["start"]) & (ts < train_r["end"])
            test_mask = (ts >= test_r["start"]) & (ts < test_r["end"])

            if training_mode == "solo":
                train_mask = train_mask & (merged["asset"] == target_asset)

            train_df = merged[train_mask].copy()
            test_df = merged[test_mask & (merged["asset"] == target_asset)].copy()

            required = feat_cols + [target_col, ret_col]
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

            pred_df = test_df[["asset", "timestamp", ret_col]].copy()
            pred_df["y_pred_prob"] = probs
            pred_df["fold_id"] = fold_id
            all_preds.append(pred_df)

        if all_preds:
            horizon_preds[h] = pd.concat(all_preds, ignore_index=True)
            log.info("  %dh: %d predictions", h, len(horizon_preds[h]))

    return horizon_preds


def stack_horizons(
    horizon_preds: dict[int, pd.DataFrame],
    *,
    threshold: float = 0.55,
    sep_gap: int = 3,
    cost_bps: float = 15.0,
    dedup_window: int = 2,
) -> pd.DataFrame:
    """Combine horizon streams into a composite signal with dedup.

    For each horizon, apply the sparse policy independently, then merge
    into a single timeline. Deduplicate trades that fire within
    *dedup_window* bars of each other across horizons (keep the one with
    the highest score).

    Returns a DataFrame with columns: timestamp, horizon, y_pred_prob,
    position, ret, net_ret.
    """
    all_signals: list[pd.DataFrame] = []

    for h, preds in sorted(horizon_preds.items()):
        ret_col = f"fwd_ret_{h}h"
        sorted_p = preds.sort_values("timestamp").copy()
        probs = sorted_p["y_pred_prob"].values
        positions = threshold_separation_policy(probs, threshold, sep_gap)
        active_mask = positions > 0

        if active_mask.sum() == 0:
            continue

        active = sorted_p[active_mask].copy()
        active["horizon"] = h
        active["position"] = 1.0
        active["ret"] = active[ret_col]
        active["net_ret"] = active[ret_col] - 2 * (cost_bps / 10_000.0)
        all_signals.append(active[["timestamp", "horizon", "y_pred_prob", "position", "ret", "net_ret", "fold_id"]])

    if not all_signals:
        return pd.DataFrame()

    combined = pd.concat(all_signals, ignore_index=True).sort_values("timestamp")

    # Deduplication: within each dedup window, keep highest-scoring signal
    combined["ts_rank"] = combined["timestamp"].rank(method="dense").astype(int)
    combined["group"] = (combined["ts_rank"].diff() > dedup_window).cumsum()
    deduped = combined.loc[combined.groupby("group")["y_pred_prob"].idxmax()].copy()
    deduped = deduped.drop(columns=["ts_rank", "group"])

    return deduped.reset_index(drop=True)


def horizon_stacking_study(
    merged: pd.DataFrame,
    fold_df: pd.DataFrame,
    feat_cols: list[str],
    pooled_preds: pd.DataFrame | None = None,
    *,
    target_asset: str = "SOL-USD",
    horizons: list[int] = (1, 2, 4),
    threshold: float = 0.55,
    sep_gap: int = 3,
    cost_bps: float = 15.0,
    dedup_window: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full multi-horizon stacking study.

    Returns
    -------
    per_horizon : DataFrame with one row per horizon + metrics.
    stacked : DataFrame with composite stream trades.
    """
    log.info("═══ Branch A: Multi-horizon stacking ═══")

    # Train per-horizon models (pooled train, target-asset deploy)
    horizon_preds = train_horizon_models(
        merged, fold_df, feat_cols, list(horizons),
        target_asset=target_asset,
        training_mode="pooled",
    )

    # If we have existing 1h predictions from exp001, use those for horizon=1
    if pooled_preds is not None and 1 in horizons:
        sol_1h = pooled_preds[
            (pooled_preds["asset"] == target_asset) &
            (pooled_preds["model_name"] == "lightgbm")
        ].copy()
        if not sol_1h.empty and "fwd_ret_1h" in sol_1h.columns:
            horizon_preds[1] = sol_1h

    # Per-horizon evaluation
    per_horizon_rows: list[dict] = []
    for h, preds in sorted(horizon_preds.items()):
        row = _evaluate_horizon(
            preds, horizon=h,
            threshold=threshold, sep_gap=sep_gap, cost_bps=cost_bps,
        )
        per_horizon_rows.append(row)
        log.info("  %dh: %d trades, Sharpe=%.2f, net_bps=%.1f",
                 h, row["trade_count"], row.get("sharpe", 0), row.get("mean_net_bps", 0))

    per_horizon = pd.DataFrame(per_horizon_rows)

    # Stacked composite
    stacked = stack_horizons(
        horizon_preds,
        threshold=threshold,
        sep_gap=sep_gap,
        cost_bps=cost_bps,
        dedup_window=dedup_window,
    )

    # Compute composite metrics
    if not stacked.empty:
        net = stacked["net_ret"].values
        mean_net = np.nanmean(net)
        std_net = np.nanstd(net)
        sharpe = (mean_net / std_net * np.sqrt(365.25 * 24)) if std_net > 0 else 0.0
        base_1h_count = per_horizon.loc[per_horizon["horizon"] == 1, "trade_count"].values
        base_count = base_1h_count[0] if len(base_1h_count) > 0 else 1

        overlap_bars = stacked.groupby("horizon").size()
        log.info("  Stacked: %d trades (%.1f× 1h baseline), Sharpe=%.2f",
                 len(stacked), len(stacked) / max(base_count, 1), sharpe)

        composite_row = {
            "horizon": "stacked",
            "trade_count": len(stacked),
            "sharpe": sharpe,
            "cumulative_return": (1 + net).prod() - 1,
            "mean_net_bps": mean_net * 10_000,
            "hit_rate": (stacked["ret"].values > 0).mean(),
            "fold_profitability": np.nan,  # mixed folds
            "trade_multiplier": len(stacked) / max(base_count, 1),
        }
        per_horizon = pd.concat([per_horizon, pd.DataFrame([composite_row])], ignore_index=True)

    return per_horizon, stacked
