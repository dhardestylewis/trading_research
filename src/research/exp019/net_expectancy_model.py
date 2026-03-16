"""Multi-head LightGBM model for the full money equation.

Trains separate LightGBM regressors for:
  - Head 1: gross_move_bps (forward return)
  - Head 2: exec_loss_bps (execution shortfall estimate)
  - Head 3: net_move_bps (gross - exec_loss - fees)

Uses temporal walk-forward splits (expanding window, no look-ahead).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.utils.logging import get_logger

log = get_logger("exp019.net_expectancy_model")


@dataclass
class HeadResult:
    """Result from training a single prediction head."""
    name: str
    model: lgb.LGBMRegressor
    feature_importance: pd.DataFrame
    oof_predictions: pd.Series        # out-of-fold predictions
    oof_actuals: pd.Series            # corresponding actuals
    metrics: dict = field(default_factory=dict)


def _build_labels(
    panel: pd.DataFrame, horizon_bars: int,
    fee_bps_per_leg: float, slippage_bps_per_leg: float,
) -> pd.DataFrame:
    """Build forward-return labels for each timestamp×asset.

    Returns DataFrame with columns:
        gross_move_bps : forward return in bps
        exec_loss_bps  : estimated execution shortfall (simplified model)
        net_move_bps   : gross - exec_loss - round-trip fees
    """
    labels = pd.DataFrame(index=panel.index)
    parts: list[pd.DataFrame] = []

    for asset, g in panel.groupby("asset", sort=False):
        g_sorted = g.sort_values("timestamp")
        c = g_sorted["close"]

        fwd_ret = c.shift(-horizon_bars) / c - 1
        gross_bps = fwd_ret * 10_000

        # Execution loss proxy: absolute intrabar range as fraction of price
        # This captures adverse selection — large range bars are harder to execute
        h = g_sorted["high"]
        lo = g_sorted["low"]
        intrabar_range = (h - lo) / c * 10_000  # in bps
        # Exec loss ~ fraction of intrabar range (conservative: 30% of range)
        exec_loss_bps = intrabar_range.rolling(3).mean() * 0.30

        # Net = gross - exec_loss - round-trip fees
        rt_fees = (fee_bps_per_leg + slippage_bps_per_leg) * 2
        net_bps = gross_bps - exec_loss_bps - rt_fees

        part = pd.DataFrame({
            "gross_move_bps": gross_bps,
            "exec_loss_bps": exec_loss_bps,
            "net_move_bps": net_bps,
        }, index=g_sorted.index)
        parts.append(part)

    labels = pd.concat(parts).loc[panel.index]
    return labels


def _walk_forward_splits(
    timestamps: pd.Series,
    min_train_bars: int,
    val_bars: int,
    step_bars: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window walk-forward train/val index splits."""
    unique_ts = timestamps.sort_values().unique()
    n = len(unique_ts)

    splits = []
    train_end = min_train_bars

    while train_end + val_bars <= n:
        train_ts = set(unique_ts[:train_end])
        val_ts = set(unique_ts[train_end:train_end + val_bars])

        train_idx = timestamps[timestamps.isin(train_ts)].index.values
        val_idx = timestamps[timestamps.isin(val_ts)].index.values

        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))

        train_end += step_bars

    return splits


def _train_single_head(
    name: str,
    features: pd.DataFrame,
    target: pd.Series,
    feature_cols: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    params: dict,
    early_stopping_rounds: int,
) -> HeadResult:
    """Train a single LightGBM regressor with walk-forward OOF predictions."""
    log.info("Training head: %s (%d splits)", name, len(splits))

    oof_preds = pd.Series(np.nan, index=features.index, name=f"pred_{name}")
    oof_actuals = pd.Series(np.nan, index=features.index, name=f"actual_{name}")
    importances: list[np.ndarray] = []
    fold_metrics: list[dict] = []

    last_model = None

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        X_train_df = features.loc[train_idx, feature_cols]
        y_train = target.loc[train_idx].values
        X_val_df = features.loc[val_idx, feature_cols]
        y_val = target.loc[val_idx].values

        # Drop NaN rows
        train_mask = np.isfinite(X_train_df.values).all(axis=1) & np.isfinite(y_train)
        val_mask = np.isfinite(X_val_df.values).all(axis=1) & np.isfinite(y_val)
        X_train_df = X_train_df[train_mask]
        y_train = y_train[train_mask]
        X_val_df = X_val_df[val_mask]
        y_val = y_val[val_mask]

        if len(X_train_df) < 100 or len(X_val_df) < 20:
            continue

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train_df, y_train,
            eval_set=[(X_val_df, y_val)],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )

        preds = model.predict(X_val_df)

        # Store OOF
        valid_val_idx = val_idx[val_mask]
        oof_preds.loc[valid_val_idx] = preds
        oof_actuals.loc[valid_val_idx] = y_val

        importances.append(model.feature_importances_)
        last_model = model

        # Fold metrics
        corr = np.corrcoef(preds, y_val)[0, 1] if len(preds) > 2 else 0
        mae = np.mean(np.abs(preds - y_val))
        fold_metrics.append({
            "fold": fold_i,
            "n_train": len(X_train_df),
            "n_val": len(X_val_df),
            "correlation": round(corr, 4),
            "mae_bps": round(mae, 2),
        })

    # Aggregate importance
    if importances:
        avg_imp = np.mean(importances, axis=0)
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": avg_imp,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
    else:
        importance_df = pd.DataFrame(columns=["feature", "importance"])

    # Aggregate OOF metrics
    valid = oof_preds.dropna()
    actual_valid = oof_actuals.loc[valid.index]
    overall_corr = np.corrcoef(valid.values, actual_valid.values)[0, 1] if len(valid) > 2 else 0
    overall_mae = np.mean(np.abs(valid.values - actual_valid.values)) if len(valid) > 0 else np.nan

    metrics = {
        "n_folds": len(fold_metrics),
        "n_oof_samples": len(valid),
        "oof_correlation": round(overall_corr, 4),
        "oof_mae_bps": round(overall_mae, 2) if not np.isnan(overall_mae) else None,
        "fold_details": fold_metrics,
    }

    log.info(
        "  %s: OOF corr=%.4f, MAE=%.1f bps (%d samples, %d folds)",
        name, overall_corr, overall_mae if not np.isnan(overall_mae) else 0,
        len(valid), len(fold_metrics),
    )

    return HeadResult(
        name=name,
        model=last_model,
        feature_importance=importance_df,
        oof_predictions=oof_preds,
        oof_actuals=oof_actuals,
        metrics=metrics,
    )


def train_multi_head(
    features: pd.DataFrame,
    panel: pd.DataFrame,
    feature_cols: list[str],
    horizon_bars: int,
    model_cfg: dict,
    exec_cfg: dict,
) -> dict:
    """Train all three prediction heads.

    Parameters
    ----------
    features : extended feature DataFrame
    panel : raw panel with OHLCV
    feature_cols : list of feature column names to use
    horizon_bars : forward-looking horizon in bars
    model_cfg : model config from YAML
    exec_cfg : execution/friction config from YAML

    Returns
    -------
    dict with:
        labels : DataFrame of targets
        heads : dict of {name: HeadResult}
        combined_predictions : DataFrame with OOF preds for all heads
    """
    # Build labels
    labels = _build_labels(
        panel, horizon_bars,
        exec_cfg["fee_bps_per_leg"],
        exec_cfg["slippage_bps_per_leg"],
    )
    log.info(
        "Labels built: %d rows, gross mean=%.1f bps, net mean=%.1f bps",
        len(labels),
        labels["gross_move_bps"].mean(),
        labels["net_move_bps"].mean(),
    )

    # Walk-forward splits
    wf = model_cfg.get("walk_forward", {})
    splits = _walk_forward_splits(
        panel["timestamp"],
        wf.get("min_train_bars", 720),
        wf.get("val_bars", 168),
        wf.get("step_bars", 168),
    )
    log.info("Walk-forward: %d splits", len(splits))

    # LightGBM params
    params = dict(model_cfg.get("params", {}))
    early_stop = model_cfg.get("early_stopping_rounds", 50)

    # Train each head
    target_names = ["gross_move_bps", "exec_loss_bps", "net_move_bps"]
    heads: dict[str, HeadResult] = {}

    for tname in target_names:
        if tname not in labels.columns:
            log.warning("Target %s not found in labels, skipping", tname)
            continue
        head = _train_single_head(
            tname, features, labels[tname],
            feature_cols, splits, params, early_stop,
        )
        heads[tname] = head

    # Combine OOF predictions
    combined = pd.DataFrame(index=features.index)
    for tname, head in heads.items():
        combined[f"pred_{tname}"] = head.oof_predictions
        combined[f"actual_{tname}"] = head.oof_actuals
    combined["asset"] = features["asset"].values
    combined["timestamp"] = features["timestamp"].values

    return {
        "labels": labels,
        "heads": heads,
        "combined_predictions": combined,
    }
