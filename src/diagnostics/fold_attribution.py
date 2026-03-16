"""Fold-level regime attribution (Branch C diagnostic).

For each fold, compute regime descriptors from the feature data,
then regress fold PnL on these descriptors to explain *when* the
model works.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger("fold_attribution")


def compute_fold_descriptors(
    features: pd.DataFrame,
    fold_df: pd.DataFrame,
    preds: pd.DataFrame,
    ret_col: str = "fwd_ret_1h",
    cost_bps: float = 15.0,
    threshold: float = 0.55,
) -> pd.DataFrame:
    """Build a per-fold descriptor table.

    For each fold test window, compute regime descriptors from
    the feature data and attach fold PnL from predictions.

    Parameters
    ----------
    features : DataFrame with columns from baseline_features_v1
               plus 'asset' and 'timestamp'.
    fold_df  : fold definition DataFrame (fold_id, split, start, end).
    preds    : predictions DataFrame with fold_id, model_name, y_pred_prob, fwd_ret_1h.

    Returns
    -------
    DataFrame with one row per (model_name, fold_id) and descriptor columns.
    """
    cost = cost_bps / 10_000.0
    rows: list[dict] = []

    # Ensure tz-naive for comparison
    if hasattr(features.get("timestamp", pd.Series()), "dt"):
        feat_ts = pd.DatetimeIndex(features["timestamp"]).tz_localize(None)
    else:
        feat_ts = features["timestamp"]

    for fold_id in sorted(fold_df["fold_id"].unique()):
        fold_rows = fold_df[fold_df["fold_id"] == fold_id]
        test_r = fold_rows[fold_rows["split"] == "test"]
        if test_r.empty:
            continue
        test_r = test_r.iloc[0]

        start, end = test_r["start"], test_r["end"]
        mask = (feat_ts >= start) & (feat_ts < end)
        feat_window = features[mask]

        if len(feat_window) == 0:
            continue

        descriptors: dict = {"fold_id": fold_id}

        # Regime descriptors from features
        for col, desc_name in [
            ("realized_vol_24h", "realized_vol_mean"),
            ("market_ret_24h", "market_return_mean"),
            ("drawdown_168h", "drawdown_mean"),
            ("drawdown_24h", "drawdown_24h_mean"),
        ]:
            if col in feat_window.columns:
                descriptors[desc_name] = feat_window[col].mean()

        # SOL-specific vol
        sol_mask = feat_window["asset"] == "SOL-USD"
        if "realized_vol_24h" in feat_window.columns and sol_mask.any():
            descriptors["sol_vol_mean"] = feat_window.loc[sol_mask, "realized_vol_24h"].mean()
        else:
            descriptors["sol_vol_mean"] = np.nan

        # Trend strength
        if "ret_24h" in feat_window.columns and "realized_vol_24h" in feat_window.columns:
            cum_ret = feat_window["ret_24h"].sum()
            avg_vol = feat_window["realized_vol_24h"].mean()
            descriptors["trend_strength"] = abs(cum_ret) / avg_vol if avg_vol > 0 else 0.0

        # Weekend fraction
        if "is_weekend" in feat_window.columns:
            descriptors["weekend_fraction"] = feat_window["is_weekend"].mean()

        # High-vol fraction
        if "realized_vol_24h" in feat_window.columns:
            vol_70 = features["realized_vol_24h"].quantile(0.70)
            descriptors["high_vol_fraction"] = (feat_window["realized_vol_24h"] > vol_70).mean()

        # Drawdown fraction
        dd_col = "drawdown_168h" if "drawdown_168h" in feat_window.columns else "drawdown_24h"
        if dd_col in feat_window.columns:
            descriptors["drawdown_fraction"] = (feat_window[dd_col] < -0.10).mean()

        # Per-model fold PnL — computed on ACTIVE TRADES ONLY
        fold_preds = preds[preds["fold_id"] == fold_id]
        for model, mg in fold_preds.groupby("model_name"):
            # ── FIX: filter to active trades above threshold ──
            active = mg[mg["y_pred_prob"] > threshold]
            r = descriptors.copy()
            r["model_name"] = model
            if len(active) > 0:
                gross = active[ret_col].values
                r["fold_gross_pnl"] = gross.sum()
                r["fold_net_pnl"] = (gross - 2 * cost).sum()
                r["fold_trade_count"] = len(active)
                r["fold_hit_rate"] = (gross > 0).mean()
            else:
                r["fold_gross_pnl"] = 0.0
                r["fold_net_pnl"] = 0.0
                r["fold_trade_count"] = 0
                r["fold_hit_rate"] = np.nan
            rows.append(r)

    return pd.DataFrame(rows)


def regress_fold_pnl(
    fold_desc: pd.DataFrame,
    target: str = "fold_net_pnl",
) -> pd.DataFrame:
    """OLS regression of fold PnL on regime descriptors.

    Returns coefficient table per model.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    descriptor_cols = [
        "realized_vol_mean", "market_return_mean", "sol_vol_mean",
        "trend_strength", "weekend_fraction", "high_vol_fraction",
        "drawdown_fraction",
    ]
    available = [c for c in descriptor_cols if c in fold_desc.columns]

    results = []
    for model, mg in fold_desc.groupby("model_name"):
        df = mg.dropna(subset=available + [target])
        if len(df) < 5:
            log.warning("Skipping regression for %s: only %d folds", model, len(df))
            continue

        X = df[available].values
        y = df[target].values

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        reg = LinearRegression().fit(X_s, y)
        for feat, coef in zip(available, reg.coef_):
            results.append({
                "model_name": model,
                "descriptor": feat,
                "coefficient": coef,
                "r_squared": reg.score(X_s, y),
            })

    return pd.DataFrame(results)
