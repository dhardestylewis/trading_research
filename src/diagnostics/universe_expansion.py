"""Branch B — Universe expansion to SOL-family assets.

Train pooled on ALL assets (core + expansion), deploy individually on
each expansion candidate under the reference sparse policy.  Test
whether the SOL signal transports to structurally similar L1/L2 tokens.

Goal: ≥2 additional assets with Sharpe > 0.5 and fold profitability > 50%.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.sparse_policy import threshold_separation_policy
from src.models.train_lightgbm import train as train_lightgbm
from src.utils.logging import get_logger

log = get_logger("universe_expansion")


def _evaluate_asset(
    preds: pd.DataFrame,
    *,
    asset: str,
    threshold: float,
    sep_gap: int,
    cost_bps: float,
    ret_col: str = "fwd_ret_1h",
    bars_per_year: float = 365.25 * 24,
) -> dict:
    """Evaluate a single asset under the sparse policy."""
    cost = cost_bps / 10_000.0

    sorted_p = preds.sort_values("timestamp")
    probs = sorted_p["y_pred_prob"].values
    positions = threshold_separation_policy(probs, threshold, sep_gap)
    active = positions > 0

    if active.sum() < 3:
        return {
            "asset": asset,
            "prediction_count": len(preds),
            "trade_count": 0,
            "sharpe": np.nan,
            "cumulative_return": np.nan,
            "mean_net_bps": np.nan,
            "hit_rate": np.nan,
            "fold_profitability": np.nan,
            "qualifies": False,
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
        "asset": asset,
        "prediction_count": len(preds),
        "trade_count": int(active.sum()),
        "sharpe": sharpe,
        "cumulative_return": cum,
        "mean_net_bps": mean_net * 10_000,
        "hit_rate": (gross > 0).mean(),
        "fold_profitability": fold_prof,
        "qualifies": False,  # set by caller
    }


def train_pooled_deploy_multi(
    merged: pd.DataFrame,
    fold_df: pd.DataFrame,
    feat_cols: list[str],
    deploy_assets: list[str],
    *,
    target_col: str = "fwd_profitable_1h",
    ret_col: str = "fwd_ret_1h",
) -> dict[str, pd.DataFrame]:
    """Train on all assets pooled, deploy per-asset.

    Returns dict mapping asset name to DataFrame of predictions.
    """
    fold_df = fold_df.copy()
    for col in ("start", "end"):
        if fold_df[col].dt.tz is not None:
            fold_df[col] = fold_df[col].dt.tz_localize(None)

    asset_preds: dict[str, list[pd.DataFrame]] = {a: [] for a in deploy_assets}

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

        train_df = merged[train_mask].copy()
        required = feat_cols + [target_col, ret_col]
        valid = train_df[required].notna().all(axis=1)
        train_df = train_df[valid]

        if len(train_df) < 50:
            continue

        # Train pooled model
        tm = train_lightgbm(
            train_df[feat_cols], train_df[target_col],
            config_path="configs/models/lightgbm_v1.yaml",
            feature_names=feat_cols,
        )

        # Deploy on each target asset
        for asset in deploy_assets:
            test_asset = merged[test_mask & (merged["asset"] == asset)].copy()
            test_valid = test_asset[required].notna().all(axis=1)
            test_asset = test_asset[test_valid]

            if len(test_asset) < 5:
                continue

            probs = tm.predict_proba(test_asset)
            pred_df = test_asset[["asset", "timestamp", ret_col]].copy()
            pred_df["y_pred_prob"] = probs
            pred_df["fold_id"] = fold_id
            asset_preds[asset].append(pred_df)

    # Concatenate per-asset predictions
    result: dict[str, pd.DataFrame] = {}
    for asset, parts in asset_preds.items():
        if parts:
            result[asset] = pd.concat(parts, ignore_index=True)
            log.info("  %s: %d predictions across %d folds",
                     asset, len(result[asset]), result[asset]["fold_id"].nunique())
    return result


def transportability_matrix(
    asset_preds: dict[str, pd.DataFrame],
    *,
    threshold: float = 0.55,
    sep_gap: int = 3,
    cost_bps: float = 15.0,
    min_sharpe: float = 0.5,
    min_fold_profitability: float = 0.50,
) -> pd.DataFrame:
    """Evaluate each asset and build the transportability matrix."""
    rows: list[dict] = []

    for asset, preds in sorted(asset_preds.items()):
        row = _evaluate_asset(
            preds, asset=asset,
            threshold=threshold, sep_gap=sep_gap, cost_bps=cost_bps,
        )
        row["qualifies"] = (
            row["sharpe"] > min_sharpe and
            row["fold_profitability"] > min_fold_profitability and
            row["trade_count"] >= 5
        ) if not np.isnan(row.get("sharpe", np.nan)) else False
        rows.append(row)

    return pd.DataFrame(rows)


def universe_expansion_study(
    merged: pd.DataFrame,
    fold_df: pd.DataFrame,
    feat_cols: list[str],
    expansion_assets: list[str],
    *,
    threshold: float = 0.55,
    sep_gap: int = 3,
    cost_bps: float = 15.0,
    min_sharpe: float = 0.5,
    min_fold_profitability: float = 0.50,
) -> pd.DataFrame:
    """Run the full universe expansion study.

    Returns
    -------
    DataFrame with one row per asset + aggregate row for qualifying assets.
    """
    log.info("═══ Branch B: Universe expansion ═══")

    # Include SOL as reference + expansion candidates
    all_deploy = ["SOL-USD"] + [a for a in expansion_assets if a != "SOL-USD"]
    available = [a for a in all_deploy if a in merged["asset"].unique()]

    if not available:
        log.warning("No expansion assets found in panel data")
        return pd.DataFrame()

    log.info("Assets available for deployment: %s", available)

    # Train pooled, deploy multi
    asset_preds = train_pooled_deploy_multi(
        merged, fold_df, feat_cols,
        deploy_assets=available,
    )

    # Build transportability matrix
    matrix = transportability_matrix(
        asset_preds,
        threshold=threshold,
        sep_gap=sep_gap,
        cost_bps=cost_bps,
        min_sharpe=min_sharpe,
        min_fold_profitability=min_fold_profitability,
    )

    # Aggregate row for qualifying assets
    qualifying = matrix[matrix["qualifies"]]
    if not qualifying.empty:
        agg = {
            "asset": "QUALIFYING_AGGREGATE",
            "prediction_count": int(qualifying["prediction_count"].sum()),
            "trade_count": int(qualifying["trade_count"].sum()),
            "sharpe": qualifying["sharpe"].mean(),
            "cumulative_return": qualifying["cumulative_return"].mean(),
            "mean_net_bps": qualifying["mean_net_bps"].mean(),
            "hit_rate": qualifying["hit_rate"].mean(),
            "fold_profitability": qualifying["fold_profitability"].mean(),
            "qualifies": True,
        }
        matrix = pd.concat([matrix, pd.DataFrame([agg])], ignore_index=True)

    n_qual = qualifying.shape[0]
    log.info("Qualifying assets: %d / %d", n_qual, len(available))
    for _, row in matrix.iterrows():
        log.info("  %s: %d trades, Sharpe=%.2f, net=%.1f bps, qualifies=%s",
                 row["asset"], row["trade_count"],
                 row.get("sharpe", 0), row.get("mean_net_bps", 0),
                 row.get("qualifies", False))

    return matrix
