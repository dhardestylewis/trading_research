"""Dilution diagnosis — exp009 core module.

Investigates whether expanding the training pool contaminates
the SOL edge. Compares per-asset performance across different
training pool compositions.

Branches:
  A: Original 3-asset pool (SOL/BTC/ETH) → SOL deploy
  B: Full 8-asset pool → SOL deploy
  C: Family pool (SOL/APT/SUI/NEAR) → per-asset deploy
  D: Per-asset heads on shared backbone (scaffold)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.train_lightgbm import train as train_lightgbm
from src.utils.logging import get_logger

log = get_logger("dilution_diagnosis")


# ── Helpers ─────────────────────────────────────────────────


def _run_pooled_backtest(
    merged: pd.DataFrame,
    fold_df: pd.DataFrame,
    feat_cols: list[str],
    *,
    train_assets: list[str],
    deploy_assets: list[str],
    threshold: float = 0.55,
    sep_gap: int = 3,
    cost_bps: float = 15.0,
) -> pd.DataFrame:
    """Train on train_assets, deploy on deploy_assets, return per-asset metrics."""
    results: list[dict] = []

    # Fold DF has columns: fold_id, split, start, end
    # Group by fold_id to pair train/test splits correctly
    if "fold_id" in fold_df.columns:
        fold_ids = fold_df["fold_id"].unique()
    else:
        # Fallback: treat each row as a fold
        fold_ids = fold_df.index.values

    for fold_id in fold_ids:
        if "fold_id" in fold_df.columns:
            fold_group = fold_df[fold_df["fold_id"] == fold_id]
            train_split = fold_group[fold_group["split"] == "train"]
            test_split = fold_group[fold_group["split"] == "test"]

            if train_split.empty or test_split.empty:
                continue

            train_start = train_split["start"].iloc[0]
            train_end = train_split["end"].iloc[0]
            test_start = test_split["start"].iloc[0]
            test_end = test_split["end"].iloc[0]
        else:
            fold_row = fold_df.iloc[fold_id]
            train_start = fold_row["start"]
            train_end = fold_row["end"]
            test_start = train_end
            test_end = None  # no upper bound

        train_mask = (
            (merged["timestamp"] >= train_start)
            & (merged["timestamp"] < train_end)
            & (merged["asset"].isin(train_assets))
        )
        if test_end is not None:
            test_mask = (
                (merged["timestamp"] >= test_start)
                & (merged["timestamp"] < test_end)
                & (merged["asset"].isin(deploy_assets))
            )
        else:
            test_mask = (
                (merged["timestamp"] >= test_start)
                & (merged["asset"].isin(deploy_assets))
            )

        train_data = merged[train_mask]
        test_data = merged[test_mask]

        if len(train_data) < 50 or len(test_data) < 10:
            continue

        label_col = "fwd_profitable_1h"
        if label_col not in train_data.columns:
            log.warning("Label column %s not found, skipping fold %s", label_col, fold_id)
            continue

        valid_train = train_data.dropna(subset=[label_col])
        valid_test = test_data.dropna(subset=[label_col])

        if len(valid_train) < 50 or len(valid_test) < 5:
            continue

        available_feats = [c for c in feat_cols if c in valid_train.columns]
        if not available_feats:
            continue

        # Also drop rows with NaN features
        feat_train_mask = valid_train[available_feats].notna().all(axis=1)
        valid_train = valid_train[feat_train_mask]
        feat_test_mask = valid_test[available_feats].notna().all(axis=1)
        valid_test = valid_test[feat_test_mask]

        if len(valid_train) < 50 or len(valid_test) < 5:
            continue

        try:
            model = train_lightgbm(
                valid_train[available_feats],
                valid_train[label_col],
                feature_names=available_feats,
            )
            preds = model.predict_proba(valid_test[available_feats])
        except Exception as e:
            log.warning("Fold %s training failed: %s", fold_id, e)
            continue

        # Per-asset signal extraction
        for asset in deploy_assets:
            asset_mask = valid_test["asset"] == asset
            if asset_mask.sum() == 0:
                continue

            asset_preds = preds[asset_mask]
            asset_labels = valid_test.loc[asset_mask, label_col].values

            # Apply threshold
            signals = asset_preds >= threshold

            if signals.sum() == 0:
                continue

            # Trade-level metrics using continuous returns
            # fwd_profitable_1h is used only as training label;
            # for return metrics we use fwd_ret_1h (raw forward return)
            ret_col = "fwd_ret_1h"
            if ret_col in valid_test.columns:
                signal_rets = valid_test.loc[asset_mask, ret_col].values[signals]
                gross_bps = np.nanmean(signal_rets) * 10_000  # actual bps
            else:
                # Fallback to binary label if continuous return unavailable
                signal_rets = asset_labels[signals]
                gross_bps = signal_rets.mean() * 100  # hit_rate-based (legacy)

            # Deduct round-trip cost once (cost_bps is one-way)
            net_bps = gross_bps - 2 * cost_bps

            # Hit rate on the cost-adjusted binary label
            signal_labels = asset_labels[signals]
            hit_rate = signal_labels.mean() if len(signal_labels) > 0 else 0

            results.append({
                "fold": fold_id,
                "asset": asset,
                "trade_count": int(signals.sum()),
                "hit_rate": hit_rate,
                "mean_gross_bps": gross_bps,
                "mean_net_bps": net_bps,
                "profitable": net_bps > 0,
            })

    return pd.DataFrame(results) if results else pd.DataFrame()


def _aggregate_per_asset(fold_results: pd.DataFrame, pool_name: str) -> pd.DataFrame:
    """Aggregate fold-level results to per-asset summary."""
    if fold_results.empty:
        return pd.DataFrame()

    agg = fold_results.groupby("asset").agg(
        trade_count=("trade_count", "sum"),
        mean_net_bps=("mean_net_bps", "mean"),
        mean_hit_rate=("hit_rate", "mean"),
        fold_count=("fold", "nunique"),
        fold_profitability=("profitable", "mean"),
    ).reset_index()

    # Sharpe proxy: mean / std across folds
    sharpe_parts = fold_results.groupby("asset")["mean_net_bps"].agg(["mean", "std"])
    agg["sharpe"] = (sharpe_parts["mean"] / sharpe_parts["std"].replace(0, np.nan)).values

    agg["pool_name"] = pool_name
    return agg


# ── Study functions ─────────────────────────────────────────


def pooled_dilution_study(
    merged: pd.DataFrame,
    fold_df: pd.DataFrame,
    feat_cols: list[str],
    *,
    original_pool: list[str],
    full_pool: list[str],
    deploy_asset: str = "SOL-USD",
    threshold: float = 0.55,
    sep_gap: int = 3,
    cost_bps: float = 15.0,
) -> pd.DataFrame:
    """Branch A+B: Compare SOL performance trained on original vs full pool.

    Returns a comparison DataFrame with pool_name, SOL metrics.
    """
    log.info("═══ Pooled dilution study: %s vs %s → %s ═══",
             original_pool, full_pool, deploy_asset)

    results = []

    for pool_name, pool_assets in [("original_3", original_pool), ("full_8", full_pool)]:
        # Filter merged to only include pool assets
        pool_mask = merged["asset"].isin(pool_assets)
        pool_data = merged[pool_mask].copy()

        if pool_data.empty:
            log.warning("Pool %s: no data available", pool_name)
            continue

        fold_results = _run_pooled_backtest(
            pool_data, fold_df, feat_cols,
            train_assets=pool_assets,
            deploy_assets=[deploy_asset],
            threshold=threshold,
            sep_gap=sep_gap,
            cost_bps=cost_bps,
        )

        agg = _aggregate_per_asset(fold_results, pool_name)
        if not agg.empty:
            results.append(agg)

    if not results:
        log.warning("No results from pooled dilution study")
        return pd.DataFrame()

    comparison = pd.concat(results, ignore_index=True)
    log.info("Pooled dilution comparison:\n%s", comparison.to_string())
    return comparison


def family_pooled_study(
    merged: pd.DataFrame,
    fold_df: pd.DataFrame,
    feat_cols: list[str],
    *,
    family_assets: list[str],
    deploy_assets: list[str] | None = None,
    threshold: float = 0.55,
    sep_gap: int = 3,
    cost_bps: float = 15.0,
    min_trade_count: int = 30,
) -> pd.DataFrame:
    """Branch C: Train on family pool, deploy on each family member.

    Tests whether a curated 'signal family' pool preserves per-asset edge.

    Returns per-asset performance matrix.
    """
    if deploy_assets is None:
        deploy_assets = family_assets

    log.info("═══ Family-pooled study: train=%s, deploy=%s ═══",
             family_assets, deploy_assets)

    pool_mask = merged["asset"].isin(family_assets)
    pool_data = merged[pool_mask].copy()

    if pool_data.empty:
        log.warning("Family pool: no data available")
        return pd.DataFrame()

    fold_results = _run_pooled_backtest(
        pool_data, fold_df, feat_cols,
        train_assets=family_assets,
        deploy_assets=deploy_assets,
        threshold=threshold,
        sep_gap=sep_gap,
        cost_bps=cost_bps,
    )

    agg = _aggregate_per_asset(fold_results, "family_4")

    if not agg.empty:
        agg["qualifies"] = (
            (agg["mean_net_bps"] > 0) &
            (agg["trade_count"] >= min_trade_count) &
            (agg["fold_profitability"] >= 0.5)
        )

    log.info("Family-pooled results:\n%s", agg.to_string() if not agg.empty else "empty")
    return agg


def backbone_head_study(**kwargs) -> pd.DataFrame:
    """Branch D: Per-asset head on shared backbone.

    NOT IMPLEMENTED — requires architecture change.

    LightGBM does not natively support multi-task heads.
    Future options:
      1. Train shared LightGBM features → per-asset logistic head
      2. Switch to neural backbone (TabNet, FT-Transformer) with multi-head output
      3. LightGBM leaf-index features → per-asset classifier

    Returns empty DataFrame with a status note.
    """
    log.warning("Branch D (backbone+head) is NOT IMPLEMENTED — requires model architecture change")
    return pd.DataFrame([{
        "branch": "D_backbone_head",
        "status": "NOT_IMPLEMENTED",
        "note": "Requires neural backbone or LightGBM leaf-index features + per-asset classifier",
    }])


# ── Main study orchestrator ─────────────────────────────────


def dilution_study(
    merged: pd.DataFrame,
    fold_df: pd.DataFrame,
    feat_cols: list[str],
    *,
    original_pool: list[str],
    full_pool: list[str],
    family_assets: list[str],
    deploy_asset: str = "SOL-USD",
    threshold: float = 0.55,
    sep_gap: int = 3,
    cost_bps: float = 15.0,
    min_trade_count: int = 30,
    run_branches: dict[str, bool] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the full dilution study across all branches.

    Returns dict of branch results keyed by branch name.
    """
    run = run_branches or {"a": True, "b": True, "c": True, "d": False}
    results: dict[str, pd.DataFrame] = {}

    # Branches A+B: pooled dilution comparison
    if run.get("a", True) or run.get("b", True):
        dilution_cmp = pooled_dilution_study(
            merged, fold_df, feat_cols,
            original_pool=original_pool,
            full_pool=full_pool,
            deploy_asset=deploy_asset,
            threshold=threshold,
            sep_gap=sep_gap,
            cost_bps=cost_bps,
        )
        results["dilution_comparison"] = dilution_cmp

    # Branch C: family-pooled
    if run.get("c", True):
        family_result = family_pooled_study(
            merged, fold_df, feat_cols,
            family_assets=family_assets,
            threshold=threshold,
            sep_gap=sep_gap,
            cost_bps=cost_bps,
            min_trade_count=min_trade_count,
        )
        results["family_pooled"] = family_result

    # Branch D: backbone + heads (scaffold)
    if run.get("d", False):
        results["backbone_head"] = backbone_head_study()

    return results
