"""exp010 experiment runner — Family Canary Deployment Validation.

Objective: Freeze the architecture from exp009 (family-pooled training,
asset-specific deployment) and validate across the L1 family.
Also produces an exp008/exp009 reconciliation appendix.

Four deployment lanes:
  SOL — primary live paper canary
  SUI — primary live paper canary
  NEAR — secondary shadow
  APT — research-only shadow

Pipeline:
  1. Load config + rebuild family panel
  2. Train family-pooled model → generate per-asset predictions
  3. Generate per-asset simulated paper-trade logs
  4. Compute per-asset execution quality metrics
  5. Build cross-asset fill profile comparison
  6. Build family weekly PnL aggregate
  7. Build exp008/exp009 reconciliation appendix
  8. Generate report

Usage:
    python run_exp010.py
    python run_exp010.py --dry-run
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.download_crypto import run as download_data
from src.data.build_panel import build as build_panel
from src.features.build_features import build as build_features
from src.labels.build_labels import build as build_labels
from src.validation.fold_builder import build_folds
from src.utils.io import ensure_dir, load_parquet, save_parquet
from src.utils.logging import get_logger

log = get_logger("run_exp010")


def _get_feature_cols(features_df: pd.DataFrame) -> list[str]:
    exclude = {"asset", "timestamp"}
    return [c for c in features_df.columns if c not in exclude]


def main(config_path: str | None = None, dry_run: bool = False):
    if config_path is None:
        config_path = "configs/experiments/crypto_1h_exp010.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp_id = cfg["experiment_id"]
    policy_cfg = cfg.get("policy", {})
    family_cfg = cfg.get("family_pool", {})

    log.info("═══ Starting experiment: %s ═══", exp_id)
    if dry_run:
        log.info("  DRY-RUN mode: using truncated data")

    report_dir = Path(cfg["reporting"]["output_dir"])
    fig_dir = ensure_dir(report_dir / "figures")
    tbl_dir = ensure_dir(report_dir / "tables")

    # ── Step 1: Build family panel ──────────────────────────────
    log.info("── Step 1: Data pipeline (family) ──")

    expanded_data_cfg_path = cfg.get("data_config_expanded")
    if expanded_data_cfg_path and Path(expanded_data_cfg_path).exists():
        download_data(expanded_data_cfg_path)
        panel = build_panel(expanded_data_cfg_path)
        with open(expanded_data_cfg_path) as f:
            exp_data_cfg = yaml.safe_load(f)
        panel_path = Path(exp_data_cfg["panel_dir"]) / "panel.parquet"
    else:
        log.warning("No expanded data config — falling back to base data")
        download_data(cfg["data_config"])
        panel = build_panel(cfg["data_config"])
        with open(cfg["data_config"]) as f:
            base_data_cfg = yaml.safe_load(f)
        panel_path = Path(base_data_cfg["panel_dir"]) / "panel.parquet"

    # Normalize panel timestamps to tz-naive (prevents merge errors downstream)
    if hasattr(panel["timestamp"].dtype, "tz") and panel["timestamp"].dt.tz is not None:
        panel["timestamp"] = panel["timestamp"].dt.tz_localize(None)

    # ── Step 2: Features + labels ───────────────────────────────
    log.info("── Step 2: Features & labels ──")
    features = build_features(panel_path, cfg.get("feature_config"))
    labels = build_labels(
        panel_path, cfg["label_config"],
        cfg.get("backtest_config", "configs/backtests/long_flat_v1.yaml"),
    )

    # ── Step 3: Build folds ─────────────────────────────────────
    log.info("── Step 3: Walk-forward folds ──")
    val_cfg = cfg["validation"]
    fold_df = build_folds(
        timestamps=panel["timestamp"],
        train_days=val_cfg["train_days"],
        val_days=val_cfg["val_days"],
        test_days=val_cfg["test_days"],
        step_days=val_cfg["step_days"],
        embargo_bars=val_cfg["embargo_bars"],
    )

    # ── Merge features + labels ─────────────────────────────────
    merged = features.copy()
    for col in labels.columns:
        if col not in merged.columns:
            merged[col] = labels[col].values
    merged["asset"] = panel["asset"].values
    merged["timestamp"] = pd.DatetimeIndex(panel["timestamp"].values).tz_localize(None)

    for col in ("start", "end"):
        if fold_df[col].dt.tz is not None:
            fold_df[col] = fold_df[col].dt.tz_localize(None)

    feat_cols = _get_feature_cols(features)

    if dry_run:
        merged = merged.head(500)
        log.info("  DRY-RUN: truncated to %d rows", len(merged))

    available_assets = merged["asset"].unique().tolist()
    family_train = [a for a in family_cfg.get("train_assets", []) if a in available_assets]
    family_deploy = [a for a in family_cfg.get("deploy_assets", []) if a in available_assets]

    log.info("Panel: %d rows, %d features, assets: %s", len(merged), len(feat_cols), available_assets)
    log.info("Family train: %s, deploy: %s", family_train, family_deploy)

    # ── Step 4: Train family-pooled model → per-asset predictions ─
    log.info("═══ Step 4: Family-pooled training + per-asset predictions ═══")

    from src.diagnostics.dilution_diagnosis import _run_pooled_backtest

    family_mask = merged["asset"].isin(family_train)
    family_data = merged[family_mask].copy()

    fold_results = _run_pooled_backtest(
        family_data, fold_df, feat_cols,
        train_assets=family_train,
        deploy_assets=family_deploy,
        threshold=policy_cfg.get("threshold", 0.55),
        sep_gap=policy_cfg.get("sep_gap", 3),
        cost_bps=policy_cfg.get("cost_bps", 15.0),
    )

    if not fold_results.empty:
        fold_results.to_csv(tbl_dir / "family_fold_results.csv", index=False)
        log.info("Fold results: %d rows", len(fold_results))

    # Build per-asset prediction DataFrames for execution validation
    # _run_pooled_backtest already applies threshold, so we need to
    # re-run the model to get raw predictions. For now, we construct
    # prediction-like DataFrames from the family panel + fold structure.
    predictions_by_asset = _generate_predictions_by_asset(
        merged=family_data,
        fold_df=fold_df,
        feat_cols=feat_cols,
        train_assets=family_train,
        deploy_assets=family_deploy,
    )

    # ── Step 5: Execution validation ────────────────────────────
    log.info("═══ Step 5: Family execution validation ═══")

    from src.diagnostics.family_canary import (
        family_execution_validation,
        cross_asset_fill_profile,
        family_weekly_pnl,
    )

    execution_scorecard, per_asset_logs = family_execution_validation(
        panel=panel,
        predictions_by_asset=predictions_by_asset,
        policy_cfg=policy_cfg,
        family_assets=family_deploy,
    )

    if not execution_scorecard.empty:
        execution_scorecard.to_csv(tbl_dir / "execution_scorecard.csv", index=False)
        log.info("Execution scorecard saved: %d assets", len(execution_scorecard))

    for asset, asset_log in per_asset_logs.items():
        safe_name = asset.replace("-", "_").lower()
        asset_log.to_csv(tbl_dir / f"paper_trade_log_{safe_name}.csv", index=False)

    # ── Step 6: Cross-asset fill profile ────────────────────────
    log.info("═══ Step 6: Cross-asset fill profile ═══")

    fill_profile = cross_asset_fill_profile(
        per_asset_logs, reference_asset="SOL-USD",
    )
    if not fill_profile.empty:
        fill_profile.to_csv(tbl_dir / "fill_profile_comparison.csv", index=False)

    # ── Step 7: Family weekly PnL ───────────────────────────────
    log.info("═══ Step 7: Family weekly PnL ═══")

    weekly_pnl, portfolio_stats = family_weekly_pnl(
        per_asset_logs, family_assets=family_deploy,
    )
    if not weekly_pnl.empty:
        weekly_pnl.to_csv(tbl_dir / "weekly_pnl.csv", index=False)

    # ── Step 8: Reconciliation appendix ─────────────────────────
    log.info("═══ Step 8: Reconciliation appendix ═══")

    from src.diagnostics.experiment_reconciliation import build_reconciliation_table

    recon_cfg = cfg.get("reconciliation", {})
    reconciliation = build_reconciliation_table(
        exp008_config_path=recon_cfg.get("exp008_config", "configs/experiments/crypto_1h_exp008.yaml"),
        exp009_config_path=recon_cfg.get("exp009_config", "configs/experiments/crypto_1h_exp009.yaml"),
        exp008_report_dir=recon_cfg.get("exp008_report_dir", "reports/exp008"),
        exp009_report_dir=recon_cfg.get("exp009_report_dir", "reports/exp009"),
    )
    if not reconciliation.empty:
        reconciliation.to_csv(tbl_dir / "reconciliation_exp008_exp009.csv", index=False)

    # ── Step 9: Generate report ─────────────────────────────────
    log.info("═══ Step 9: Generating report ═══")

    from src.reporting.exp010_report import build_exp010_summary, generate_all_plots

    generate_all_plots(
        fig_dir=fig_dir,
        execution_scorecard=execution_scorecard,
        fill_profile=fill_profile,
        weekly_pnl=weekly_pnl,
        per_asset_logs=per_asset_logs,
    )

    summary_path = build_exp010_summary(
        report_dir=report_dir,
        execution_scorecard=execution_scorecard,
        fill_profile=fill_profile,
        weekly_pnl=weekly_pnl,
        portfolio_stats=portfolio_stats,
        reconciliation=reconciliation,
        fold_results=fold_results,
        cfg=cfg,
        go_no_go=cfg.get("go_no_go", {}),
    )
    log.info("═══ Report generated: %s ═══", summary_path)

    return execution_scorecard


def _generate_predictions_by_asset(
    *,
    merged: pd.DataFrame,
    fold_df: pd.DataFrame,
    feat_cols: list[str],
    train_assets: list[str],
    deploy_assets: list[str],
) -> dict[str, pd.DataFrame]:
    """Generate raw prediction DataFrames per-asset from family-pooled training.

    Returns dict mapping asset → DataFrame with y_pred_prob, asset, timestamp columns.
    """
    from src.models.train_lightgbm import train as train_lightgbm

    all_preds: dict[str, list[pd.DataFrame]] = {a: [] for a in deploy_assets}

    if "fold_id" in fold_df.columns:
        fold_ids = fold_df["fold_id"].unique()
    else:
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
            test_end = None

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

        label_col = "fwd_profitable_1h"
        if label_col not in train_data.columns or len(train_data) < 50 or len(test_data) < 10:
            continue

        valid_train = train_data.dropna(subset=[label_col])
        valid_test = test_data.dropna(subset=[label_col])

        available_feats = [c for c in feat_cols if c in valid_train.columns]
        if not available_feats:
            continue

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

        # Split predictions by deploy asset
        for asset in deploy_assets:
            asset_mask = valid_test["asset"] == asset
            if asset_mask.sum() == 0:
                continue

            asset_df = pd.DataFrame({
                "y_pred_prob": preds[asset_mask],
                "asset": asset,
                "timestamp": valid_test.loc[asset_mask, "timestamp"].values,
                "model_name": "lightgbm",
                "fold_id": fold_id,
            })
            all_preds[asset].append(asset_df)

    result = {}
    for asset in deploy_assets:
        if all_preds[asset]:
            result[asset] = pd.concat(all_preds[asset], ignore_index=True)
            log.info("Generated %d predictions for %s", len(result[asset]), asset)
        else:
            log.warning("No predictions generated for %s", asset)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="exp010: Family Canary Validation")
    parser.add_argument(
        "config", nargs="?",
        default="configs/experiments/crypto_1h_exp010.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run with truncated data for pipeline validation",
    )
    args = parser.parse_args()
    main(config_path=args.config, dry_run=args.dry_run)
