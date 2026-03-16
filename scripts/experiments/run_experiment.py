"""Master experiment orchestrator.

Usage:
    python run_experiment.py configs/experiments/crypto_1h_exp001.yaml
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.download_crypto import run as download_data
from src.data.build_panel import build as build_panel
from src.features.build_features import build as build_features
from src.labels.build_labels import build as build_labels
from src.validation.fold_builder import build_folds, save_fold_definitions
from src.models.predict import TrainedModel, NaiveMomentumModel
from src.models.train_logistic import train as train_logistic
from src.models.train_lightgbm import train as train_lightgbm
from src.backtest.simulator import simulate_all
from src.backtest.metrics import trading_metrics
from src.reporting.tables import save_all_tables
from src.reporting.plots import (
    plot_equity_curves,
    plot_drawdown,
    plot_calibration,
    plot_feature_importance,
    plot_threshold_sensitivity,
)
from src.reporting.build_report import build as build_report
from src.utils.io import save_parquet, load_parquet, ensure_dir
from src.utils.logging import get_logger

log = get_logger("run_experiment")


def _get_feature_cols(features_df: pd.DataFrame) -> list[str]:
    """Return numeric feature column names (exclude identifiers)."""
    exclude = {"asset", "timestamp"}
    return [c for c in features_df.columns if c not in exclude]


def main(config_path: str | None = None):
    if config_path is None:
        config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/experiments/crypto_1h_exp001.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp_id = cfg["experiment_id"]
    log.info("═══ Starting experiment: %s ═══", exp_id)

    # ── Load sub-configs ─────────────────────────────────────────
    with open(cfg["data_config"]) as f:
        data_cfg = yaml.safe_load(f)
    with open(cfg["backtest_config"]) as f:
        bt_cfg = yaml.safe_load(f)

    report_dir = Path(cfg["reporting"]["output_dir"])
    fig_dir = ensure_dir(report_dir / "figures")

    # ── Step 1: Download data ────────────────────────────────────
    log.info("── Step 1: Downloading data ──")
    download_data(cfg["data_config"])

    # ── Step 2: Build panel ──────────────────────────────────────
    log.info("── Step 2: Building panel ──")
    panel = build_panel(cfg["data_config"])

    # ── Step 3: Build features ───────────────────────────────────
    log.info("── Step 3: Building features ──")
    panel_path = Path(data_cfg["panel_dir"]) / "panel.parquet"
    features = build_features(panel_path, cfg.get("feature_config"))

    # ── Step 4: Build labels ─────────────────────────────────────
    log.info("── Step 4: Building labels ──")
    labels = build_labels(panel_path, cfg["label_config"], cfg["backtest_config"])

    # ── Step 5: Build folds ──────────────────────────────────────
    log.info("── Step 5: Building walk-forward folds ──")
    val_cfg = cfg["validation"]
    fold_df = build_folds(
        timestamps=panel["timestamp"],
        train_days=val_cfg["train_days"],
        val_days=val_cfg["val_days"],
        test_days=val_cfg["test_days"],
        step_days=val_cfg["step_days"],
        embargo_bars=val_cfg["embargo_bars"],
    )
    save_fold_definitions(fold_df)
    log.info("  %d folds generated", fold_df["fold_id"].nunique())

    # ── Merge features + labels ──────────────────────────────────
    merged = features.merge(labels.drop(columns=["asset", "timestamp"]),
                            left_index=True, right_index=True)
    merged["asset"] = panel["asset"].values
    merged["timestamp"] = pd.DatetimeIndex(panel["timestamp"].values).tz_localize(None)

    # Normalize fold boundaries to tz-naive for safe comparison
    for col in ("start", "end"):
        if fold_df[col].dt.tz is not None:
            fold_df[col] = fold_df[col].dt.tz_localize(None)

    feat_cols = _get_feature_cols(features)
    target_col = "fwd_profitable_1h"  # primary classification target
    ret_col = "fwd_ret_1h"  # for P&L accounting

    log.info("Feature cols: %d, target: %s", len(feat_cols), target_col)


    # ── Step 6: Train & predict per fold ─────────────────────────
    log.info("── Step 6: Training & predicting ──")
    thresholds = bt_cfg["thresholds"]
    cost_regimes = list(bt_cfg["costs"].keys())

    all_predictions: list[pd.DataFrame] = []

    for fold_id in sorted(fold_df["fold_id"].unique()):
        fold_rows = fold_df[fold_df["fold_id"] == fold_id]
        train_r = fold_rows[fold_rows["split"] == "train"].iloc[0]
        val_r = fold_rows[fold_rows["split"] == "val"].iloc[0]
        test_r = fold_rows[fold_rows["split"] == "test"].iloc[0]

        ts = merged["timestamp"]
        train_mask = (ts >= train_r["start"]) & (ts < train_r["end"])
        val_mask = (ts >= val_r["start"]) & (ts < val_r["end"])
        test_mask = (ts >= test_r["start"]) & (ts < test_r["end"])

        train_df = merged[train_mask].copy()
        val_df = merged[val_mask].copy()
        test_df = merged[test_mask].copy()

        # Drop rows where target or features are NaN
        for df_ in (train_df, val_df, test_df):
            valid = df_[feat_cols + [target_col, ret_col]].notna().all(axis=1)
            df_.drop(df_[~valid].index, inplace=True)

        if len(train_df) < 100 or len(test_df) < 10:
            log.warning("  Fold %d skipped: train=%d, test=%d", fold_id, len(train_df), len(test_df))
            continue

        log.info("  Fold %d: train=%d, val=%d, test=%d", fold_id, len(train_df), len(val_df), len(test_df))

        # ── Model configs ────────────────────────────────────────
        model_cfgs = cfg["models"]
        for mcfg in model_cfgs:
            model_name = mcfg["name"]

            if model_name == "naive_momentum":
                naive = NaiveMomentumModel()
                probs = naive.predict_proba(test_df[feat_cols])
            elif model_name == "logistic_regression":
                tm = train_logistic(
                    train_df[feat_cols], train_df[target_col],
                    val_df[feat_cols], val_df[target_col],
                    config_path=mcfg.get("config"),
                    feature_names=feat_cols,
                )
                probs = tm.predict_proba(test_df)
            elif model_name == "lightgbm":
                tm = train_lightgbm(
                    train_df[feat_cols], train_df[target_col],
                    val_df[feat_cols], val_df[target_col],
                    config_path=mcfg.get("config"),
                    feature_names=feat_cols,
                )
                probs = tm.predict_proba(test_df)
            else:
                log.warning("  Unknown model: %s — skipping", model_name)
                continue

            pred_df = test_df[["asset", "timestamp", target_col, ret_col]].copy()
            pred_df = pred_df.rename(columns={target_col: "y_true", ret_col: "fwd_ret_1h"})
            pred_df["y_pred_prob"] = probs
            pred_df["model_name"] = model_name
            pred_df["fold_id"] = fold_id
            all_predictions.append(pred_df)

    predictions = pd.concat(all_predictions, ignore_index=True)
    log.info("Total predictions: %d rows", len(predictions))

    # ── Step 7: Simulate backtests ───────────────────────────────
    log.info("── Step 7: Running backtests ──")
    all_sim_parts: list[pd.DataFrame] = []

    for model_name, mgrp in predictions.groupby("model_name"):
        sim = simulate_all(
            mgrp,
            prob_col="y_pred_prob",
            thresholds=thresholds,
            cost_regimes=cost_regimes,
            asset_ret_col="fwd_ret_1h",
            cost_overrides=bt_cfg["costs"],
        )
        sim["model_name"] = model_name
        all_sim_parts.append(sim)

    all_sim = pd.concat(all_sim_parts, ignore_index=True)

    # ── Step 8: Generate reports ─────────────────────────────────
    log.info("── Step 8: Generating reports ──")

    # Tables
    table_paths = save_all_tables(all_sim, report_dir, y_true_col="y_true", y_prob_col="y_pred_prob")
    log.info("  Tables saved: %s", list(table_paths.keys()))

    # Plots
    # Equity curves per model (base cost, threshold 0.55)
    for model_name in predictions["model_name"].unique():
        msim = all_sim[(all_sim["model_name"] == model_name) & (all_sim["threshold"] == 0.55)]
        if not msim.empty:
            plot_equity_curves(msim, fig_dir, model_name=model_name)
            plot_drawdown(msim, fig_dir, model_name=model_name)

    # Calibration for ML models
    for model_name in ["logistic_regression", "lightgbm"]:
        mp = predictions[predictions["model_name"] == model_name]
        mask = mp["y_true"].notna() & mp["y_pred_prob"].notna()
        if mask.sum() > 50:
            plot_calibration(mp.loc[mask, "y_true"].values, mp.loc[mask, "y_pred_prob"].values, fig_dir, model_name)

    # Feature importance for LightGBM (use last fold's model)
    if "lightgbm" in [m["name"] for m in cfg["models"]]:
        try:
            # Retrain on full last fold to get feature importances for the report
            last_fold = sorted(fold_df["fold_id"].unique())[-1]
            fold_rows = fold_df[fold_df["fold_id"] == last_fold]
            train_r = fold_rows[fold_rows["split"] == "train"].iloc[0]
            ts = merged["timestamp"]
            train_mask = (ts >= train_r["start"]) & (ts < train_r["end"])
            train_df = merged[train_mask].dropna(subset=feat_cols + [target_col])
            lgb_model = train_lightgbm(train_df[feat_cols], train_df[target_col],
                                       config_path="configs/models/lightgbm_v1.yaml",
                                       feature_names=feat_cols)
            plot_feature_importance(lgb_model.model, feat_cols, fig_dir)
        except Exception as e:
            log.warning("Could not plot feature importance: %s", e)

    # Threshold sensitivity
    mc = pd.read_csv(report_dir / "tables" / "model_comparison.csv")
    plot_threshold_sensitivity(mc, fig_dir)

    # Save predictions
    if cfg["reporting"].get("save_predictions", True):
        pred_dir = ensure_dir("data/artifacts/predictions")
        save_parquet(predictions, pred_dir / f"{exp_id}_predictions.parquet")

    # Build markdown report
    summary_path = build_report(report_dir)
    log.info("═══ Report generated: %s ═══", summary_path)


if __name__ == "__main__":
    main()
