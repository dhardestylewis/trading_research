"""exp009 experiment runner — Dilution Study.

Objective: Investigate whether expanding the training pool
contaminates the SOL edge. Determine optimal training pool
composition to preserve SOL while adding transportable assets.

Four research branches:
  A: Original 3-asset pool (SOL/BTC/ETH) → SOL deploy
  B: Full 8-asset pool → SOL deploy
  C: Family pool (SOL/APT/SUI/NEAR) → per-asset deploy
  D: Per-asset head on shared backbone (scaffold, not implemented)

Pipeline:
  1. Load config + rebuild expanded panel
  2. Run dilution study (Branches A-C)
  3. Scaffold Branch D status
  4. Generate report

Usage:
    python run_exp009.py
    python run_exp009.py --dry-run
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

log = get_logger("run_exp009")


def _get_feature_cols(features_df: pd.DataFrame) -> list[str]:
    exclude = {"asset", "timestamp"}
    return [c for c in features_df.columns if c not in exclude]


def main(config_path: str | None = None, dry_run: bool = False):
    if config_path is None:
        config_path = "configs/experiments/crypto_1h_exp009.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp_id = cfg["experiment_id"]
    branches = cfg.get("branches", {})
    policy_cfg = cfg.get("policy", {})

    log.info("═══ Starting experiment: %s ═══", exp_id)
    if dry_run:
        log.info("  DRY-RUN mode: using truncated data")

    report_dir = Path(cfg["reporting"]["output_dir"])
    fig_dir = ensure_dir(report_dir / "figures")
    tbl_dir = ensure_dir(report_dir / "tables")

    # ── Load sub-configs ────────────────────────────────────
    with open(cfg["data_config"]) as f:
        data_cfg = yaml.safe_load(f)

    # ── Step 1: Build expanded panel ────────────────────────
    log.info("── Step 1: Data pipeline (expanded) ──")

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
        panel_path = Path(data_cfg["panel_dir"]) / "panel.parquet"

    # ── Step 2: Features + labels ───────────────────────────
    log.info("── Step 2: Features & labels ──")
    features = build_features(panel_path, cfg.get("feature_config"))
    labels = build_labels(panel_path, cfg["label_config"], cfg.get("backtest_config", "configs/backtests/long_flat_v1.yaml"))

    # ── Step 3: Build folds ─────────────────────────────────
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

    # ── Merge features + labels ─────────────────────────────
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
    log.info("Panel: %d rows, %d features, assets: %s",
             len(merged), len(feat_cols), available_assets)

    # ── Extract pool configs ────────────────────────────────
    pool_orig_cfg = cfg.get("pool_original", {})
    pool_full_cfg = cfg.get("pool_full", {})
    pool_family_cfg = cfg.get("pool_family", {})

    original_pool = [a for a in pool_orig_cfg.get("assets", ["SOL-USD", "BTC-USD", "ETH-USD"])
                     if a in available_assets]
    full_pool = [a for a in pool_full_cfg.get("assets", []) if a in available_assets]
    family_assets = [a for a in pool_family_cfg.get("assets", []) if a in available_assets]
    family_deploy = [a for a in pool_family_cfg.get("deploy_on", family_assets)
                     if a in available_assets]

    log.info("Original pool: %s", original_pool)
    log.info("Full pool: %s", full_pool)
    log.info("Family pool: %s (deploy: %s)", family_assets, family_deploy)

    # ── Run dilution study ──────────────────────────────────
    log.info("═══ Running dilution study ═══")
    from src.diagnostics.dilution_diagnosis import dilution_study

    run_map = {
        "a": branches.get("run_a", True),
        "b": branches.get("run_b", True),
        "c": branches.get("run_c", True),
        "d": branches.get("run_d", False),
    }

    study_results = dilution_study(
        merged, fold_df, feat_cols,
        original_pool=original_pool,
        full_pool=full_pool,
        family_assets=family_assets,
        deploy_asset=cfg.get("target_asset", "SOL-USD"),
        threshold=policy_cfg.get("threshold", 0.55),
        sep_gap=policy_cfg.get("sep_gap", 3),
        cost_bps=policy_cfg.get("cost_bps", 15.0),
        min_trade_count=cfg.get("go_no_go", {}).get("min_trade_count", 30),
        run_branches=run_map,
    )

    # Save tables
    for name, df in study_results.items():
        if df is not None and not df.empty:
            df.to_csv(tbl_dir / f"{name}.csv", index=False)
            log.info("Saved table: %s (%d rows)", name, len(df))

    # ── Generate report ─────────────────────────────────────
    log.info("═══ Generating report ═══")
    from src.reporting.exp009_report import build_exp009_summary, generate_all_plots

    dilution_cmp = study_results.get("dilution_comparison")
    family_result = study_results.get("family_pooled")
    backbone_result = study_results.get("backbone_head")

    generate_all_plots(
        fig_dir=fig_dir,
        dilution_cmp=dilution_cmp,
        family_result=family_result,
    )

    summary_path = build_exp009_summary(
        report_dir=report_dir,
        dilution_cmp=dilution_cmp,
        family_result=family_result,
        backbone_result=backbone_result,
        cfg=cfg,
        go_no_go=cfg.get("go_no_go", {}),
    )
    log.info("═══ Report generated: %s ═══", summary_path)

    return study_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="exp009: Dilution Study")
    parser.add_argument(
        "config", nargs="?",
        default="configs/experiments/crypto_1h_exp009.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run with truncated data for pipeline validation",
    )
    args = parser.parse_args()
    main(config_path=args.config, dry_run=args.dry_run)
