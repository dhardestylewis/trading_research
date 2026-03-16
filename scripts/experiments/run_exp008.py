"""exp008 experiment runner — Capacity Research.

Objective: Shift from "does signal exist?" to "can we increase expected
net PnL per week?" by jointly optimizing edge, trade frequency, and
deployable notional.

Five research branches:
  A: Multi-horizon stacking (1h/2h/4h parallel policies)
  B: Universe expansion (SOL-family transportability)
  C: Score calibration (ranking quality, monotone ordering)
  D: Execution-aware alpha (slippage-adjusted targets)
  E: Capacity economics (unified $/week metric)

Pipeline:
  1. Load config + re-use exp001 predictions as pooled baseline
  2. Rebuild panel, features, labels, folds
  3. Run branches A–D independently (skippable via config)
  4. Run Branch E (integrates A–D outputs)
  5. Generate report

Usage:
    python run_exp008.py
    python run_exp008.py --dry-run
    python run_exp008.py configs/experiments/crypto_1h_exp008.yaml
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
from src.models.predict import TrainedModel
from src.utils.io import ensure_dir, load_parquet, save_parquet
from src.utils.logging import get_logger

log = get_logger("run_exp008")


def _get_feature_cols(features_df: pd.DataFrame) -> list[str]:
    exclude = {"asset", "timestamp"}
    return [c for c in features_df.columns if c not in exclude]


def main(config_path: str | None = None, dry_run: bool = False):
    if config_path is None:
        config_path = "configs/experiments/crypto_1h_exp008.yaml"

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
    with open(cfg["backtest_config"]) as f:
        bt_cfg = yaml.safe_load(f)

    # ── Step 1: Download + build panel ──────────────────────
    log.info("── Step 1: Data pipeline ──")
    download_data(cfg["data_config"])
    panel = build_panel(cfg["data_config"])

    # ── Step 2: Features + labels ───────────────────────────
    log.info("── Step 2: Features & labels ──")
    panel_path = Path(data_cfg["panel_dir"]) / "panel.parquet"
    features = build_features(panel_path, cfg.get("feature_config"))
    labels = build_labels(panel_path, cfg["label_config"], cfg["backtest_config"])

    # Also build multi-horizon labels if available
    multi_label_cfg = cfg.get("label_config_multi")
    if multi_label_cfg and Path(multi_label_cfg).exists():
        labels_multi = build_labels(panel_path, multi_label_cfg, cfg["backtest_config"])
    else:
        labels_multi = labels

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
    # Add multi-horizon labels
    for col in labels_multi.columns:
        if col not in merged.columns:
            merged[col] = labels_multi[col].values
    # Add standard labels if missing
    for col in labels.columns:
        if col not in merged.columns:
            merged[col] = labels[col].values

    merged["asset"] = panel["asset"].values
    merged["timestamp"] = pd.DatetimeIndex(panel["timestamp"].values).tz_localize(None)

    for col in ("start", "end"):
        if fold_df[col].dt.tz is not None:
            fold_df[col] = fold_df[col].dt.tz_localize(None)

    feat_cols = _get_feature_cols(features)
    target_asset = cfg.get("target_asset", "SOL-USD")

    if dry_run:
        # Truncate for fast validation
        merged = merged.head(500)
        log.info("  DRY-RUN: truncated to %d rows", len(merged))

    # ── Load exp001 predictions for reference ───────────────
    exp001_path = cfg.get("exp001_predictions")
    pooled_preds = None
    if exp001_path and Path(exp001_path).exists():
        pooled_preds = load_parquet(exp001_path)
        if "timestamp" in pooled_preds.columns:
            pooled_preds["timestamp"] = pd.DatetimeIndex(pooled_preds["timestamp"].values).tz_localize(None)
        log.info("Loaded exp001 predictions: %d rows", len(pooled_preds))

    log.info("Panel: %d rows, %d features, assets: %s",
             len(merged), len(feat_cols), merged["asset"].unique().tolist())

    # ── Branch results collectors ───────────────────────────
    horizon_result = None
    universe_result = None
    calibration_curve = None
    rank_quality = None
    cs_ranking = None
    exec_aware_result = None

    # ────────────────────────────────────────────────────────
    # BRANCH A: Multi-horizon stacking
    # ────────────────────────────────────────────────────────
    if branches.get("run_a", True):
        log.info("═══ Running Branch A: Multi-horizon stacking ═══")
        from src.diagnostics.multi_horizon_stacking import horizon_stacking_study

        mh_cfg = cfg.get("multi_horizon", {})
        horizon_result, stacked_trades = horizon_stacking_study(
            merged, fold_df, feat_cols,
            pooled_preds=pooled_preds,
            target_asset=target_asset,
            horizons=mh_cfg.get("horizons", [1, 2, 4]),
            threshold=mh_cfg.get("threshold", policy_cfg.get("threshold", 0.55)),
            sep_gap=mh_cfg.get("sep_gap", policy_cfg.get("sep_gap", 3)),
            cost_bps=mh_cfg.get("cost_bps", policy_cfg.get("cost_bps", 15.0)),
            dedup_window=mh_cfg.get("dedup_window_bars", 2),
        )
        horizon_result.to_csv(tbl_dir / "horizon_stacking.csv", index=False)
        if not stacked_trades.empty:
            stacked_trades.to_csv(tbl_dir / "stacked_trades.csv", index=False)
        log.info("Branch A complete")

    # ────────────────────────────────────────────────────────
    # BRANCH B: Universe expansion
    # ────────────────────────────────────────────────────────
    if branches.get("run_b", True):
        log.info("═══ Running Branch B: Universe expansion ═══")
        from src.diagnostics.universe_expansion import universe_expansion_study

        ue_cfg = cfg.get("universe_expansion", {})
        expansion_assets = ue_cfg.get("expansion_assets", [])

        # Check which expansion assets are already in our data
        available_assets = merged["asset"].unique().tolist()
        valid_expansion = [a for a in expansion_assets if a in available_assets]

        # If expansion assets aren't in the core panel, build an expanded panel
        expanded_data_cfg = cfg.get("data_config_expanded")
        if not valid_expansion and expanded_data_cfg and Path(expanded_data_cfg).exists():
            log.info("Building expanded panel from %s", expanded_data_cfg)
            with open(expanded_data_cfg) as f:
                exp_data_cfg = yaml.safe_load(f)

            # Download if needed (skips if already cached)
            download_data(expanded_data_cfg)
            panel_exp = build_panel(expanded_data_cfg)

            # Build features + labels on expanded panel
            exp_panel_path = Path(exp_data_cfg["panel_dir"]) / "panel.parquet"
            features_exp = build_features(exp_panel_path, cfg.get("feature_config"))
            labels_exp = build_labels(exp_panel_path, cfg["label_config"], cfg["backtest_config"])

            # Build folds on expanded panel
            fold_df_exp = build_folds(
                timestamps=panel_exp["timestamp"],
                train_days=val_cfg["train_days"],
                val_days=val_cfg["val_days"],
                test_days=val_cfg["test_days"],
                step_days=val_cfg["step_days"],
                embargo_bars=val_cfg["embargo_bars"],
            )

            # Merge expanded panel
            merged_exp = features_exp.copy()
            for col in labels_exp.columns:
                if col not in merged_exp.columns:
                    merged_exp[col] = labels_exp[col].values
            merged_exp["asset"] = panel_exp["asset"].values
            merged_exp["timestamp"] = pd.DatetimeIndex(panel_exp["timestamp"].values).tz_localize(None)
            for col in ("start", "end"):
                if fold_df_exp[col].dt.tz is not None:
                    fold_df_exp[col] = fold_df_exp[col].dt.tz_localize(None)

            feat_cols_exp = _get_feature_cols(features_exp)
            available_assets = merged_exp["asset"].unique().tolist()
            valid_expansion = [a for a in expansion_assets if a in available_assets]
            log.info("Expanded panel: %d rows, assets: %s", len(merged_exp), available_assets)

            if valid_expansion:
                if dry_run:
                    merged_exp = merged_exp.head(500)
                universe_result = universe_expansion_study(
                    merged_exp, fold_df_exp, feat_cols_exp,
                    expansion_assets=valid_expansion,
                    threshold=ue_cfg.get("threshold", policy_cfg.get("threshold", 0.55)),
                    sep_gap=ue_cfg.get("sep_gap", policy_cfg.get("sep_gap", 3)),
                    cost_bps=ue_cfg.get("cost_bps", policy_cfg.get("cost_bps", 15.0)),
                    min_sharpe=ue_cfg.get("min_sharpe", 0.5),
                    min_fold_profitability=ue_cfg.get("min_fold_profitability", 0.50),
                )
                universe_result.to_csv(tbl_dir / "universe_expansion.csv", index=False)
                log.info("Branch B complete")
            else:
                log.warning("Branch B: expansion assets still not available after download")
        elif valid_expansion:
            universe_result = universe_expansion_study(
                merged, fold_df, feat_cols,
                expansion_assets=valid_expansion,
                threshold=ue_cfg.get("threshold", policy_cfg.get("threshold", 0.55)),
                sep_gap=ue_cfg.get("sep_gap", policy_cfg.get("sep_gap", 3)),
                cost_bps=ue_cfg.get("cost_bps", policy_cfg.get("cost_bps", 15.0)),
                min_sharpe=ue_cfg.get("min_sharpe", 0.5),
                min_fold_profitability=ue_cfg.get("min_fold_profitability", 0.50),
            )
            universe_result.to_csv(tbl_dir / "universe_expansion.csv", index=False)
            log.info("Branch B complete")
        else:
            log.warning("Branch B skipped: no expansion assets and no expanded data config")
            log.info("  Set data_config_expanded in exp008 config to enable")

    # ────────────────────────────────────────────────────────
    # BRANCH C: Score calibration
    # ────────────────────────────────────────────────────────
    if branches.get("run_c", True):
        log.info("═══ Running Branch C: Score calibration ═══")
        from src.diagnostics.score_calibration import calibration_study

        sc_cfg = cfg.get("score_calibration", {})

        # Use exp001 predictions if available, else generate fresh
        if pooled_preds is not None and not pooled_preds.empty:
            cal_preds = pooled_preds.copy()
        else:
            log.warning("No exp001 predictions — skipping Branch C")
            cal_preds = None

        if cal_preds is not None:
            calibration_curve, rank_quality, cs_ranking = calibration_study(
                cal_preds,
                n_bins=sc_cfg.get("n_bins", 20),
                top_k=sc_cfg.get("top_k", 3),
                cost_bps=policy_cfg.get("cost_bps", 15.0),
            )
            calibration_curve.to_csv(tbl_dir / "calibration_curve.csv", index=False)
            rank_quality.to_csv(tbl_dir / "rank_quality.csv", index=False)
            if not cs_ranking.empty:
                cs_ranking.to_csv(tbl_dir / "cross_sectional_ranking.csv", index=False)
            log.info("Branch C complete")

    # ────────────────────────────────────────────────────────
    # BRANCH D: Execution-aware alpha
    # ────────────────────────────────────────────────────────
    if branches.get("run_d", True):
        log.info("═══ Running Branch D: Execution-aware alpha ═══")
        from src.diagnostics.execution_aware_alpha import execution_aware_study

        ea_cfg = cfg.get("execution_aware", {})
        exec_aware_result = execution_aware_study(
            merged, fold_df, feat_cols,
            target_asset=target_asset,
            slippage_scenarios_bps=ea_cfg.get("slippage_scenarios_bps", [0, 5, 10, 15, 20]),
            threshold=ea_cfg.get("threshold", policy_cfg.get("threshold", 0.55)),
            sep_gap=ea_cfg.get("sep_gap", policy_cfg.get("sep_gap", 3)),
            cost_bps=policy_cfg.get("cost_bps", 15.0),
        )
        exec_aware_result.to_csv(tbl_dir / "execution_aware.csv", index=False)
        log.info("Branch D complete")

    # ────────────────────────────────────────────────────────
    # BRANCH E: Capacity economics (always runs)
    # ────────────────────────────────────────────────────────
    log.info("═══ Running Branch E: Capacity economics ═══")
    from src.diagnostics.capacity_economics import capacity_economics_study

    ce_cfg = cfg.get("capacity_economics", {})
    scorecard, grid = capacity_economics_study(
        baseline_edge_bps=ce_cfg.get("baseline_edge_bps", 44.0),
        baseline_trades_per_week=ce_cfg.get("baseline_trades_per_week", 4.0),
        horizon_result=horizon_result,
        universe_result=universe_result,
        exec_aware_result=exec_aware_result,
        notional=ce_cfg.get("notional_levels", [100_000])[1] if len(ce_cfg.get("notional_levels", [])) > 1 else 100_000,
        cost_bps=policy_cfg.get("cost_bps", 15.0),
        notional_levels=ce_cfg.get("notional_levels", [50_000, 100_000, 250_000, 500_000]),
        slippage_levels_bps=ce_cfg.get("slippage_levels_bps", [5, 10, 15, 20]),
        frequency_multipliers=ce_cfg.get("frequency_multipliers", [1, 2, 5, 10]),
    )
    scorecard.to_csv(tbl_dir / "capacity_scorecard.csv", index=False)
    grid.to_csv(tbl_dir / "sensitivity_grid.csv", index=False)
    log.info("Branch E complete")

    # ────────────────────────────────────────────────────────
    # Generate report
    # ────────────────────────────────────────────────────────
    log.info("═══ Generating report ═══")
    from src.reporting.exp008_report import build_exp008_summary, generate_all_plots

    generate_all_plots(
        fig_dir=fig_dir,
        horizon_result=horizon_result,
        universe_result=universe_result,
        calibration_curve=calibration_curve,
        rank_quality=rank_quality,
        cs_ranking=cs_ranking,
        exec_aware_result=exec_aware_result,
        scorecard=scorecard,
        grid=grid,
    )

    summary_path = build_exp008_summary(
        report_dir=report_dir,
        horizon_result=horizon_result,
        universe_result=universe_result,
        calibration_curve=calibration_curve,
        rank_quality=rank_quality,
        cs_ranking=cs_ranking,
        exec_aware_result=exec_aware_result,
        scorecard=scorecard,
        grid=grid,
        cfg=cfg,
        go_no_go=cfg.get("go_no_go", {}),
    )
    log.info("═══ Report generated: %s ═══", summary_path)

    return scorecard


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="exp008: Capacity Research")
    parser.add_argument(
        "config", nargs="?",
        default="configs/experiments/crypto_1h_exp008.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run with truncated data for pipeline validation",
    )
    args = parser.parse_args()
    main(config_path=args.config, dry_run=args.dry_run)
