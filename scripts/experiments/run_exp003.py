"""exp003 experiment runner — SOL-only execution-validity experiment.

Usage:
    python run_exp003.py                                    # full run
    python run_exp003.py configs/experiments/crypto_1h_exp003.yaml
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.data.download_crypto import run as download_data
from src.data.build_panel import build as build_panel
from src.features.build_features import build as build_features
from src.labels.build_labels import build as build_labels
from src.validation.fold_builder import build_folds
from src.diagnostics.regime_labeller import label_regimes
from src.diagnostics.regime_performance import regime_conditional_metrics
from src.diagnostics.fold_attribution import compute_fold_descriptors, regress_fold_pnl
from src.diagnostics.sol_robustness import sol_only_robustness
from src.diagnostics.sol_horizon_study import run_sol_horizon_study
from src.diagnostics.cost_sensitivity import cost_sensitivity_grid
from src.backtest.sparse_policy import evaluate_sparse_policies
from src.reporting.exp003_report import (
    build_summary, plot_fill_comparison, plot_horizon_delay_heatmap, plot_cost_sensitivity,
)
from src.utils.io import load_parquet, save_parquet, ensure_dir
from src.utils.logging import get_logger

log = get_logger("run_exp003")


def _load_features_with_identifiers(data_cfg_path: str) -> pd.DataFrame:
    """Load features parquet and merge back asset + timestamp from the panel."""
    with open(data_cfg_path) as f:
        data_cfg = yaml.safe_load(f)
    panel_path = Path(data_cfg["panel_dir"]) / "panel.parquet"
    feat_path = Path("data/processed/features/features.parquet")

    panel = load_parquet(panel_path)
    features = load_parquet(feat_path)

    features["asset"] = panel["asset"].values
    features["timestamp"] = pd.DatetimeIndex(panel["timestamp"].values).tz_localize(None)
    return features


def _get_feature_cols(features_df: pd.DataFrame) -> list[str]:
    """Return numeric feature column names (exclude identifiers)."""
    exclude = {"asset", "timestamp"}
    return [c for c in features_df.columns if c not in exclude]


def main(config_path: str | None = None):
    if config_path is None:
        config_path = (
            sys.argv[1]
            if len(sys.argv) > 1 and not sys.argv[1].startswith("--")
            else "configs/experiments/crypto_1h_exp003.yaml"
        )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp_id = cfg["experiment_id"]
    target_asset = cfg.get("target_asset", "SOL-USD")
    target_model = cfg.get("target_model", "lightgbm")
    log.info("═══ Starting experiment: %s ═══", exp_id)
    log.info("Target: %s / %s", target_asset, target_model)

    report_dir = Path(cfg["reporting"]["output_dir"])
    fig_dir = ensure_dir(report_dir / "figures")
    tbl_dir = ensure_dir(report_dir / "tables")

    # ── Load data configs ────────────────────────────────────────
    with open(cfg["data_config"]) as f:
        data_cfg = yaml.safe_load(f)
    with open(cfg["backtest_config"]) as f:
        bt_cfg = yaml.safe_load(f)

    # ── Load panel (needed for fill simulation) ──────────────────
    panel_path = Path(data_cfg["panel_dir"]) / "panel.parquet"
    if not panel_path.exists():
        log.info("Panel not found — downloading data and building panel…")
        download_data(cfg["data_config"])
        build_panel(cfg["data_config"])
    panel = load_parquet(panel_path)
    # Normalize timestamps
    if hasattr(panel["timestamp"].dtype, "tz") and panel["timestamp"].dt.tz is not None:
        panel["timestamp"] = panel["timestamp"].dt.tz_localize(None)

    # ── Load exp001 predictions ──────────────────────────────────
    pred_path = cfg.get("exp001_predictions", "data/artifacts/predictions/crypto_1h_exp001_predictions.parquet")
    log.info("Loading predictions from %s", pred_path)
    preds = load_parquet(pred_path)
    log.info("Loaded %d prediction rows", len(preds))

    # Ensure tz-naive timestamps
    if hasattr(preds["timestamp"].dtype, "tz") and preds["timestamp"].dt.tz is not None:
        preds["timestamp"] = preds["timestamp"].dt.tz_localize(None)

    # ── Load features for regime labelling ────────────────────────
    log.info("Loading features…")
    features = _load_features_with_identifiers(cfg["data_config"])

    # ── Load labels (for multi-horizon study) ────────────────────
    label_path = Path("data/processed/labels/labels.parquet")
    if not label_path.exists():
        log.info("Labels not found — building…")
        build_labels(panel_path, cfg["label_config"], cfg["backtest_config"])
    labels = load_parquet(label_path)

    # ── Load fold definitions ────────────────────────────────────
    fold_path = Path("data/artifacts/folds/fold_definitions.parquet")
    if not fold_path.exists():
        log.info("Fold definitions not found — building…")
        val_cfg = cfg["validation"]
        fold_df = build_folds(
            timestamps=panel["timestamp"],
            train_days=val_cfg["train_days"],
            val_days=val_cfg["val_days"],
            test_days=val_cfg["test_days"],
            step_days=val_cfg["step_days"],
            embargo_bars=val_cfg["embargo_bars"],
        )
    else:
        fold_df = load_parquet(fold_path)
    for col in ("start", "end"):
        if fold_df[col].dt.tz is not None:
            fold_df[col] = fold_df[col].dt.tz_localize(None)
    log.info("Loaded %d fold definitions", fold_df["fold_id"].nunique())

    # ── Label regimes on predictions ─────────────────────────────
    log.info("Labelling regimes…")
    regime_feat_cols = [
        "realized_vol_24h", "ret_24h", "ret_1h", "drawdown_168h",
        "drawdown_24h", "dollar_volume_24h", "is_weekend", "hour_of_day",
    ]
    available_cols = [c for c in regime_feat_cols if c in features.columns]
    feat_for_merge = features[["asset", "timestamp"] + available_cols].copy()
    preds_with_feats = preds.merge(
        feat_for_merge, on=["asset", "timestamp"], how="left", suffixes=("", "_feat")
    )
    preds_labelled = label_regimes(preds_with_feats)

    # ═══════════════════════════════════════════════════════════════
    #  Branch 1: Corrected Diagnostics
    # ═══════════════════════════════════════════════════════════════
    log.info("═══ Branch 1: Corrected Diagnostics ═══")

    # Filter to SOL-only for all corrected metrics
    sol_preds = preds_labelled[preds_labelled["asset"] == target_asset].copy()
    log.info("SOL-only predictions: %d rows", len(sol_preds))

    corrected_regime_df = regime_conditional_metrics(sol_preds, cost_bps=15.0, threshold=0.55)
    corrected_regime_df.to_csv(tbl_dir / "corrected_regime_metrics.csv", index=False)
    log.info("  Corrected regime metrics: %d rows", len(corrected_regime_df))

    corrected_fold_df = pd.DataFrame()
    if fold_df is not None:
        sol_features = features[features["asset"] == target_asset].copy()
        corrected_fold_df = compute_fold_descriptors(
            sol_features, fold_df, sol_preds, cost_bps=15.0, threshold=0.55
        )
        corrected_fold_df.to_csv(tbl_dir / "corrected_fold_attribution.csv", index=False)

        fold_reg = regress_fold_pnl(corrected_fold_df)
        if not fold_reg.empty:
            fold_reg.to_csv(tbl_dir / "corrected_fold_regression.csv", index=False)

    # ═══════════════════════════════════════════════════════════════
    #  Branch 2: Execution Timing Audit
    # ═══════════════════════════════════════════════════════════════
    log.info("═══ Branch 2: Execution Timing Audit ═══")

    fill_cfg = cfg.get("fill_simulation", {})
    robustness = sol_only_robustness(
        panel, preds_labelled,
        model_name=target_model,
        thresholds=fill_cfg.get("thresholds", [0.50, 0.55, 0.60]),
        fill_cost_bps=15.0,
    )
    fill_grid_df = robustness["fill_grid"]
    delay_grid_df = robustness["delay_grid"]
    fill_grid_df.to_csv(tbl_dir / "sol_fill_comparison.csv", index=False)
    delay_grid_df.to_csv(tbl_dir / "sol_delay_grid.csv", index=False)

    # ═══════════════════════════════════════════════════════════════
    #  Branch 3: Horizon Extension
    # ═══════════════════════════════════════════════════════════════
    log.info("═══ Branch 3: Horizon Extension ═══")

    h_cfg = cfg.get("horizon_study", {})
    feat_cols = _get_feature_cols(features)
    horizon_df = run_sol_horizon_study(
        panel, features, labels, fold_df,
        feat_cols=feat_cols,
        horizons=h_cfg.get("horizons", [1, 2, 4]),
        thresholds=h_cfg.get("thresholds", [0.50, 0.55, 0.60]),
        delays=h_cfg.get("delays", [0, 1, 2]),
        cost_bps=15.0,
    )
    horizon_df.to_csv(tbl_dir / "sol_horizon_study.csv", index=False)

    # ═══════════════════════════════════════════════════════════════
    #  Branch 4: Sparse Event Policies
    # ═══════════════════════════════════════════════════════════════
    log.info("═══ Branch 4: Sparse Event Policies ═══")

    sp_cfg = cfg.get("sparse_policies", {})
    sol_lgb_preds = preds_labelled[
        (preds_labelled["asset"] == target_asset) &
        (preds_labelled["model_name"] == target_model)
    ].copy()

    sparse_df = evaluate_sparse_policies(
        sol_lgb_preds,
        ret_col="fwd_ret_1h",
        cost_bps=15.0,
        top_pcts=sp_cfg.get("top_pcts", [0.10, 0.05, 0.025]),
        sep_gaps=sp_cfg.get("sep_gaps", [3, 6, 12]),
        cooldowns=sp_cfg.get("cooldowns", [3, 6, 12]),
        threshold=sp_cfg.get("threshold", 0.55),
        delays=sp_cfg.get("delays", [0, 1]),
    )
    sparse_df.to_csv(tbl_dir / "sol_sparse_policies.csv", index=False)

    # ═══════════════════════════════════════════════════════════════
    #  Branch 5: Cost Sensitivity
    # ═══════════════════════════════════════════════════════════════
    log.info("═══ Branch 5: Cost Sensitivity ═══")

    cs_cfg = cfg.get("cost_sensitivity", {})
    cost_df = cost_sensitivity_grid(
        sol_lgb_preds,
        ret_col="fwd_ret_1h",
        cost_levels_bps=cs_cfg.get("cost_levels_bps", [5.0, 7.5, 10.0, 15.0]),
        threshold=cs_cfg.get("threshold", 0.55),
        delays=cs_cfg.get("delays", [0, 1]),
    )
    cost_df.to_csv(tbl_dir / "sol_cost_sensitivity.csv", index=False)

    # ═══════════════════════════════════════════════════════════════
    #  Plots
    # ═══════════════════════════════════════════════════════════════
    log.info("═══ Generating plots ═══")

    if not fill_grid_df.empty:
        plot_fill_comparison(fill_grid_df, fig_dir)
    if not horizon_df.empty:
        plot_horizon_delay_heatmap(horizon_df, fig_dir)
    if not cost_df.empty:
        plot_cost_sensitivity(cost_df, fig_dir)

    # ═══════════════════════════════════════════════════════════════
    #  Summary Report
    # ═══════════════════════════════════════════════════════════════
    log.info("═══ Building summary report ═══")

    summary_path = build_summary(
        report_dir,
        corrected_regime_df=corrected_regime_df,
        corrected_fold_df=corrected_fold_df,
        fill_grid_df=fill_grid_df,
        delay_grid_df=delay_grid_df,
        horizon_df=horizon_df,
        sparse_df=sparse_df,
        cost_df=cost_df,
        cfg=cfg,
    )
    log.info("═══ Report generated: %s ═══", summary_path)

    # Save enriched predictions
    if cfg["reporting"].get("save_predictions", True):
        pred_dir = ensure_dir("data/artifacts/predictions")
        save_parquet(sol_lgb_preds, pred_dir / "exp003_sol_predictions.parquet")
        log.info("Saved SOL predictions to %s", pred_dir / "exp003_sol_predictions.parquet")


if __name__ == "__main__":
    main()
