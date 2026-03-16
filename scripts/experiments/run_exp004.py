"""exp004 experiment runner — Execution-design experiment.

Objective: Determine whether the SOL event signal can be converted into
a realistically executable strategy through entry redesign.

Branches:
  A — Fix Branch 3 reconciliation
  B — Intrabar entry approximation
  C — Passive-entry feasibility
  D — Signal-age decay curve
  E — Pooled train / SOL-only deploy
  F — Regime-gated sparse event policy

Usage:
    python run_exp004.py
    python run_exp004.py configs/experiments/crypto_1h_exp004.yaml
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
from src.diagnostics.branch_reconciliation import reconcile_branches
from src.diagnostics.intrabar_fill import intrabar_fill_grid
from src.diagnostics.passive_entry import passive_entry_study
from src.diagnostics.signal_decay import signal_decay_curve
from src.diagnostics.pooled_vs_solo import pooled_vs_solo_comparison
from src.diagnostics.regime_gated_policy import regime_gated_policy_study
from src.diagnostics.sol_horizon_study import run_sol_horizon_study
from src.reporting.exp004_report import build_exp004_summary, plot_signal_decay, plot_intrabar_comparison, plot_passive_fill_rate, plot_regime_gated_matrix
from src.utils.io import load_parquet, save_parquet, ensure_dir
from src.utils.logging import get_logger

log = get_logger("run_exp004")


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
            else "configs/experiments/crypto_1h_exp004.yaml"
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

    # ── Load panel ───────────────────────────────────────────────
    panel_path = Path(data_cfg["panel_dir"]) / "panel.parquet"
    if not panel_path.exists():
        log.info("Panel not found — downloading data and building panel…")
        download_data(cfg["data_config"])
        build_panel(cfg["data_config"])
    panel = load_parquet(panel_path)
    if hasattr(panel["timestamp"].dtype, "tz") and panel["timestamp"].dt.tz is not None:
        panel["timestamp"] = panel["timestamp"].dt.tz_localize(None)

    # ── Load exp001 predictions ──────────────────────────────────
    pred_path = cfg.get("exp001_predictions", "data/artifacts/predictions/crypto_1h_exp001_predictions.parquet")
    log.info("Loading predictions from %s", pred_path)
    preds = load_parquet(pred_path)
    log.info("Loaded %d prediction rows", len(preds))
    if hasattr(preds["timestamp"].dtype, "tz") and preds["timestamp"].dt.tz is not None:
        preds["timestamp"] = preds["timestamp"].dt.tz_localize(None)

    # ── Load features for regime labelling ────────────────────────
    log.info("Loading features…")
    features = _load_features_with_identifiers(cfg["data_config"])
    feat_cols = _get_feature_cols(features)

    # ── Load labels ──────────────────────────────────────────────
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

    # Filter to SOL + LightGBM
    sol_preds = preds_labelled[
        (preds_labelled["asset"] == target_asset) &
        (preds_labelled["model_name"] == target_model)
    ].copy()
    log.info("SOL LightGBM predictions: %d rows", len(sol_preds))

    # SOL panel
    sol_panel = panel[panel["asset"] == target_asset].copy()

    # ═══════════════════════════════════════════════════════════════
    #  Branch A: Reconciliation
    # ═══════════════════════════════════════════════════════════════
    log.info("═══ Branch A: Reconciliation ═══")

    recon_cfg = cfg.get("reconciliation", {})

    # Generate Branch 3 predictions (SOL-only retrained, horizon=1)
    log.info("Generating Branch 3 SOL-only retrained predictions for reconciliation…")
    branch3_preds_all = run_sol_horizon_study(
        panel, features, labels, fold_df,
        feat_cols=feat_cols,
        horizons=[1],
        thresholds=[recon_cfg.get("threshold", 0.55)],
        delays=[recon_cfg.get("delay", 0)],
        cost_bps=recon_cfg.get("cost_bps", 15.0),
    )
    # Also get raw predictions for reconciliation
    # Re-do SOL-only training to get the actual prediction DataFrame
    from src.models.train_lightgbm import train as train_lightgbm

    branch3_pred_parts: list[pd.DataFrame] = []
    merged_sol = features.copy()
    merged_sol["asset"] = panel["asset"].values
    merged_sol["timestamp"] = pd.DatetimeIndex(panel["timestamp"].values).tz_localize(None)
    for col in labels.columns:
        if col not in merged_sol.columns:
            merged_sol[col] = labels[col].values
    merged_sol = merged_sol[merged_sol["asset"] == target_asset].copy()

    target_col_h1 = "fwd_profitable_1h"
    ret_col_h1 = "fwd_ret_1h"

    if target_col_h1 in merged_sol.columns and ret_col_h1 in merged_sol.columns:
        fold_df_local = fold_df.copy()
        for col in ("start", "end"):
            if fold_df_local[col].dt.tz is not None:
                fold_df_local[col] = fold_df_local[col].dt.tz_localize(None)

        for fold_id in sorted(fold_df_local["fold_id"].unique()):
            fold_rows = fold_df_local[fold_df_local["fold_id"] == fold_id]
            train_r = fold_rows[fold_rows["split"] == "train"]
            test_r = fold_rows[fold_rows["split"] == "test"]
            if train_r.empty or test_r.empty:
                continue
            train_r = train_r.iloc[0]
            test_r = test_r.iloc[0]

            ts = merged_sol["timestamp"]
            train_mask = (ts >= train_r["start"]) & (ts < train_r["end"])
            test_mask = (ts >= test_r["start"]) & (ts < test_r["end"])

            train_df = merged_sol[train_mask].copy()
            test_df = merged_sol[test_mask].copy()

            required = feat_cols + [target_col_h1, ret_col_h1]
            for df_ in (train_df, test_df):
                valid = df_[required].notna().all(axis=1)
                df_.drop(df_[~valid].index, inplace=True)

            if len(train_df) < 50 or len(test_df) < 5:
                continue

            tm = train_lightgbm(
                train_df[feat_cols], train_df[target_col_h1],
                config_path="configs/models/lightgbm_v1.yaml",
                feature_names=feat_cols,
            )
            probs = tm.predict_proba(test_df)

            pred_df = test_df[["asset", "timestamp", ret_col_h1]].copy()
            pred_df = pred_df.rename(columns={ret_col_h1: "fwd_ret"})
            pred_df["y_pred_prob"] = probs
            pred_df["fold_id"] = fold_id
            pred_df["model_name"] = "lightgbm"
            branch3_pred_parts.append(pred_df)

    if branch3_pred_parts:
        branch3_preds_raw = pd.concat(branch3_pred_parts, ignore_index=True)
    else:
        branch3_preds_raw = pd.DataFrame()

    recon_df = reconcile_branches(
        sol_preds,
        branch3_preds_raw,
        threshold=recon_cfg.get("threshold", 0.55),
        delay=recon_cfg.get("delay", 0),
        cost_bps=recon_cfg.get("cost_bps", 15.0),
    )
    recon_df.to_csv(tbl_dir / "reconciliation.csv", index=False)

    # ═══════════════════════════════════════════════════════════════
    #  Branch B: Intrabar Fill Approximation
    # ═══════════════════════════════════════════════════════════════
    log.info("═══ Branch B: Intrabar Fill Approximation ═══")

    ib_cfg = cfg.get("intrabar_fill", {})
    intrabar_df = intrabar_fill_grid(
        sol_panel, sol_preds,
        thresholds=ib_cfg.get("thresholds", [0.55]),
        cost_bps=ib_cfg.get("cost_bps", 15.0),
        limit_bps_levels=ib_cfg.get("limit_bps_levels", [5.0, 10.0]),
    )
    intrabar_df.to_csv(tbl_dir / "intrabar_fill_grid.csv", index=False)

    # ═══════════════════════════════════════════════════════════════
    #  Branch C: Passive-Entry Feasibility
    # ═══════════════════════════════════════════════════════════════
    log.info("═══ Branch C: Passive-Entry Feasibility ═══")

    pe_cfg = cfg.get("passive_entry", {})
    passive_df = passive_entry_study(
        sol_panel, sol_preds,
        limit_offsets_bps=pe_cfg.get("limit_offsets_bps", [0.0, 5.0, 10.0]),
        threshold=pe_cfg.get("threshold", 0.55),
        cost_bps=pe_cfg.get("cost_bps", 15.0),
    )
    passive_df.to_csv(tbl_dir / "passive_entry.csv", index=False)

    # ═══════════════════════════════════════════════════════════════
    #  Branch D: Signal-Age Decay Curve
    # ═══════════════════════════════════════════════════════════════
    log.info("═══ Branch D: Signal-Age Decay Curve ═══")

    sd_cfg = cfg.get("signal_decay", {})
    decay_df = signal_decay_curve(
        sol_panel, sol_preds,
        deltas=sd_cfg.get("deltas", [0.0, 0.25, 0.5, 1.0, 2.0]),
        threshold=sd_cfg.get("threshold", 0.55),
        cost_bps=sd_cfg.get("cost_bps", 15.0),
    )
    decay_df.to_csv(tbl_dir / "signal_decay_curve.csv", index=False)

    # ═══════════════════════════════════════════════════════════════
    #  Branch E: Pooled vs SOL-Only
    # ═══════════════════════════════════════════════════════════════
    log.info("═══ Branch E: Pooled vs SOL-Only ═══")

    pvs_cfg = cfg.get("pooled_vs_solo", {})
    pooled_solo_df = pooled_vs_solo_comparison(
        panel, features, labels, fold_df, preds,
        feat_cols=feat_cols,
        target_asset=target_asset,
        threshold=pvs_cfg.get("threshold", 0.55),
        sep_gap=pvs_cfg.get("sep_gap", 3),
        cost_bps=pvs_cfg.get("cost_bps", 15.0),
    )
    pooled_solo_df.to_csv(tbl_dir / "pooled_vs_solo.csv", index=False)

    # ═══════════════════════════════════════════════════════════════
    #  Branch F: Regime-Gated Sparse Event Policy
    # ═══════════════════════════════════════════════════════════════
    log.info("═══ Branch F: Regime-Gated Sparse Event Policy ═══")

    rg_cfg = cfg.get("regime_gated", {})
    regime_gated_df = regime_gated_policy_study(
        sol_panel, sol_preds,
        threshold=rg_cfg.get("threshold", 0.55),
        sep_gap=rg_cfg.get("sep_gap", 3),
        cost_bps=rg_cfg.get("cost_bps", 15.0),
        fill_types=rg_cfg.get("fill_types", ["close_to_next_open", "next_bar_midpoint", "next_bar_vwap"]),
    )
    regime_gated_df.to_csv(tbl_dir / "regime_gated_policy.csv", index=False)

    # ═══════════════════════════════════════════════════════════════
    #  Plots
    # ═══════════════════════════════════════════════════════════════
    log.info("═══ Generating plots ═══")

    if not decay_df.empty:
        plot_signal_decay(decay_df, fig_dir)
    if not intrabar_df.empty:
        plot_intrabar_comparison(intrabar_df, fig_dir)
    if not passive_df.empty:
        plot_passive_fill_rate(passive_df, fig_dir)
    if not regime_gated_df.empty:
        plot_regime_gated_matrix(regime_gated_df, fig_dir)

    # ═══════════════════════════════════════════════════════════════
    #  Summary Report
    # ═══════════════════════════════════════════════════════════════
    log.info("═══ Building summary report ═══")

    summary_path = build_exp004_summary(
        report_dir,
        recon_df=recon_df,
        intrabar_df=intrabar_df,
        passive_df=passive_df,
        decay_df=decay_df,
        pooled_solo_df=pooled_solo_df,
        regime_gated_df=regime_gated_df,
        cfg=cfg,
    )
    log.info("═══ Report generated: %s ═══", summary_path)

    # Save enriched predictions
    if cfg["reporting"].get("save_predictions", True):
        pred_dir = ensure_dir("data/artifacts/predictions")
        save_parquet(sol_preds, pred_dir / "exp004_sol_predictions.parquet")
        log.info("Saved SOL predictions to %s", pred_dir / "exp004_sol_predictions.parquet")


if __name__ == "__main__":
    main()
