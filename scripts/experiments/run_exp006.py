"""exp006 experiment runner — Paper-trade validation study.

Objective: Bridge exp005 simulated fills to live order lifecycle
evaluation.  This is NOT a new model experiment.  The model gates
were already passed in exp005.

Pipeline:
  1. Load config
  2. Load exp005 gated predictions + panel OHLC
  3. For each lane (primary + shadow):
     a. Generate simulated paper-trade log via simulate_from_backtest()
     b. Compute execution quality metrics
  4. Build report

Usage:
    python run_exp006.py
    python run_exp006.py configs/experiments/crypto_1h_exp006.yaml
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.download_crypto import run as download_data
from src.data.build_panel import build as build_panel
from src.diagnostics.regime_labeller import label_regimes
from src.diagnostics.paper_trade_logger import simulate_from_backtest
from src.diagnostics.execution_quality import compute_all_metrics
from src.reporting.exp006_report import (
    build_exp006_summary,
    plot_slippage_distribution,
    plot_fill_rate_by_hour,
    plot_adverse_selection,
    plot_shortfall_scatter,
    plot_lane_comparison,
)
from src.utils.io import load_parquet, save_parquet, ensure_dir
from src.utils.logging import get_logger

log = get_logger("run_exp006")


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


def main(config_path: str | None = None):
    if config_path is None:
        config_path = (
            sys.argv[1]
            if len(sys.argv) > 1 and not sys.argv[1].startswith("--")
            else "configs/experiments/crypto_1h_exp006.yaml"
        )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp_id = cfg["experiment_id"]
    target_asset = cfg.get("target_asset", "SOL-USD")
    target_model = cfg.get("target_model", "lightgbm")
    policy_cfg = cfg.get("policy", {})
    go_no_go_cfg = cfg.get("go_no_go", {})
    metric_dict = cfg.get("metric_dictionary", {})

    log.info("═══ Starting experiment: %s ═══", exp_id)
    log.info("Policy: %s", policy_cfg)

    report_dir = Path(cfg["reporting"]["output_dir"])
    fig_dir = ensure_dir(report_dir / "figures")
    tbl_dir = ensure_dir(report_dir / "tables")

    # ── Load data ────────────────────────────────────────────────
    with open(cfg["data_config"]) as f:
        data_cfg = yaml.safe_load(f)

    panel_path = Path(data_cfg["panel_dir"]) / "panel.parquet"
    if not panel_path.exists():
        log.info("Panel not found — downloading data and building panel…")
        download_data(cfg["data_config"])
        build_panel(cfg["data_config"])
    panel = load_parquet(panel_path)
    if hasattr(panel["timestamp"].dtype, "tz") and panel["timestamp"].dt.tz is not None:
        panel["timestamp"] = panel["timestamp"].dt.tz_localize(None)

    # ── Load predictions ─────────────────────────────────────────
    pred_path = cfg.get(
        "exp005_predictions",
        "data/artifacts/predictions/exp005_sol_gated_predictions.parquet",
    )
    log.info("Loading predictions from %s", pred_path)
    preds = load_parquet(pred_path)
    log.info("Loaded %d prediction rows", len(preds))
    if hasattr(preds["timestamp"].dtype, "tz") and preds["timestamp"].dt.tz is not None:
        preds["timestamp"] = preds["timestamp"].dt.tz_localize(None)

    # ── Regime labelling ────────────────────────────────────────────
    # exp005 predictions are already regime-labelled and gated.
    # Skip re-labelling to avoid duplicate column errors.
    if "regime" in preds.columns:
        log.info("Predictions already have regime labels — skipping label_regimes")
        preds_labelled = preds
    else:
        features = _load_features_with_identifiers(cfg["data_config"])
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

    # Filter to target asset + model
    sol_preds = preds_labelled[
        (preds_labelled["asset"] == target_asset) &
        (preds_labelled["model_name"] == target_model)
    ].copy()
    log.info("SOL LightGBM predictions: %d rows", len(sol_preds))

    # Apply regime gate
    gate = policy_cfg.get("regime_gate", "NOT_rebound")
    if gate == "NOT_rebound" and "regime" in sol_preds.columns:
        sol_preds_gated = sol_preds[sol_preds["regime"] != "rebound"].copy()
    elif gate.startswith("AND_") and "regime" in sol_preds.columns:
        gate_regime = gate.replace("AND_", "")
        sol_preds_gated = sol_preds[sol_preds["regime"] == gate_regime].copy()
    else:
        sol_preds_gated = sol_preds.copy()
    log.info("After %s gate: %d prediction rows", gate, len(sol_preds_gated))

    sol_panel = panel[panel["asset"] == target_asset].copy()

    # ═══════════════════════════════════════════════════════════════
    #  Build lanes
    # ═══════════════════════════════════════════════════════════════
    primary_cfg = cfg.get("primary_lane", {})
    shadow_cfgs = cfg.get("shadow_lanes", [])

    all_lanes = [(primary_cfg, "primary")] + [
        (sc, "shadow") for sc in shadow_cfgs
    ]

    all_logs: list[pd.DataFrame] = []
    lane_rows: list[dict] = []

    for lane_cfg, lane_type in all_lanes:
        lane_name = lane_cfg.get("name", "unknown")
        log.info("═══ Lane: %s (%s) ═══", lane_name, lane_type)

        # Generate simulated paper-trade log
        lane_log = simulate_from_backtest(
            panel=sol_panel,
            predictions=sol_preds_gated,
            entry_mode_cfg=lane_cfg,
            policy_cfg=policy_cfg,
            lane_type=lane_type,
        )

        if lane_log.empty:
            log.warning("No signals for lane %s — skipping", lane_name)
            continue

        # Compute execution quality metrics
        metrics = compute_all_metrics(lane_log)
        metrics["lane"] = lane_name
        metrics["lane_type"] = lane_type
        lane_rows.append(metrics)

        # Save per-lane log
        lane_log.to_csv(tbl_dir / f"paper_trade_log_{lane_name}.csv", index=False)
        all_logs.append(lane_log)
        log.info(
            "Lane %s: %d signals, fill rate %.1f%%, sharpe %.2f",
            lane_name,
            metrics.get("submitted", 0),
            metrics.get("realized_fill_rate", 0) * 100,
            metrics.get("sharpe", np.nan),
        )

    # ── Combine results ──────────────────────────────────────────
    lane_summary = pd.DataFrame(lane_rows)
    full_log = pd.concat(all_logs, ignore_index=True) if all_logs else pd.DataFrame()

    lane_summary.to_csv(tbl_dir / "execution_quality.csv", index=False)
    if not full_log.empty:
        full_log.to_csv(tbl_dir / "full_paper_trade_log.csv", index=False)

    # ═══════════════════════════════════════════════════════════════
    #  Plots
    # ═══════════════════════════════════════════════════════════════
    log.info("═══ Generating plots ═══")

    if not full_log.empty:
        plot_slippage_distribution(full_log, fig_dir)
        plot_fill_rate_by_hour(full_log, fig_dir)
        plot_adverse_selection(full_log, fig_dir)
        plot_shortfall_scatter(full_log, fig_dir)

    if not lane_summary.empty:
        plot_lane_comparison(lane_summary, fig_dir)

    # ═══════════════════════════════════════════════════════════════
    #  Summary Report
    # ═══════════════════════════════════════════════════════════════
    log.info("═══ Building summary report ═══")

    summary_path = build_exp006_summary(
        report_dir=report_dir,
        lane_summary=lane_summary,
        full_log=full_log,
        go_no_go_cfg=go_no_go_cfg,
        policy_cfg=policy_cfg,
        metric_dict=metric_dict,
        cfg=cfg,
    )
    log.info("═══ Report generated: %s ═══", summary_path)

    # Save predictions
    if cfg["reporting"].get("save_predictions", True):
        pred_dir = ensure_dir("data/artifacts/predictions")
        # Deduplicate columns (exp005 preds may have _feat suffixes)
        preds_to_save = sol_preds_gated.loc[:, ~sol_preds_gated.columns.duplicated()]
        save_parquet(preds_to_save, pred_dir / "exp006_sol_gated_predictions.parquet")
        log.info("Saved gated predictions")


if __name__ == "__main__":
    main()
