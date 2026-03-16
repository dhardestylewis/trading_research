"""exp011 experiment runner — Gross-Move Atlas.

Objective: Non-ML economic screening. Find asset × horizon × regime × entry
cells where gross move magnitude plausibly clears 30+ bps friction.

Pipeline:
  1. Download/load panel for atlas universe
  2. Build features + regime labels (standard + event-driven)
  3. Build forward returns at configured horizons
  4. Run gross_move_atlas.build_atlas() across all cells
  5. Rank and filter by kill gate
  6. Generate report with gross bps table as first page

Kill criterion: ≥3 cells with ≥100 trades and gross move distribution
plausibly clearing 30 bps friction. If not met, STOP.

Usage:
    python run_exp011.py
    python run_exp011.py --dry-run
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
from src.diagnostics.regime_labeller import label_regimes
from src.diagnostics.event_regimes import label_event_regimes
from src.diagnostics.gross_move_atlas import (
    AtlasConfig,
    build_atlas,
    rank_cells,
    check_kill_gate,
    atlas_summary_table,
)
from src.utils.io import ensure_dir
from src.utils.logging import get_logger

log = get_logger("run_exp011")


def main(config_path: str | None = None, dry_run: bool = False):
    if config_path is None:
        config_path = "configs/experiments/crypto_1h_exp011.yaml"

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_id = cfg.get("experiment_id", "exp011")
    log.info("═══ Starting experiment: %s ═══", exp_id)
    if dry_run:
        log.info("  DRY-RUN mode: using truncated data")

    report_dir = Path(cfg["reporting"]["output_dir"])
    fig_dir = ensure_dir(report_dir / "figures")
    tbl_dir = ensure_dir(report_dir / "tables")

    # ── Step 1: Data pipeline ────────────────────────────────────
    log.info("── Step 1: Data pipeline ──")

    data_cfg_path = cfg["data_config"]
    with open(data_cfg_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    download_data(data_cfg_path)
    panel = build_panel(data_cfg_path)

    # Normalize timestamps
    if hasattr(panel["timestamp"].dtype, "tz") and panel["timestamp"].dt.tz is not None:
        panel["timestamp"] = panel["timestamp"].dt.tz_localize(None)

    available_assets = sorted(panel["asset"].unique().tolist())
    log.info("Panel: %d rows, %d assets: %s", len(panel), len(available_assets), available_assets)

    if dry_run:
        # Keep only first 3 assets and recent data
        top_assets = available_assets[:3]
        panel = panel[panel["asset"].isin(top_assets)].copy()
        panel = panel.groupby("asset").tail(500).reset_index(drop=True)
        log.info("  DRY-RUN: truncated to %d rows, %d assets", len(panel), len(top_assets))

    # ── Step 2: Features + regime labels ─────────────────────────
    log.info("── Step 2: Features + regime labels ──")

    # Build baseline features (for regime labelling)
    panel_path = Path(data_cfg["panel_dir"]) / "panel.parquet"
    features = build_features(panel_path, cfg.get("feature_config"))

    # Standard regime labels
    regime_df = label_regimes(features)

    # Event-driven regime labels
    event_df = label_event_regimes(panel)

    # Combine regime flags
    all_regime_cols = []
    for col in regime_df.columns:
        if col.startswith("regime_") and col not in panel.columns:
            panel[col] = regime_df[col].values
            all_regime_cols.append(col)
    for col in event_df.columns:
        if col.startswith("event_") and col not in panel.columns:
            panel[col] = event_df[col].values
            all_regime_cols.append(col)

    log.info("Regime flags attached: %d columns", len(all_regime_cols))

    # ── Step 3: Build atlas config ───────────────────────────────
    log.info("── Step 3: Configuring atlas ──")

    horizons = [h["bars"] for h in cfg.get("horizons", [{"bars": 1}])]
    friction_cfg = cfg.get("friction", {})
    kill_cfg = cfg.get("kill_gate", {})

    atlas_config = AtlasConfig(
        horizons=horizons,
        friction_bps=friction_cfg.get("round_trip_bps", 30.0),
        gross_thresholds_bps=friction_cfg.get("gross_thresholds_bps", [20, 30, 40, 50]),
        min_trades=kill_cfg.get("min_trades_per_cell", 100),
        entry_conventions=cfg.get("entry_conventions", [
            {"name": "market_next_open", "type": "market", "offset_bps": 0.0},
        ]),
        regime_slices=cfg.get("regime_slices", [{"name": "all", "filter": None}]),
    )

    log.info(
        "Atlas config: %d horizons, %d entries, %d regimes, friction=%.0f bps",
        len(atlas_config.horizons), len(atlas_config.entry_conventions),
        len(atlas_config.regime_slices), atlas_config.friction_bps,
    )

    # ── Step 4: Build atlas ──────────────────────────────────────
    log.info("═══ Step 4: Building gross-move atlas ═══")

    atlas = build_atlas(
        panel=panel,
        config=atlas_config,
        regime_df=None,  # regimes already merged into panel
    )

    if atlas.empty:
        log.error("Atlas is empty — no cells computed. Check data pipeline.")
        return None

    atlas.to_csv(tbl_dir / "gross_move_atlas_full.csv", index=False)
    log.info("Full atlas saved: %d cells", len(atlas))

    # ── Step 5: Rank and filter ──────────────────────────────────
    log.info("═══ Step 5: Ranking and filtering ═══")

    ranked = rank_cells(
        atlas,
        min_trades=kill_cfg.get("min_trades_per_cell", 100),
        friction_bps=friction_cfg.get("round_trip_bps", 30.0),
    )

    ranked.to_csv(tbl_dir / "gross_move_atlas_ranked.csv", index=False)

    # Kill gate check
    passes, reason = check_kill_gate(
        ranked,
        min_viable_cells=kill_cfg.get("min_viable_cells", 3),
    )
    log.info("Kill gate: %s", reason)

    # Summary table (first page)
    summary = atlas_summary_table(ranked)
    summary.to_csv(tbl_dir / "gross_bps_distribution.csv", index=False)

    # ── Per-asset summary ────────────────────────────────────────
    log.info("── Per-asset summary ──")
    if not atlas.empty and "asset" in atlas.columns:
        asset_summary = (
            atlas[atlas["entry"] == "market_next_open"]
            .groupby("asset")
            .agg(
                total_cells=("fill_count", "count"),
                avg_gross_bps_median=("gross_bps_median", "mean"),
                avg_gross_bps_p75=("gross_bps_p75", "mean"),
                max_gross_bps_p75=("gross_bps_p75", "max"),
                avg_frac_gt_30bps=("frac_abs_gt_30bps", "mean"),
            )
            .sort_values("avg_gross_bps_p75", ascending=False)
        )
        asset_summary.to_csv(tbl_dir / "per_asset_summary.csv")
        log.info("Per-asset summary:\n%s", asset_summary.to_string())

    # ── Per-regime summary (market entry only) ───────────────────
    if not atlas.empty and "regime" in atlas.columns:
        regime_summary = (
            atlas[atlas["entry"] == "market_next_open"]
            .groupby("regime")
            .agg(
                total_cells=("fill_count", "count"),
                avg_gross_bps_median=("gross_bps_median", "mean"),
                avg_gross_bps_p75=("gross_bps_p75", "mean"),
                max_gross_bps_p75=("gross_bps_p75", "max"),
                avg_frac_gt_30bps=("frac_abs_gt_30bps", "mean"),
            )
            .sort_values("avg_gross_bps_p75", ascending=False)
        )
        regime_summary.to_csv(tbl_dir / "per_regime_summary.csv")
        log.info("Per-regime summary:\n%s", regime_summary.to_string())

    # ── Step 6: Generate report ──────────────────────────────────
    log.info("═══ Step 6: Generating report ═══")

    from src.reporting.exp011_report import build_exp011_summary, generate_atlas_plots

    generate_atlas_plots(
        fig_dir=fig_dir,
        atlas=ranked,
    )

    summary_path = build_exp011_summary(
        report_dir=report_dir,
        atlas=ranked,
        kill_gate_result=(passes, reason),
        cfg=cfg,
    )

    log.info("═══ Report: %s ═══", summary_path)
    log.info("═══ Kill gate verdict: %s ═══", reason)

    return ranked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="exp011: Gross-Move Atlas")
    parser.add_argument(
        "config", nargs="?",
        default="configs/experiments/crypto_1h_exp011.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run with truncated data for pipeline validation",
    )
    args = parser.parse_args()
    main(config_path=args.config, dry_run=args.dry_run)
