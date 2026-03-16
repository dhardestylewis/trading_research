"""exp012 experiment runner — Passive Realism.

Takes the best cells from exp011 and tests whether passive entry
can reduce effective roundtrip cost enough that a weak but real
gross move becomes net-positive.

Pipeline:
  1. Load exp011 atlas results (ranked cells)
  2. Load or rebuild panel for winning assets
  3. Run passive_realism_study() with queue haircut grid
  4. Check kill gate: ≥2 cells net-positive at 50% haircut
  5. Generate report

Usage:
    python run_exp012.py
    python run_exp012.py --dry-run
    python run_exp012.py --all-cells    # test all cells, not just viable
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.data.download_crypto import run as download_data
from src.data.build_panel import build as build_panel
from src.diagnostics.passive_realism import passive_realism_study, passive_realism_summary
from src.utils.io import ensure_dir
from src.utils.logging import get_logger

log = get_logger("run_exp012")


def main(config_path: str | None = None, dry_run: bool = False, all_cells: bool = False):
    if config_path is None:
        config_path = "configs/experiments/crypto_1h_exp012.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp_id = cfg.get("experiment_id", "exp012")
    log.info("═══ Starting experiment: %s ═══", exp_id)

    report_dir = Path(cfg["reporting"]["output_dir"])
    fig_dir = ensure_dir(report_dir / "figures")
    tbl_dir = ensure_dir(report_dir / "tables")

    # ── Step 1: Load exp011 atlas ────────────────────────────────
    log.info("── Step 1: Loading exp011 atlas ──")

    atlas_path = Path(cfg.get("exp011_atlas", "reports/exp011/tables/gross_move_atlas_ranked.csv"))
    if not atlas_path.exists():
        log.error("exp011 atlas not found at %s. Run exp011 first.", atlas_path)
        return None

    atlas = pd.read_csv(atlas_path)
    log.info("Loaded atlas: %d cells", len(atlas))

    # Select cells to test
    if all_cells:
        target_cells = atlas
        log.info("Testing all %d cells", len(target_cells))
    elif "viable" in atlas.columns:
        target_cells = atlas[atlas["viable"] == True]
        if target_cells.empty:
            log.warning("No viable cells in atlas. Using top 20 by p75 gross bps.")
            target_cells = atlas.nlargest(20, "gross_bps_abs_p75")
        log.info("Testing %d viable cells", len(target_cells))
    else:
        target_cells = atlas.nlargest(20, "gross_bps_p75" if "gross_bps_p75" in atlas.columns else "fill_count")
        log.info("Testing top %d cells by gross bps", len(target_cells))

    # Determine unique assets to load
    test_assets = target_cells["asset"].unique().tolist()
    log.info("Assets to test: %s", test_assets)

    # ── Step 2: Load panel ───────────────────────────────────────
    log.info("── Step 2: Loading panel ──")

    data_cfg_path = cfg["data_config"]
    download_data(data_cfg_path)
    panel = build_panel(data_cfg_path)

    if hasattr(panel["timestamp"].dtype, "tz") and panel["timestamp"].dt.tz is not None:
        panel["timestamp"] = panel["timestamp"].dt.tz_localize(None)

    # Filter to test assets
    panel = panel[panel["asset"].isin(test_assets)].copy()
    log.info("Panel filtered: %d rows, %d assets", len(panel), panel["asset"].nunique())

    if dry_run:
        panel = panel.groupby("asset").tail(500).reset_index(drop=True)
        log.info("  DRY-RUN: truncated to %d rows", len(panel))

    # ── Step 3: Run passive realism ──────────────────────────────
    log.info("═══ Step 3: Passive realism study ═══")

    passive_cfg = cfg.get("passive_grid", {})
    friction_cfg = cfg.get("friction", {})
    horizons = cfg.get("horizons", [{"bars": 1}])

    results_parts: list[pd.DataFrame] = []

    for h_cfg in horizons:
        horizon = h_cfg["bars"]
        log.info("── Horizon: %d bars ──", horizon)

        result = passive_realism_study(
            panel=panel,
            horizon=horizon,
            offset_grid_bps=passive_cfg.get("offset_bps", [0.0, 5.0, 10.0]),
            queue_haircuts=passive_cfg.get("queue_haircuts", [0.0, 0.25, 0.50]),
            round_trip_cost_bps=friction_cfg.get("market_round_trip_bps", 30.0),
            passive_cost_mult=passive_cfg.get("passive_cost_multiplier", 0.3),
        )

        if not result.empty:
            result["horizon"] = f"{horizon}h"
            results_parts.append(result)

    if not results_parts:
        log.error("No passive realism results generated.")
        return None

    full_results = pd.concat(results_parts, ignore_index=True)
    full_results.to_csv(tbl_dir / "passive_realism_full.csv", index=False)
    log.info("Passive realism: %d rows", len(full_results))

    # Summary
    summary = passive_realism_summary(full_results)
    summary.to_csv(tbl_dir / "passive_realism_summary.csv", index=False)
    log.info("Summary:\n%s", summary.to_string())

    # ── Step 4: Kill gate check ──────────────────────────────────
    log.info("═══ Step 4: Kill gate ═══")

    kill_cfg = cfg.get("kill_gate", {})
    haircut_for_gate = kill_cfg.get("haircut_for_gate", 0.50)
    min_positive = kill_cfg.get("min_net_positive_cells", 2)

    conservative = full_results[full_results["queue_haircut"] == haircut_for_gate]
    if "net_positive" in conservative.columns:
        n_positive = conservative["net_positive"].sum()
    else:
        n_positive = 0

    passes = n_positive >= min_positive
    reason = (
        f"{'PASS' if passes else 'FAIL'}: "
        f"{n_positive} net-positive cells at {haircut_for_gate:.0%} haircut "
        f"(need {min_positive})"
    )
    log.info("Kill gate: %s", reason)

    # ── Step 5: Generate report ──────────────────────────────────
    log.info("═══ Step 5: Generating report ═══")

    from src.reporting.exp012_report import build_exp012_summary, generate_passive_plots

    generate_passive_plots(fig_dir=fig_dir, results=full_results)

    summary_path = build_exp012_summary(
        report_dir=report_dir,
        results=full_results,
        summary=summary,
        kill_gate_result=(passes, reason),
        cfg=cfg,
    )
    log.info("═══ Report: %s ═══", summary_path)
    log.info("═══ Kill gate: %s ═══", reason)

    return full_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="exp012: Passive Realism")
    parser.add_argument(
        "config", nargs="?",
        default="configs/experiments/crypto_1h_exp012.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--all-cells", action="store_true",
                        help="Test all atlas cells, not just viable ones")
    args = parser.parse_args()
    main(config_path=args.config, dry_run=args.dry_run, all_cells=args.all_cells)
