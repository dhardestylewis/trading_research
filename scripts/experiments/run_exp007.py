"""exp007 experiment runner — Live paper canary.

Objective: Replace simulated fills with real paper fills, rerun the
identical execution-quality pipeline, compare realized vs simulated
lane metrics.  No model changes, no policy changes.

Two operating modes:
  --mode simulate   Load exp006 simulated logs, run canary health
                    checks, generate comparison report.  Useful for
                    testing the pipeline without live exchange data.

  --mode live       (future) Connect to exchange sandbox, run live
                    signal loop, log to PaperTradeLogger, compare
                    against exp006 baseline.

Pipeline:
  1. Load config + exp006 simulated baseline
  2. Load or generate realized logs
  3. Compute execution quality per lane
  4. Run compare_realized_vs_simulated() vs exp006 baseline
  5. Run check_canary_health() against tolerance bands
  6. Generate report

Usage:
    python run_exp007.py
    python run_exp007.py --mode simulate
    python run_exp007.py configs/experiments/crypto_1h_exp007.yaml
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.diagnostics.execution_quality import compute_all_metrics
from src.diagnostics.live_canary_monitor import (
    CanaryHealthCheck,
    compare_realized_vs_simulated,
    weekly_execution_error,
)
from src.reporting.exp007_report import (
    build_exp007_summary,
    plot_simulated_vs_realized_deltas,
    plot_shortfall_analysis,
    plot_weekly_stability,
    plot_lane_overlay,
)
from src.utils.io import ensure_dir
from src.utils.logging import get_logger

log = get_logger("run_exp007")


def main(config_path: str | None = None, mode: str = "simulate"):
    if config_path is None:
        config_path = "configs/experiments/crypto_1h_exp007.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp_id = cfg["experiment_id"]
    tolerance_bands = cfg.get("tolerance_bands", {})
    simulated_baseline_cfg = cfg.get("simulated_baseline", {})

    log.info("═══ Starting experiment: %s (mode=%s) ═══", exp_id, mode)

    report_dir = Path(cfg["reporting"]["output_dir"])
    fig_dir = ensure_dir(report_dir / "figures")
    tbl_dir = ensure_dir(report_dir / "tables")

    # ── Load simulated baseline (exp006) ────────────────────────
    sim_log_path = simulated_baseline_cfg.get(
        "exp006_log", "reports/exp006/tables/full_paper_trade_log.csv"
    )
    sim_summary_path = simulated_baseline_cfg.get(
        "exp006_lane_summary", "reports/exp006/tables/execution_quality.csv"
    )

    log.info("Loading simulated baseline from %s", sim_log_path)
    simulated_log = pd.read_csv(sim_log_path)
    simulated_summary = pd.read_csv(sim_summary_path)
    log.info(
        "Simulated baseline: %d signals, %d lanes",
        len(simulated_log),
        len(simulated_summary),
    )

    # ── Get realized logs ───────────────────────────────────────
    if mode == "simulate":
        # Self-comparison mode: use simulated data as "realized"
        # This validates the pipeline — all health checks should pass.
        log.info("Simulate mode: using exp006 simulated log as realized data")
        realized_log = simulated_log.copy()
    elif mode == "live":
        # Future: load from live paper-trade logs
        live_log_path = tbl_dir / "live_paper_trade_log.csv"
        if live_log_path.exists():
            log.info("Loading live paper-trade log from %s", live_log_path)
            realized_log = pd.read_csv(live_log_path)
        else:
            log.warning(
                "No live log found at %s — falling back to simulate mode",
                live_log_path,
            )
            realized_log = simulated_log.copy()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ── Compute per-lane metrics on realized data ───────────────
    lane_col = "entry_mode"
    realized_lane_rows: list[dict] = []

    if lane_col in realized_log.columns:
        for lane_name, lane_df in realized_log.groupby(lane_col):
            metrics = compute_all_metrics(lane_df)
            metrics["lane"] = lane_name
            realized_lane_rows.append(metrics)
    else:
        metrics = compute_all_metrics(realized_log)
        metrics["lane"] = "all"
        realized_lane_rows.append(metrics)

    realized_summary = pd.DataFrame(realized_lane_rows)
    realized_summary.to_csv(tbl_dir / "realized_execution_quality.csv", index=False)
    log.info("Realized metrics computed for %d lanes", len(realized_summary))

    # ── Compare realized vs simulated ───────────────────────────
    log.info("═══ Comparing realized vs simulated ═══")
    comparison = compare_realized_vs_simulated(
        simulated_log=simulated_log,
        realized_log=realized_log,
        lane_col=lane_col,
    )
    comparison.to_csv(tbl_dir / "realized_vs_simulated.csv", index=False)

    n_degraded = comparison["degradation_flag"].sum()
    log.info(
        "Comparison: %d metric-lane pairs, %d degraded",
        len(comparison),
        n_degraded,
    )

    # ── Canary health checks ────────────────────────────────────
    log.info("═══ Running canary health checks ═══")
    checker = CanaryHealthCheck(tolerance_bands)
    health_checks = checker.check_all_lanes(realized_log, lane_col=lane_col)
    health_table = checker.summary_table(health_checks)
    health_table.to_csv(tbl_dir / "canary_health.csv", index=False)

    all_ok = checker.all_passed(health_checks)
    log.info("Canary health: %s", "ALL PASS" if all_ok else "SOME FAILURES")

    for lane_name, checks in health_checks.items():
        for bc in checks:
            log.info(
                "  %s %s: %s = %.4f (threshold: %s %.4f)",
                bc.status_icon, lane_name, bc.metric,
                bc.value, bc.direction, bc.threshold,
            )

    # ── Weekly execution error ──────────────────────────────────
    log.info("═══ Computing weekly execution error ═══")
    weekly = weekly_execution_error(realized_log)
    if not weekly.empty:
        weekly.to_csv(tbl_dir / "weekly_execution_error.csv", index=False)
        log.info("Weekly execution error computed for %d weeks", len(weekly))

    # ── Plots ───────────────────────────────────────────────────
    log.info("═══ Generating plots ═══")
    if not comparison.empty:
        plot_simulated_vs_realized_deltas(comparison, fig_dir)
    if not realized_log.empty:
        plot_shortfall_analysis(realized_log, fig_dir)
    if not weekly.empty:
        plot_weekly_stability(weekly, fig_dir)
    if not simulated_summary.empty and not realized_summary.empty:
        plot_lane_overlay(simulated_summary, realized_summary, fig_dir)

    # ── Summary report ──────────────────────────────────────────
    log.info("═══ Building summary report ═══")
    summary_path = build_exp007_summary(
        report_dir=report_dir,
        simulated_summary=simulated_summary,
        realized_summary=realized_summary,
        comparison=comparison,
        health_table=health_table,
        weekly=weekly,
        all_passed=all_ok,
        tolerance_bands=tolerance_bands,
        policy_cfg=cfg.get("policy", {}),
        metric_dict=cfg.get("metric_dictionary", {}),
        stage_gates=cfg.get("stage_gates", {}),
        mode=mode,
        cfg=cfg,
    )
    log.info("═══ Report generated: %s ═══", summary_path)

    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="exp007: Live paper canary")
    parser.add_argument(
        "config", nargs="?",
        default="configs/experiments/crypto_1h_exp007.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--mode", choices=["simulate", "live"], default="simulate",
        help="Operating mode: 'simulate' (self-comparison) or 'live' (exchange data)",
    )
    args = parser.parse_args()
    main(config_path=args.config, mode=args.mode)
