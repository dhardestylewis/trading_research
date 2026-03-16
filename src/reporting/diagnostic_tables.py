"""Diagnostic tables for exp002 reporting.

Thin orchestration layer: calls diagnostic modules and saves CSVs
to the report directory.
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

from src.diagnostics.decile_analysis import (
    decile_metrics, tail_quantile_metrics, check_monotonicity,
)
from src.diagnostics.regime_performance import (
    regime_conditional_metrics, regime_gate_evaluation,
)
from src.diagnostics.asset_isolation import pooled_per_asset_metrics
from src.diagnostics.robustness_grid import robustness_grid
from src.diagnostics.fold_attribution import compute_fold_descriptors, regress_fold_pnl
from src.backtest.conditional_policy import evaluate_policies
from src.utils.io import save_csv, ensure_dir
from src.utils.logging import get_logger

log = get_logger("diagnostic_tables")


def save_all_diagnostic_tables(
    preds: pd.DataFrame,
    features: pd.DataFrame | None = None,
    fold_df: pd.DataFrame | None = None,
    out_dir: str | Path = "reports/exp002",
    cost_bps: float = 15.0,
) -> dict[str, Path]:
    """Generate and save all exp002 diagnostic tables.

    Returns dict mapping table name → saved path.
    """
    tbl_dir = ensure_dir(Path(out_dir) / "tables")
    paths: dict[str, Path] = {}

    # ── Branch B: Score deciles ──────────────────────────────────
    log.info("  Computing score-decile metrics…")
    dec = decile_metrics(preds, cost_bps=cost_bps)
    paths["score_deciles"] = save_csv(dec, tbl_dir / "score_deciles.csv")

    # ── Branch B: Tail quantiles ─────────────────────────────────
    log.info("  Computing tail-quantile metrics…")
    tq = tail_quantile_metrics(preds, cost_bps=cost_bps)
    paths["tail_quantile_metrics"] = save_csv(tq, tbl_dir / "tail_quantile_metrics.csv")

    # ── Branch B: Monotonicity ───────────────────────────────────
    mono = check_monotonicity(dec)
    paths["monotonicity"] = save_csv(mono, tbl_dir / "monotonicity.csv")

    # ── Branch A: Asset-mode metrics ─────────────────────────────
    log.info("  Computing asset-mode metrics…")
    am = pooled_per_asset_metrics(preds, cost_bps=cost_bps)
    paths["asset_mode_metrics"] = save_csv(am, tbl_dir / "asset_mode_metrics.csv")

    # ── Branch C: Regime conditional metrics ─────────────────────
    regime_cols_present = [c for c in preds.columns if c.startswith("regime_")]
    if regime_cols_present:
        log.info("  Computing regime-conditional metrics…")
        rcm = regime_conditional_metrics(preds, cost_bps=cost_bps)
        paths["regime_metrics"] = save_csv(rcm, tbl_dir / "regime_metrics.csv")
    else:
        log.warning("  No regime columns in predictions — skipping regime metrics")

    # ── Branch C: Fold regime attribution ────────────────────────
    if features is not None and fold_df is not None:
        log.info("  Computing fold-regime attribution…")
        fd = compute_fold_descriptors(features, fold_df, preds, cost_bps=cost_bps)
        paths["fold_regime_attribution"] = save_csv(fd, tbl_dir / "fold_regime_attribution.csv")

        reg = regress_fold_pnl(fd)
        if not reg.empty:
            paths["fold_regime_regression"] = save_csv(reg, tbl_dir / "fold_regime_regression.csv")

    # ── Branch D: Policy comparison ──────────────────────────────
    if regime_cols_present:
        log.info("  Evaluating conditional policies…")
        pc = evaluate_policies(preds, cost_bps=cost_bps)
        paths["policy_comparison"] = save_csv(pc, tbl_dir / "policy_comparison.csv")

    # ── Branch E: Execution robustness ───────────────────────────
    log.info("  Computing execution robustness grid…")
    rg = robustness_grid(preds)
    paths["delay_cost_robustness"] = save_csv(rg, tbl_dir / "delay_cost_robustness.csv")

    log.info("  Saved %d diagnostic tables", len(paths))
    return paths
