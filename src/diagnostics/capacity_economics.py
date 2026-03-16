"""Branch E — Capacity economics scorecard.

Integrator module that computes the unified $/week metric:

    weekly_pnl = edge_per_trade_bps × trade_count_per_week × notional

Rolls up results from Branches A–D into a single economics table and
produces a sensitivity heatmap across notional, slippage, and frequency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger("capacity_economics")


def compute_capacity_scorecard(
    baseline_edge_bps: float,
    baseline_trades_per_week: float,
    *,
    horizon_result: pd.DataFrame | None = None,
    universe_result: pd.DataFrame | None = None,
    exec_aware_result: pd.DataFrame | None = None,
    notional: float = 100_000.0,
    cost_bps: float = 15.0,
) -> pd.DataFrame:
    """Build the capacity economics scorecard.

    Computes $/week under different scenarios:
      1. Current baseline (SOL-only, 1h, ~4 trades/week, ~44 bps)
      2. + multi-horizon stacking
      3. + universe expansion
      4. + execution-aware targeting
      5. Combined best-of

    Returns DataFrame with one row per scenario.
    """
    rows: list[dict] = []

    # Helper
    def _pnl(edge_bps: float, trades_week: float, notional_usd: float) -> float:
        return (edge_bps / 10_000.0) * trades_week * notional_usd

    # ── Scenario 1: Current baseline ────────────────────────
    base_pnl = _pnl(baseline_edge_bps, baseline_trades_per_week, notional)
    rows.append({
        "scenario": "1_baseline_sol_1h",
        "edge_bps": baseline_edge_bps,
        "trades_per_week": baseline_trades_per_week,
        "notional_usd": notional,
        "weekly_pnl_usd": base_pnl,
        "improvement_pct": 0.0,
    })

    # ── Scenario 2: + Multi-horizon stacking ────────────────
    if horizon_result is not None and not horizon_result.empty:
        stacked_row = horizon_result[horizon_result["horizon"].astype(str) == "stacked"]
        if not stacked_row.empty:
            multiplier = stacked_row["trade_multiplier"].values[0] if "trade_multiplier" in stacked_row.columns else 1.0
            stacked_bps = stacked_row["mean_net_bps"].values[0]
            stacked_trades = baseline_trades_per_week * multiplier
            pnl = _pnl(stacked_bps, stacked_trades, notional)
        else:
            stacked_bps = baseline_edge_bps
            stacked_trades = baseline_trades_per_week
            pnl = base_pnl

        rows.append({
            "scenario": "2_multi_horizon",
            "edge_bps": stacked_bps,
            "trades_per_week": stacked_trades,
            "notional_usd": notional,
            "weekly_pnl_usd": pnl,
            "improvement_pct": ((pnl - base_pnl) / abs(base_pnl) * 100) if base_pnl != 0 else 0,
        })
    else:
        stacked_bps = baseline_edge_bps
        stacked_trades = baseline_trades_per_week

    # ── Scenario 3: + Universe expansion ────────────────────
    if universe_result is not None and not universe_result.empty:
        qualifying = universe_result[
            (universe_result["qualifies"] == True) &
            (universe_result["asset"] != "QUALIFYING_AGGREGATE")
        ]
        n_qualifying = len(qualifying)
        # Each qualifying asset adds its own trade stream
        if n_qualifying > 0:
            avg_trades = qualifying["trade_count"].mean()
            # Rough: total data span ~ 3 years, so trades_per_week ≈ trade_count / (52*3)
            extra_trades_week = (qualifying["trade_count"].sum()) / (52 * 3)
            avg_edge = qualifying["mean_net_bps"].mean()
            combined_trades = baseline_trades_per_week + extra_trades_week
            combined_edge = (baseline_edge_bps * baseline_trades_per_week + avg_edge * extra_trades_week) / combined_trades if combined_trades > 0 else baseline_edge_bps
            pnl = _pnl(combined_edge, combined_trades, notional)
        else:
            combined_edge = baseline_edge_bps
            combined_trades = baseline_trades_per_week
            pnl = base_pnl

        rows.append({
            "scenario": "3_universe_expansion",
            "edge_bps": combined_edge,
            "trades_per_week": combined_trades,
            "notional_usd": notional,
            "weekly_pnl_usd": pnl,
            "improvement_pct": ((pnl - base_pnl) / abs(base_pnl) * 100) if base_pnl != 0 else 0,
        })
    else:
        combined_edge = baseline_edge_bps
        combined_trades = baseline_trades_per_week

    # ── Scenario 4: + Execution-aware targeting ─────────────
    if exec_aware_result is not None and not exec_aware_result.empty:
        best = exec_aware_result.loc[exec_aware_result["mean_net_bps"].idxmax()]
        exec_edge = best["mean_net_bps"]
        pnl = _pnl(exec_edge, baseline_trades_per_week, notional)

        rows.append({
            "scenario": "4_execution_aware",
            "edge_bps": exec_edge,
            "trades_per_week": baseline_trades_per_week,
            "notional_usd": notional,
            "weekly_pnl_usd": pnl,
            "improvement_pct": ((pnl - base_pnl) / abs(base_pnl) * 100) if base_pnl != 0 else 0,
        })
        best_exec_edge = exec_edge
    else:
        best_exec_edge = baseline_edge_bps

    # ── Scenario 5: Deployable best ────────────────────────
    # NOTE: Only use scenarios from the SAME validated branch.
    # Do NOT mix best-edge from one branch with best-frequency
    # from another — that produces a non-deployable synthetic.
    # The only branch with validated additive economics is
    # universe expansion (Branch B).
    deploy_candidates = [
        ("1_baseline_sol_1h", baseline_edge_bps, baseline_trades_per_week),
    ]
    if universe_result is not None and not universe_result.empty:
        deploy_candidates.append(("3_universe_expansion", combined_edge, combined_trades))

    # Pick the candidate with highest $/week
    best_scenario, best_edge, best_trades = max(
        deploy_candidates, key=lambda x: _pnl(x[1], x[2], notional)
    )
    best_pnl = _pnl(best_edge, best_trades, notional)

    rows.append({
        "scenario": "5_deployable_best",
        "edge_bps": best_edge,
        "trades_per_week": best_trades,
        "notional_usd": notional,
        "weekly_pnl_usd": best_pnl,
        "improvement_pct": ((best_pnl - base_pnl) / abs(base_pnl) * 100) if base_pnl != 0 else 0,
        "source_branch": best_scenario,
    })

    return pd.DataFrame(rows)


def sensitivity_grid(
    baseline_edge_bps: float,
    baseline_trades_per_week: float,
    *,
    notional_levels: list[float] = (50_000, 100_000, 250_000, 500_000),
    slippage_levels_bps: list[float] = (5, 10, 15, 20),
    frequency_multipliers: list[float] = (1, 2, 5, 10),
) -> pd.DataFrame:
    """Sweep over notional × slippage × frequency to produce $/week heatmap data.

    Returns DataFrame with columns: notional, slippage_bps, freq_multiplier,
    adj_edge_bps, trades_per_week, weekly_pnl_usd.
    """
    rows: list[dict] = []

    for notional in notional_levels:
        for slip in slippage_levels_bps:
            for freq_mult in frequency_multipliers:
                # Edge degrades with slippage
                adj_edge = baseline_edge_bps - 2 * slip  # round-trip slippage subtraction
                adj_edge = max(adj_edge, 0)
                trades = baseline_trades_per_week * freq_mult
                pnl = (adj_edge / 10_000.0) * trades * notional

                rows.append({
                    "notional_usd": notional,
                    "slippage_bps": slip,
                    "freq_multiplier": freq_mult,
                    "adj_edge_bps": adj_edge,
                    "trades_per_week": trades,
                    "weekly_pnl_usd": pnl,
                    "annual_pnl_usd": pnl * 52,
                })

    return pd.DataFrame(rows)


def capacity_economics_study(
    *,
    baseline_edge_bps: float = 44.0,
    baseline_trades_per_week: float = 4.0,
    horizon_result: pd.DataFrame | None = None,
    universe_result: pd.DataFrame | None = None,
    exec_aware_result: pd.DataFrame | None = None,
    notional: float = 100_000.0,
    cost_bps: float = 15.0,
    notional_levels: list[float] = (50_000, 100_000, 250_000, 500_000),
    slippage_levels_bps: list[float] = (5, 10, 15, 20),
    frequency_multipliers: list[float] = (1, 2, 5, 10),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full capacity economics study.

    Returns
    -------
    scorecard : scenario comparison table
    grid : sensitivity sweep data
    """
    log.info("═══ Branch E: Capacity economics ═══")

    scorecard = compute_capacity_scorecard(
        baseline_edge_bps,
        baseline_trades_per_week,
        horizon_result=horizon_result,
        universe_result=universe_result,
        exec_aware_result=exec_aware_result,
        notional=notional,
        cost_bps=cost_bps,
    )

    log.info("Capacity scorecard:")
    for _, row in scorecard.iterrows():
        log.info("  %s: $%.0f/week (%.0f%% vs baseline)",
                 row["scenario"], row["weekly_pnl_usd"], row["improvement_pct"])

    grid = sensitivity_grid(
        baseline_edge_bps,
        baseline_trades_per_week,
        notional_levels=list(notional_levels),
        slippage_levels_bps=list(slippage_levels_bps),
        frequency_multipliers=list(frequency_multipliers),
    )
    log.info("Sensitivity grid: %d scenarios", len(grid))

    return scorecard, grid
