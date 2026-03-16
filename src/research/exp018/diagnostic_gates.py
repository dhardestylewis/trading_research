"""Corrected diagnostic gates for exp018.

Operates at PAIR level, not combo level, to prevent search artifacts
from overriding the primary economic result.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from src.utils.logging import get_logger

log = get_logger("diagnostic_gates")


def evaluate_pair_level_gates(trades_df: pd.DataFrame,
                              gates_cfg: dict) -> dict:
    """Evaluate corrected pair-level kill gates.

    Gates:
    1. ≥ min_pairs with ≥ min_non_overlapping_trades
    2. Pair-level median realized net spread > pair_level_median_net_bps
    3. Pair-level mean realized net spread > pair_level_mean_floor_bps

    Returns
    -------
    dict with per-gate results, per-pair details, and overall verdict.
    """
    results = {
        "pair_details": {},
        "gates": {},
        "overall_pass": False,
        "verdict": "FAIL",
    }

    if trades_df.empty:
        results["verdict"] = "FAIL — no trades"
        return results

    # Per-pair statistics
    qualifying_pairs = 0
    pair_details = {}

    for pair, group in trades_df.groupby("pair"):
        vals = group["net_spread_bps"].values
        n = len(vals)

        detail = {
            "n_trades": n,
            "median_net_bps": round(float(np.median(vals)), 2),
            "mean_net_bps": round(float(np.mean(vals)), 2),
            "frac_positive": round(float(np.mean(vals > 0)), 3),
            "passes_count": n >= gates_cfg["min_non_overlapping_trades"],
            "passes_median": float(np.median(vals)) > gates_cfg["pair_level_median_net_bps"],
            "passes_mean": float(np.mean(vals)) > gates_cfg["pair_level_mean_floor_bps"],
        }
        detail["passes_all"] = (detail["passes_count"] and
                                detail["passes_median"] and
                                detail["passes_mean"])

        if detail["passes_all"]:
            qualifying_pairs += 1

        pair_details[pair] = detail

    results["pair_details"] = pair_details

    # Gate 1: enough qualifying pairs
    results["gates"]["qualifying_pairs"] = {
        "value": qualifying_pairs,
        "threshold": gates_cfg["min_pairs"],
        "pass": qualifying_pairs >= gates_cfg["min_pairs"],
    }

    # Gate 2: pair-level median
    pair_medians = [d["median_net_bps"] for d in pair_details.values()]
    median_of_medians = float(np.median(pair_medians)) if pair_medians else 0.0
    results["gates"]["pair_median_net_bps"] = {
        "value": round(median_of_medians, 2),
        "threshold": gates_cfg["pair_level_median_net_bps"],
        "pass": median_of_medians > gates_cfg["pair_level_median_net_bps"],
    }

    # Gate 3: pair-level mean
    pair_means = [d["mean_net_bps"] for d in pair_details.values()]
    median_of_means = float(np.median(pair_means)) if pair_means else 0.0
    results["gates"]["pair_mean_net_bps"] = {
        "value": round(median_of_means, 2),
        "threshold": gates_cfg["pair_level_mean_floor_bps"],
        "pass": median_of_means > gates_cfg["pair_level_mean_floor_bps"],
    }

    # Overall
    results["overall_pass"] = all(g["pass"] for g in results["gates"].values())

    if results["overall_pass"]:
        results["verdict"] = "PASS — pair-level economics positive"
    else:
        # Determine if fixable or kill
        any_pair_positive = any(d["passes_median"] for d in pair_details.values())
        if any_pair_positive:
            results["verdict"] = "FAIL — some pairs marginal, direction rules may be improvable"
        else:
            results["verdict"] = "KILL — all pairs negative at pair level, opportunity untradeable"

    log.info(f"Corrected gate verdict: {results['verdict']}")
    return results


def compare_rule_families(heuristic_trades: pd.DataFrame,
                          regression_trades: pd.DataFrame,
                          gates_cfg: dict) -> dict:
    """Run corrected gates on both heuristic and regression trades.

    Returns
    -------
    dict with:
        heuristic_gates, regression_gates, comparison_summary
    """
    heuristic_gates = evaluate_pair_level_gates(heuristic_trades, gates_cfg)
    regression_gates = evaluate_pair_level_gates(regression_trades, gates_cfg)

    # Build comparison summary
    comparison = {
        "heuristic_verdict": heuristic_gates["verdict"],
        "regression_verdict": regression_gates["verdict"],
    }

    # Per-pair comparison
    pair_comparison = []
    all_pairs = set()
    if heuristic_gates["pair_details"]:
        all_pairs.update(heuristic_gates["pair_details"].keys())
    if regression_gates["pair_details"]:
        all_pairs.update(regression_gates["pair_details"].keys())

    for pair in sorted(all_pairs):
        h_detail = heuristic_gates["pair_details"].get(pair, {})
        r_detail = regression_gates["pair_details"].get(pair, {})
        pair_comparison.append({
            "pair": pair,
            "heuristic_n": h_detail.get("n_trades", 0),
            "heuristic_median": h_detail.get("median_net_bps", None),
            "heuristic_mean": h_detail.get("mean_net_bps", None),
            "regression_n": r_detail.get("n_trades", 0),
            "regression_median": r_detail.get("median_net_bps", None),
            "regression_mean": r_detail.get("mean_net_bps", None),
            "regression_improves": (
                r_detail.get("median_net_bps", -999) >
                h_detail.get("median_net_bps", -999)
            ),
        })

    comparison["pair_comparison"] = pd.DataFrame(pair_comparison)

    return {
        "heuristic_gates": heuristic_gates,
        "regression_gates": regression_gates,
        "comparison": comparison,
    }
