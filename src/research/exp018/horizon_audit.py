"""Horizon audit for exp018 — verify whether horizons produce distinct outcomes.

Tests whether the stop/TP logic causes identical exits regardless of
nominal horizon, which would confirm the identical-median red flag.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from src.utils.logging import get_logger

log = get_logger("horizon_audit")


def exit_reason_breakdown(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Compute exit reason fractions per (pair, rule, spread_type, horizon).

    Returns
    -------
    DataFrame with columns: pair, rule, spread_type, horizon_h,
    n_trades, pct_stop, pct_tp, pct_horizon
    """
    if trades_df.empty:
        return pd.DataFrame()

    group_cols = ["pair", "rule", "spread_type", "horizon_h"]
    rows = []

    for keys, group in trades_df.groupby(group_cols):
        pair, rule, spread_type, horizon_h = keys
        n = len(group)
        reasons = group["exit_reason"].value_counts(normalize=True)

        rows.append({
            "pair": pair,
            "rule": rule,
            "spread_type": spread_type,
            "horizon_h": horizon_h,
            "n_trades": n,
            "pct_stop": round(reasons.get("stop_loss", 0) * 100, 1),
            "pct_tp": round(reasons.get("take_profit", 0) * 100, 1),
            "pct_horizon": round(reasons.get("horizon", 0) * 100, 1),
        })

    return pd.DataFrame(rows)


def horizon_sensitivity_test(trades_df: pd.DataFrame) -> pd.DataFrame:
    """KS-test on net_bps distributions across horizons for each combo.

    For each (pair, rule, spread_type), compare the 12h vs 24h vs 48h
    net_bps distributions. If p-value > 0.05, the distributions are
    statistically identical — confirming the horizon is a no-op.

    Returns
    -------
    DataFrame with: pair, rule, spread_type, horizons_compared,
    ks_stat, p_value, identical_flag
    """
    if trades_df.empty:
        return pd.DataFrame()

    group_cols = ["pair", "rule", "spread_type"]
    horizons = sorted(trades_df["horizon_h"].unique())
    rows = []

    for keys, group in trades_df.groupby(group_cols):
        pair, rule, spread_type = keys

        for i in range(len(horizons)):
            for j in range(i + 1, len(horizons)):
                h1, h2 = horizons[i], horizons[j]
                d1 = group[group["horizon_h"] == h1]["net_spread_bps"].values
                d2 = group[group["horizon_h"] == h2]["net_spread_bps"].values

                if len(d1) < 5 or len(d2) < 5:
                    continue

                ks_stat, p_val = sp_stats.ks_2samp(d1, d2)

                rows.append({
                    "pair": pair,
                    "rule": rule,
                    "spread_type": spread_type,
                    "h1": int(h1),
                    "h2": int(h2),
                    "n_h1": len(d1),
                    "n_h2": len(d2),
                    "ks_stat": round(ks_stat, 4),
                    "p_value": round(p_val, 4),
                    "identical": p_val > 0.05,
                })

    result = pd.DataFrame(rows)
    if not result.empty:
        n_identical = result["identical"].sum()
        n_total = len(result)
        log.info(f"Horizon KS test: {n_identical}/{n_total} pairs are statistically "
                 f"identical across horizons")
    return result


def exit_bar_comparison(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Compare mean/median exit bars across horizons per combo.

    Returns
    -------
    DataFrame with: pair, rule, spread_type, horizon_h,
    mean_exit_bar, median_exit_bar, max_exit_bar
    """
    if trades_df.empty:
        return pd.DataFrame()

    group_cols = ["pair", "rule", "spread_type", "horizon_h"]
    rows = []

    for keys, group in trades_df.groupby(group_cols):
        pair, rule, spread_type, horizon_h = keys
        bars = group["holding_bars"].values

        rows.append({
            "pair": pair,
            "rule": rule,
            "spread_type": spread_type,
            "horizon_h": int(horizon_h),
            "mean_exit_bar": round(float(np.mean(bars)), 1),
            "median_exit_bar": float(np.median(bars)),
            "max_exit_bar": int(np.max(bars)),
            "pct_at_horizon": round(
                float(np.mean(bars == horizon_h)) * 100, 1),
        })

    return pd.DataFrame(rows)


def run_horizon_audit(trades_df: pd.DataFrame) -> dict:
    """Run the full horizon audit.

    Returns
    -------
    dict with:
        exit_breakdown (DataFrame),
        ks_results (DataFrame),
        exit_bar_stats (DataFrame),
        summary (dict with aggregate stats)
    """
    breakdown = exit_reason_breakdown(trades_df)
    ks = horizon_sensitivity_test(trades_df)
    exit_bars = exit_bar_comparison(trades_df)

    # Aggregate summary
    summary = {}
    if not ks.empty:
        summary["total_pairs_tested"] = len(ks)
        summary["identical_pairs"] = int(ks["identical"].sum())
        summary["pct_identical"] = round(
            ks["identical"].mean() * 100, 1)

    if not breakdown.empty:
        summary["overall_pct_stop"] = round(breakdown["pct_stop"].mean(), 1)
        summary["overall_pct_tp"] = round(breakdown["pct_tp"].mean(), 1)
        summary["overall_pct_horizon"] = round(breakdown["pct_horizon"].mean(), 1)

    log.info(f"Horizon audit summary: {summary}")
    return {
        "exit_breakdown": breakdown,
        "ks_results": ks,
        "exit_bar_stats": exit_bars,
        "summary": summary,
    }
