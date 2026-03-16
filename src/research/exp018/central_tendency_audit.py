"""Central tendency audit for exp018 — median vs mean vs trimmed mean.

Identifies combos where median masks negative mean (fat-tail illusion)
and combos with extreme skewness from a few large winners.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from src.utils.logging import get_logger

log = get_logger("central_tendency_audit")


def compute_central_tendency(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Compute multiple central tendency measures per combo.

    For each (pair, rule, spread_type), computes on deduplicated trades:
    - median, mean, 10% trimmed mean
    - IQR, std
    - skewness, kurtosis
    - fraction positive
    - flag: median > 0 but mean < 0

    Returns
    -------
    DataFrame with one row per combo and all statistics.
    """
    if trades_df.empty:
        return pd.DataFrame()

    group_cols = ["pair", "rule", "spread_type"]
    rows = []

    for keys, group in trades_df.groupby(group_cols):
        pair, rule, spread_type = keys
        vals = group["net_spread_bps"].values
        n = len(vals)

        if n < 3:
            continue

        median_val = float(np.median(vals))
        mean_val = float(np.mean(vals))

        # 10% trimmed mean
        trim_frac = min(0.1, (n - 1) / (2 * n))
        trimmed_mean = float(sp_stats.trim_mean(vals, trim_frac))

        q25, q75 = np.percentile(vals, [25, 75])
        iqr = float(q75 - q25)
        std_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0

        skew = float(sp_stats.skew(vals)) if n > 2 else 0.0
        kurt = float(sp_stats.kurtosis(vals)) if n > 3 else 0.0

        frac_positive = float(np.mean(vals > 0))

        rows.append({
            "pair": pair,
            "rule": rule,
            "spread_type": spread_type,
            "n_trades": n,
            "median_net_bps": round(median_val, 2),
            "mean_net_bps": round(mean_val, 2),
            "trimmed_mean_bps": round(trimmed_mean, 2),
            "iqr_bps": round(iqr, 2),
            "std_bps": round(std_val, 2),
            "skewness": round(skew, 3),
            "kurtosis": round(kurt, 3),
            "frac_positive": round(frac_positive, 3),
            "median_positive_mean_negative": median_val > 0 and mean_val < 0,
            "high_skew": skew > 2.0,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        n_masked = result["median_positive_mean_negative"].sum()
        n_skew = result["high_skew"].sum()
        log.info(f"Central tendency: {n_masked}/{len(result)} combos have "
                 f"median > 0 but mean < 0; {n_skew} have skew > 2")
    return result


def pair_level_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Compute pair-level aggregate central tendency (the decisive table).

    This is the table that should be the sole top-level gate.

    Returns
    -------
    DataFrame with one row per pair.
    """
    if trades_df.empty:
        return pd.DataFrame()

    rows = []
    for pair, group in trades_df.groupby("pair"):
        vals = group["net_spread_bps"].values
        n = len(vals)

        if n < 3:
            continue

        trim_frac = min(0.1, (n - 1) / (2 * n))

        rows.append({
            "pair": pair,
            "n_trades": n,
            "median_net_bps": round(float(np.median(vals)), 2),
            "mean_net_bps": round(float(np.mean(vals)), 2),
            "trimmed_mean_bps": round(float(sp_stats.trim_mean(vals, trim_frac)), 2),
            "std_bps": round(float(np.std(vals, ddof=1)), 2),
            "frac_positive": round(float(np.mean(vals > 0)), 3),
            "p25_bps": round(float(np.percentile(vals, 25)), 2),
            "p75_bps": round(float(np.percentile(vals, 75)), 2),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        log.info(f"Pair-level summary:\n{result.to_string(index=False)}")
    return result


def run_central_tendency_audit(trades_df: pd.DataFrame) -> dict:
    """Run full central tendency audit.

    Returns
    -------
    dict with:
        combo_stats (DataFrame) — per-combo statistics
        pair_stats (DataFrame) — pair-level aggregates
        flags (dict) — summary of red flags found
    """
    combo_stats = compute_central_tendency(trades_df)
    pair_stats = pair_level_summary(trades_df)

    flags = {}
    if not combo_stats.empty:
        flags["combos_with_masked_mean"] = int(
            combo_stats["median_positive_mean_negative"].sum())
        flags["combos_with_high_skew"] = int(
            combo_stats["high_skew"].sum())
        flags["total_combos"] = len(combo_stats)

    if not pair_stats.empty:
        flags["pairs_with_positive_median"] = int(
            (pair_stats["median_net_bps"] > 0).sum())
        flags["pairs_with_positive_mean"] = int(
            (pair_stats["mean_net_bps"] > 0).sum())
        flags["total_pairs"] = len(pair_stats)

    return {
        "combo_stats": combo_stats,
        "pair_stats": pair_stats,
        "flags": flags,
    }
