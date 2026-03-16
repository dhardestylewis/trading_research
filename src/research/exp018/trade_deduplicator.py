"""Trade deduplicator for exp018 — detect and remove overlapping trades.

Two deduplication passes:
1. Cross-horizon collapse: same entry → keep only the canonical horizon
2. Temporal dedup: entries within min_gap_bars of each other → keep first only
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from src.utils.logging import get_logger

log = get_logger("trade_deduplicator")


def collapse_cross_horizon(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate trades that are identical across horizons.

    When stop/TP fires before the shortest horizon, trades at 12/24/48h
    are literally the same trade. Keep only the row matching the actual
    exit horizon (or the shortest nominal horizon if exit < shortest).

    Returns
    -------
    DataFrame with duplicates removed, plus 'was_horizon_dup' flag on kept rows.
    """
    if trades_df.empty:
        return trades_df

    group_cols = ["pair", "rule", "spread_type", "entry_time", "direction"]
    results = []

    for keys, group in trades_df.groupby(group_cols):
        if len(group) == 1:
            row = group.iloc[0].copy()
            row["horizons_collapsed"] = 1
            results.append(row)
            continue

        # Sort by horizon ascending
        group = group.sort_values("horizon_h")

        # Check if all trades have same exit bar and exit reason
        exit_bars = group["holding_bars"].unique()
        exit_reasons = group["exit_reason"].unique()

        if len(exit_bars) == 1 and len(exit_reasons) == 1:
            # Identical trade across horizons — keep shortest horizon
            row = group.iloc[0].copy()
            row["horizons_collapsed"] = len(group)
            results.append(row)
        else:
            # Genuinely different across horizons — keep all
            for _, row in group.iterrows():
                row = row.copy()
                row["horizons_collapsed"] = 1
                results.append(row)

    result = pd.DataFrame(results)
    n_original = len(trades_df)
    n_after = len(result)
    n_collapsed = result["horizons_collapsed"].sum() - len(result)
    log.info(f"Cross-horizon collapse: {n_original} → {n_after} trades "
             f"({n_original - n_after} removed, {n_collapsed} were duplicates)")
    return result


def temporal_dedup(trades_df: pd.DataFrame,
                   min_gap_bars: int = 12) -> pd.DataFrame:
    """Remove temporally overlapping trades.

    For each (pair, rule, spread_type), sort by entry time and keep
    only entries that are at least min_gap_bars apart from the previous
    kept entry.

    Returns
    -------
    DataFrame with temporally deduplicated trades.
    """
    if trades_df.empty:
        return trades_df

    group_cols = ["pair", "rule", "spread_type"]
    results = []

    for keys, group in trades_df.groupby(group_cols):
        group = group.sort_values("entry_time").reset_index(drop=True)
        entry_times = pd.to_datetime(group["entry_time"])

        kept_indices = [0]
        last_kept_time = entry_times.iloc[0]

        for i in range(1, len(group)):
            gap_hours = (entry_times.iloc[i] - last_kept_time).total_seconds() / 3600
            if gap_hours >= min_gap_bars:
                kept_indices.append(i)
                last_kept_time = entry_times.iloc[i]

        results.append(group.iloc[kept_indices])

    result = pd.concat(results, ignore_index=True)
    n_before = len(trades_df)
    log.info(f"Temporal dedup (gap={min_gap_bars}h): {n_before} → {len(result)} trades "
             f"({n_before - len(result)} removed)")
    return result


def full_dedup_pipeline(trades_df: pd.DataFrame,
                        min_gap_bars: int = 12) -> dict:
    """Run the full dedup pipeline and return diagnostic info.

    Returns
    -------
    dict with:
        original_count, after_horizon_collapse, after_temporal_dedup,
        deduplicated_trades (DataFrame),
        horizon_collapse_stats (DataFrame), temporal_dedup_stats (DataFrame)
    """
    original_count = len(trades_df)

    # Step 1: cross-horizon collapse
    after_horizon = collapse_cross_horizon(trades_df)

    # Step 2: temporal dedup
    after_temporal = temporal_dedup(after_horizon, min_gap_bars)

    # Build per-group stats
    group_cols = ["pair", "rule", "spread_type"]
    horizon_stats_rows = []
    temporal_stats_rows = []

    for keys, orig_group in trades_df.groupby(group_cols):
        pair, rule, spread_type = keys
        n_orig = len(orig_group)

        # After horizon collapse
        mask = ((after_horizon["pair"] == pair) &
                (after_horizon["rule"] == rule) &
                (after_horizon["spread_type"] == spread_type))
        n_horizon = mask.sum()

        horizon_stats_rows.append({
            "pair": pair, "rule": rule, "spread_type": spread_type,
            "original": n_orig, "after_collapse": n_horizon,
            "reduction_pct": round((1 - n_horizon / n_orig) * 100, 1) if n_orig > 0 else 0,
        })

        # After temporal dedup
        mask2 = ((after_temporal["pair"] == pair) &
                 (after_temporal["rule"] == rule) &
                 (after_temporal["spread_type"] == spread_type))
        n_temporal = mask2.sum()

        temporal_stats_rows.append({
            "pair": pair, "rule": rule, "spread_type": spread_type,
            "after_collapse": n_horizon, "after_dedup": n_temporal,
            "reduction_pct": round((1 - n_temporal / n_horizon) * 100, 1) if n_horizon > 0 else 0,
        })

    return {
        "original_count": original_count,
        "after_horizon_collapse": len(after_horizon),
        "after_temporal_dedup": len(after_temporal),
        "deduplicated_trades": after_temporal,
        "horizon_collapse_stats": pd.DataFrame(horizon_stats_rows),
        "temporal_dedup_stats": pd.DataFrame(temporal_stats_rows),
    }
