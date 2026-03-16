"""Hard economic kill gates for discovered cells.

Every cell must pass all gates to advance. Lessons from exp018:
  - ≥100 non-overlapping trades
  - Positive median net expectancy
  - Positive trimmed mean net expectancy
  - ≥2 assets represented
  - No single asset >70% of cell
  - Survive conservative friction + stress test
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger("exp019.economic_gates")


def _dedup_trades_in_cluster(
    timestamps: pd.Series, min_gap_bars: int,
) -> pd.Series:
    """Return boolean mask of non-overlapping entries (simple temporal dedup)."""
    sorted_idx = timestamps.sort_values().index
    keep = pd.Series(False, index=timestamps.index)
    last_ts = None

    for idx in sorted_idx:
        ts = timestamps.loc[idx]
        if last_ts is None or (ts - last_ts).total_seconds() / 3600 >= min_gap_bars:
            keep.loc[idx] = True
            last_ts = ts

    return keep


def apply_gates(
    cell_economics: pd.DataFrame,
    cell_cards: list[dict],
    combined_predictions: pd.DataFrame,
    labels: np.ndarray,
    gates_cfg: dict,
) -> dict:
    """Apply kill gates to every discovered cell.

    Parameters
    ----------
    cell_economics : DataFrame from cell_extractor
    cell_cards : list of cell card dicts
    combined_predictions : OOF predictions DataFrame
    labels : cluster assignments
    gates_cfg : kill gate config from YAML

    Returns
    -------
    dict with:
        gate_results : DataFrame with pass/fail per gate per cell
        advancing_cells : list of cluster IDs that pass all gates
        killed_cells : list of cluster IDs that fail
        summary : overall verdict
    """
    min_trades = gates_cfg.get("min_non_overlapping_trades", 100)
    min_gap = gates_cfg.get("min_gap_bars", 4)
    median_floor = gates_cfg.get("median_net_bps_floor", 0.0)
    trim_frac = gates_cfg.get("trimmed_mean_trim", 0.10)
    trim_floor = gates_cfg.get("trimmed_mean_floor_bps", 0.0)
    min_assets = gates_cfg.get("min_assets", 2)
    max_asset_frac = gates_cfg.get("max_single_asset_frac", 0.70)
    stress_test = gates_cfg.get("stress_test_pass", True)

    results: list[dict] = []
    advancing: list[int] = []
    killed: list[int] = []

    for _, row in cell_economics.iterrows():
        cid = int(row["cluster_id"])
        mask = labels == cid
        cluster = combined_predictions[mask]

        # Gate 1: Non-overlapping trade count
        if "timestamp" in cluster.columns:
            dedup_mask = _dedup_trades_in_cluster(
                cluster["timestamp"], min_gap
            )
            n_deduped = dedup_mask.sum()
        else:
            n_deduped = len(cluster)

        gate_trade_count = n_deduped >= min_trades

        # Gate 2: Median net > floor
        gate_median = row["median_net_bps"] > median_floor

        # Gate 3: Trimmed mean > floor
        gate_trimmed_mean = row["trimmed_mean_net_bps"] > trim_floor

        # Gate 4: Asset diversity
        gate_assets = row["n_assets"] >= min_assets

        # Gate 5: No single-asset domination
        gate_concentration = row["top_asset_frac"] <= max_asset_frac

        # Gate 6: Stress test
        if stress_test:
            gate_stress = row["stressed_median_bps"] > 0
        else:
            gate_stress = True

        all_pass = all([
            gate_trade_count,
            gate_median,
            gate_trimmed_mean,
            gate_assets,
            gate_concentration,
            gate_stress,
        ])

        result = {
            "cluster_id": cid,
            "n_deduped_trades": int(n_deduped),
            "gate_trade_count": gate_trade_count,
            "gate_median_positive": gate_median,
            "gate_trimmed_mean_positive": gate_trimmed_mean,
            "gate_asset_diversity": gate_assets,
            "gate_no_concentration": gate_concentration,
            "gate_stress_test": gate_stress,
            "all_gates_pass": all_pass,
            # Economics
            "median_net_bps": row["median_net_bps"],
            "trimmed_mean_net_bps": row["trimmed_mean_net_bps"],
            "stressed_median_bps": row["stressed_median_bps"],
            "n_assets": int(row["n_assets"]),
            "top_asset_frac": row["top_asset_frac"],
            "pct_positive": row["pct_positive"],
        }
        results.append(result)

        if all_pass:
            advancing.append(cid)
        else:
            killed.append(cid)

    gate_df = pd.DataFrame(results)

    # Summary verdict
    if len(advancing) == 0:
        verdict = (
            "KILL — No discovered cell passes all economic gates. "
            "Latent states do not produce net-positive economics under "
            "conservative assumptions."
        )
    elif len(advancing) <= 2:
        verdict = (
            f"MARGINAL — {len(advancing)} cell(s) pass all gates. "
            "Cautious advancement warranted; verify OOS stability."
        )
    else:
        verdict = (
            f"ADVANCE — {len(advancing)} cells pass all gates. "
            "Proceed to out-of-sample validation and capacity sizing."
        )

    log.info(
        "Gate results: %d advancing, %d killed out of %d cells",
        len(advancing), len(killed), len(results),
    )
    log.info("Verdict: %s", verdict)

    return {
        "gate_results": gate_df,
        "advancing_cells": advancing,
        "killed_cells": killed,
        "verdict": verdict,
    }
