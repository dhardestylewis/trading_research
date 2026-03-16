"""SOL-only robustness analysis wrapper (Branch 2).

Filters predictions and panel to SOL-USD, then runs:
  1. Fill simulation grid across fill types and thresholds
  2. Standard delay × cost robustness grid (from robustness_grid.py)
"""
from __future__ import annotations

import pandas as pd

from src.diagnostics.fill_simulation import fill_simulation_grid
from src.diagnostics.robustness_grid import robustness_grid
from src.utils.logging import get_logger

log = get_logger("sol_robustness")


def sol_only_robustness(
    panel: pd.DataFrame,
    preds: pd.DataFrame,
    model_name: str = "lightgbm",
    thresholds: list[float] = (0.50, 0.55, 0.60),
    fill_cost_bps: float = 15.0,
) -> dict[str, pd.DataFrame]:
    """Run comprehensive SOL-only robustness analysis.

    Parameters
    ----------
    panel : full OHLCV panel (will be filtered to SOL-USD).
    preds : full predictions DataFrame (will be filtered to SOL-USD + model).
    model_name : which model to analyse (default: lightgbm).
    thresholds : score thresholds for fill simulation.
    fill_cost_bps : base cost assumption.

    Returns
    -------
    Dictionary with keys 'fill_grid' and 'delay_grid', each a DataFrame.
    """
    # Filter to SOL-only
    sol_panel = panel[panel["asset"] == "SOL-USD"].copy()
    sol_preds = preds[
        (preds["asset"] == "SOL-USD") & (preds["model_name"] == model_name)
    ].copy()

    log.info("SOL-only: %d panel rows, %d prediction rows", len(sol_panel), len(sol_preds))

    if len(sol_preds) < 10:
        log.warning("Too few SOL predictions (%d) — skipping robustness", len(sol_preds))
        return {"fill_grid": pd.DataFrame(), "delay_grid": pd.DataFrame()}

    # Ensure tz-naive timestamps for matching
    for df in (sol_panel, sol_preds):
        if hasattr(df["timestamp"].dtype, "tz") and df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    # Branch 2a: Fill simulation grid
    log.info("Running fill simulation grid…")
    fill_grid = fill_simulation_grid(
        sol_panel, sol_preds,
        thresholds=list(thresholds),
        cost_bps=fill_cost_bps,
    )

    # Branch 2b: Standard delay × cost grid (SOL-only slice)
    log.info("Running delay × cost robustness grid…")
    delay_grid = robustness_grid(
        sol_preds,
        delays=[0, 1, 2],
        threshold=0.55,
    )

    return {
        "fill_grid": fill_grid,
        "delay_grid": delay_grid,
    }
