"""Delay simulator for execution robustness analysis.

Thin wrapper around robustness_grid.shift_predictions for use in the
main experiment runner.  Kept separate from the full grid module for
simpler direct use.
"""
from __future__ import annotations

import pandas as pd

from src.diagnostics.robustness_grid import shift_predictions  # re-export


def simulate_with_delay(
    preds: pd.DataFrame,
    delay_bars: int,
    prob_col: str = "y_pred_prob",
) -> pd.DataFrame:
    """Return a copy of predictions with the signal shifted by *delay_bars*.

    Convenience wrapper that matches the backtest module API convention.
    """
    return shift_predictions(preds, delay_bars, prob_col)
