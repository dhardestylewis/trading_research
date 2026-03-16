"""Compute forward-return and cost-adjusted labels."""
from __future__ import annotations
import pandas as pd


def compute_forward_labels(
    panel: pd.DataFrame,
    horizons: list[int],
    one_way_cost_bps: float = 15.0,
) -> pd.DataFrame:
    """Compute forward return labels per asset.

    Args:
        panel: DataFrame with columns [asset, timestamp, close], sorted by (asset, timestamp).
        horizons: list of bar-horizons (e.g. [1, 4]).
        one_way_cost_bps: cost in basis points for the *label_cost_regime* used to define
                          the tradable target.  Round-trip cost = 2 × one_way_cost_bps.

    Returns:
        DataFrame aligned to panel index with label columns.
    """
    cost_frac = one_way_cost_bps / 10_000.0  # one-way
    rt_cost = 2 * cost_frac  # round trip entry + exit

    parts: list[pd.DataFrame] = []

    for _asset, g in panel.groupby("asset", sort=False):
        g = g.sort_values("timestamp")
        c = g["close"]
        labels = pd.DataFrame(index=g.index)

        for h in horizons:
            fwd = c.shift(-h) / c - 1
            labels[f"fwd_ret_{h}h"] = fwd
            labels[f"fwd_sign_{h}h"] = (fwd > 0).astype(int)
            labels[f"fwd_ret_after_cost_{h}h"] = fwd - rt_cost
            labels[f"fwd_profitable_{h}h"] = (labels[f"fwd_ret_after_cost_{h}h"] > 0).astype(int)

        parts.append(labels)

    return pd.concat(parts).loc[panel.index]
