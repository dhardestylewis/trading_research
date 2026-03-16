"""Gross-Move Atlas — magnitude-first economic screening.

For each cell in the (asset × horizon × regime × entry_convention) grid,
compute the forward-return distribution and determine whether gross move
magnitude plausibly clears a friction hurdle.

This is Phase 1 of the magnitude-first program.  No ML.
The atlas table is the FIRST page of the exp011 report.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger("gross_move_atlas")


# ── Types ────────────────────────────────────────────────────────────

@dataclass
class AtlasConfig:
    """Parameters for atlas construction."""
    horizons: list[int]                     # bar-horizons, e.g. [1, 4]
    friction_bps: float = 30.0             # round-trip friction hurdle
    gross_thresholds_bps: list[float] = None  # exceedance thresholds
    min_trades: int = 100                   # minimum trades for viable cell
    entry_conventions: list[dict] = None    # [{name, type, offset_bps}, ...]
    regime_slices: list[dict] = None        # [{name, filter}, ...]

    def __post_init__(self):
        if self.gross_thresholds_bps is None:
            self.gross_thresholds_bps = [20.0, 30.0, 40.0, 50.0]
        if self.entry_conventions is None:
            self.entry_conventions = [
                {"name": "market_next_open", "type": "market", "offset_bps": 0.0},
            ]
        if self.regime_slices is None:
            self.regime_slices = [{"name": "all", "filter": None}]


# ── Atlas construction ───────────────────────────────────────────────

def _compute_forward_returns(
    panel: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Compute raw forward returns per asset.

    Returns DataFrame aligned to panel index with:
      fwd_ret_{horizon}h : raw forward return (close-to-close)
      fwd_ret_abs_{horizon}h : absolute forward return in bps
      next_open, next_low, next_high, next_close : for entry simulation
    """
    parts: list[pd.DataFrame] = []

    for _asset, g in panel.groupby("asset", sort=False):
        g = g.sort_values("timestamp")
        c = g["close"]
        ret = c.shift(-horizon) / c - 1

        out = pd.DataFrame(index=g.index)
        out[f"fwd_ret_{horizon}h"] = ret
        out[f"fwd_ret_abs_{horizon}h_bps"] = ret.abs() * 10_000

        # Entry simulation fields (next bar)
        out["next_open"] = g["open"].shift(-1)
        out["next_low"] = g["low"].shift(-1)
        out["next_high"] = g["high"].shift(-1)
        out["next_close"] = g["close"].shift(-1)

        # For the forward close at horizon
        out[f"close_at_{horizon}h"] = c.shift(-horizon)

        parts.append(out)

    return pd.concat(parts).loc[panel.index]


def _apply_entry_convention(
    row_data: pd.DataFrame,
    entry: dict,
    horizon: int,
) -> pd.DataFrame:
    """Compute gross return under a specific entry convention.

    Returns a copy with columns:
      entry_price : computed entry price
      exit_price  : close at horizon
      gross_ret_bps : (exit / entry - 1) × 10000
      filled : bool, whether the entry would have been filled
    """
    df = row_data.copy()
    entry_type = entry.get("type", "market")
    offset_bps = entry.get("offset_bps", 0.0)

    if entry_type == "market":
        # Market entry at next-bar open
        df["entry_price"] = df["next_open"]
        df["filled"] = df["next_open"].notna()
    elif entry_type == "passive":
        # Passive limit at open - offset_bps
        offset_frac = offset_bps / 10_000.0
        limit_price = df["next_open"] * (1 - offset_frac)
        df["entry_price"] = limit_price
        # Fill if next-bar low touches the limit
        df["filled"] = df["next_low"] <= limit_price
    else:
        raise ValueError(f"Unknown entry type: {entry_type}")

    df["exit_price"] = df[f"close_at_{horizon}h"]
    df["gross_ret_bps"] = (df["exit_price"] / df["entry_price"] - 1) * 10_000

    return df


def _apply_regime_filter(
    df: pd.DataFrame,
    regime_filter: Optional[str],
) -> pd.DataFrame:
    """Apply a regime filter expression (e.g. 'regime_vol_high == 1')."""
    if regime_filter is None:
        return df
    try:
        mask = df.eval(regime_filter)
        return df[mask]
    except Exception as e:
        log.warning("Failed to apply regime filter '%s': %s", regime_filter, e)
        return df


def _cell_statistics(
    gross_ret_bps: pd.Series,
    filled_mask: pd.Series,
    thresholds_bps: list[float],
) -> dict:
    """Compute distribution statistics for one cell."""
    # Only use filled trades
    filled = gross_ret_bps[filled_mask].dropna()
    n = len(filled)

    if n == 0:
        return {
            "trade_count": 0,
            "fill_count": 0,
            "fill_rate": 0.0,
        }

    total_signals = len(gross_ret_bps.dropna())
    fill_rate = n / total_signals if total_signals > 0 else 0.0

    stats = {
        "trade_count": total_signals,
        "fill_count": n,
        "fill_rate": fill_rate,
        "gross_bps_mean": filled.mean(),
        "gross_bps_median": filled.median(),
        "gross_bps_p25": filled.quantile(0.25),
        "gross_bps_p75": filled.quantile(0.75),
        "gross_bps_p90": filled.quantile(0.90),
        "gross_bps_std": filled.std(),
        "gross_bps_abs_mean": filled.abs().mean(),
        "gross_bps_abs_median": filled.abs().median(),
        "gross_bps_abs_p75": filled.abs().quantile(0.75),
        "gross_bps_abs_p90": filled.abs().quantile(0.90),
    }

    # Exceedance fractions (absolute gross move)
    for t in thresholds_bps:
        stats[f"frac_abs_gt_{int(t)}bps"] = (filled.abs() > t).mean()

    # Signed exceedance (directional: gross move > threshold)
    for t in thresholds_bps:
        stats[f"frac_gt_{int(t)}bps"] = (filled > t).mean()
        stats[f"frac_lt_neg_{int(t)}bps"] = (filled < -t).mean()

    # Adverse selection proxy: mean return of losing filled trades
    losers = filled[filled < 0]
    stats["adverse_sel_mean_bps"] = losers.mean() if len(losers) > 0 else 0.0
    stats["adverse_sel_count"] = len(losers)

    return stats


def build_atlas(
    panel: pd.DataFrame,
    config: AtlasConfig,
    regime_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the gross-move atlas across all cells.

    Parameters
    ----------
    panel : OHLCV panel with columns [asset, timestamp, open, high, low, close, volume]
    config : AtlasConfig with horizons, entries, regime slices
    regime_df : DataFrame aligned to panel with regime flag columns.
                If None, only 'all' regime slice can be used.

    Returns
    -------
    DataFrame with one row per (asset, horizon, regime, entry_convention) cell.
    """
    rows: list[dict] = []
    assets = panel["asset"].unique()

    log.info(
        "Building atlas: %d assets × %d horizons × %d regimes × %d entries",
        len(assets), len(config.horizons), len(config.regime_slices),
        len(config.entry_conventions),
    )

    for horizon in config.horizons:
        log.info("── Horizon: %dh ──", horizon)

        # Compute forward returns for this horizon
        fwd = _compute_forward_returns(panel, horizon)
        panel_with_fwd = pd.concat([panel, fwd], axis=1)

        # Merge regime flags if available
        if regime_df is not None:
            for col in regime_df.columns:
                if col not in panel_with_fwd.columns:
                    panel_with_fwd[col] = regime_df[col].values

        for asset in assets:
            asset_data = panel_with_fwd[panel_with_fwd["asset"] == asset].copy()
            if len(asset_data) < 10:
                continue

            for regime in config.regime_slices:
                regime_name = regime["name"]
                regime_filter = regime.get("filter")

                sliced = _apply_regime_filter(asset_data, regime_filter)
                if len(sliced) < 5:
                    continue

                for entry in config.entry_conventions:
                    entry_name = entry["name"]

                    entry_data = _apply_entry_convention(sliced, entry, horizon)
                    valid = entry_data.dropna(subset=["gross_ret_bps"])

                    if len(valid) < 5:
                        continue

                    stats = _cell_statistics(
                        valid["gross_ret_bps"],
                        valid["filled"],
                        config.gross_thresholds_bps,
                    )

                    row = {
                        "asset": asset,
                        "horizon": f"{horizon}h",
                        "regime": regime_name,
                        "entry": entry_name,
                        **stats,
                    }
                    rows.append(row)

    atlas = pd.DataFrame(rows)
    log.info("Atlas built: %d cells", len(atlas))
    return atlas


# ── Ranking and filtering ────────────────────────────────────────────

def rank_cells(
    atlas: pd.DataFrame,
    min_trades: int = 100,
    friction_bps: float = 30.0,
    rank_by: str = "gross_bps_abs_p75",
) -> pd.DataFrame:
    """Rank cells by friction-clearing potential.

    Filters to cells with sufficient trades, then ranks by the chosen
    metric in descending order.

    Returns atlas subset with 'viable' flag and 'rank' column.
    """
    if atlas.empty:
        return atlas

    df = atlas.copy()

    # Filter by minimum trades
    df["has_enough_trades"] = df["fill_count"] >= min_trades

    # Check if p75 or median absolute gross bps clears friction
    df["median_clears_friction"] = df["gross_bps_abs_median"] > friction_bps
    df["p75_clears_friction"] = df["gross_bps_abs_p75"] > friction_bps

    # A cell is viable if it has enough trades AND (median or p75 clears friction)
    df["viable"] = (
        df["has_enough_trades"]
        & (df["median_clears_friction"] | df["p75_clears_friction"])
    )

    # Rank all cells by the chosen metric
    df = df.sort_values(rank_by, ascending=False, na_position="last")
    df["rank"] = range(1, len(df) + 1)

    viable_count = df["viable"].sum()
    log.info(
        "Ranked %d cells: %d viable (≥%d trades, abs gross clears %.0f bps)",
        len(df), viable_count, min_trades, friction_bps,
    )

    return df


def check_kill_gate(
    ranked: pd.DataFrame,
    min_viable_cells: int = 3,
) -> tuple[bool, str]:
    """Check whether the atlas passes the kill gate.

    Returns (passes, reason).
    """
    viable = ranked[ranked["viable"]] if "viable" in ranked.columns else ranked.head(0)
    n_viable = len(viable)

    if n_viable >= min_viable_cells:
        return True, f"PASS: {n_viable} viable cells found (need {min_viable_cells})"
    else:
        return False, f"FAIL: only {n_viable} viable cells found (need {min_viable_cells}). STOP."


def atlas_summary_table(
    atlas: pd.DataFrame,
    top_n: int = 30,
) -> pd.DataFrame:
    """Produce the summary table for the report (gross bps distribution).

    This is the FIRST TABLE in the report per program rule.
    """
    if atlas.empty:
        return atlas

    cols = [
        "rank", "asset", "horizon", "regime", "entry",
        "fill_count", "fill_rate",
        "gross_bps_mean", "gross_bps_median",
        "gross_bps_p75", "gross_bps_p90",
        "frac_abs_gt_30bps", "frac_abs_gt_50bps",
        "adverse_sel_mean_bps",
        "viable",
    ]
    available = [c for c in cols if c in atlas.columns]
    return atlas[available].head(top_n)
