"""Family canary diagnostic — exp010 core module.

Deployment-style validation of the L1 family (SOL, SUI, NEAR, APT)
under the frozen architecture from exp009 (family-pooled training,
asset-specific deployment lanes).

Three analysis functions:
  1. family_execution_validation — per-asset simulated paper-trade logs
  2. cross_asset_fill_profile   — fill-quality comparison + KS tests
  3. family_weekly_pnl          — aggregated portfolio PnL series
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from src.diagnostics.dilution_diagnosis import _run_pooled_backtest, _aggregate_per_asset
from src.diagnostics.paper_trade_logger import simulate_from_backtest, SIGNAL_LOG_COLUMNS
from src.diagnostics.execution_quality import compute_all_metrics
from src.utils.logging import get_logger

log = get_logger("family_canary")


# ── Lane definitions ────────────────────────────────────────────


LANE_TYPES = {
    "primary": ["SOL-USD", "SUI-USD"],
    "secondary_shadow": ["NEAR-USD"],
    "research_shadow": ["APT-USD"],
}


def _lane_type_for_asset(asset: str) -> str:
    for lane_type, assets in LANE_TYPES.items():
        if asset in assets:
            return lane_type
    return "research_shadow"


# ── 1. Family execution validation ─────────────────────────────


def family_execution_validation(
    *,
    panel: pd.DataFrame,
    predictions_by_asset: dict[str, pd.DataFrame],
    policy_cfg: dict,
    family_assets: list[str],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run execution validation per asset using simulated paper-trade logs.

    Parameters
    ----------
    panel : OHLC panel for all family assets.
    predictions_by_asset : dict mapping asset → predictions DataFrame
        with y_pred_prob, asset, timestamp columns.
    policy_cfg : policy config dict (threshold, cost_bps, sep_gap, etc.).
    family_assets : list of family asset names.

    Returns
    -------
    (execution_scorecard, per_asset_logs)
        execution_scorecard: DataFrame with one row per asset, all execution metrics.
        per_asset_logs: dict mapping asset → simulated paper-trade log DataFrame.
    """
    log.info("═══ Family execution validation ═══")

    entry_mode_cfg = {
        "name": "market_next_open",
        "type": "market",
        "offset_bps": 0.0,
    }

    scorecard_rows = []
    per_asset_logs = {}

    for asset in family_assets:
        asset_preds = predictions_by_asset.get(asset)
        if asset_preds is None or asset_preds.empty:
            log.warning("No predictions for %s — skipping", asset)
            continue

        asset_panel = panel[panel["asset"] == asset].copy()
        if asset_panel.empty:
            log.warning("No panel data for %s — skipping", asset)
            continue

        lane_type = _lane_type_for_asset(asset)

        trade_log = simulate_from_backtest(
            panel=asset_panel,
            predictions=asset_preds,
            entry_mode_cfg=entry_mode_cfg,
            policy_cfg=policy_cfg,
            lane_type=lane_type,
        )

        if trade_log.empty:
            log.warning("No trades simulated for %s", asset)
            continue

        per_asset_logs[asset] = trade_log

        metrics = compute_all_metrics(trade_log)
        metrics["asset"] = asset
        metrics["lane_type"] = lane_type
        scorecard_rows.append(metrics)

        log.info(
            "  %s (%s): %d signals, fill rate %.1f%%, net bps %.1f",
            asset, lane_type,
            metrics.get("submitted", 0),
            metrics.get("realized_fill_rate", 0) * 100,
            metrics.get("mean_return_per_trade_bps", np.nan),
        )

    scorecard = pd.DataFrame(scorecard_rows)
    return scorecard, per_asset_logs


# ── 2. Cross-asset fill profile ────────────────────────────────


def cross_asset_fill_profile(
    per_asset_logs: dict[str, pd.DataFrame],
    reference_asset: str = "SOL-USD",
) -> pd.DataFrame:
    """Compare fill-quality distributions across assets.

    Uses KS test between reference asset and each other asset for:
      - slippage (fill_price − simulated_fill_price) / simulated_fill_price
      - adverse selection (midprice_after_15m − fill_price) / fill_price
      - realized PnL at horizon

    Returns
    -------
    DataFrame with one row per (ref, other) pair, KS stat and p-value
    for each distribution.
    """
    log.info("═══ Cross-asset fill profile (ref=%s) ═══", reference_asset)

    ref_log = per_asset_logs.get(reference_asset)
    if ref_log is None or ref_log.empty:
        log.warning("Reference asset %s has no log data", reference_asset)
        return pd.DataFrame()

    ref_filled = ref_log[ref_log["cancel_status"] == "filled"].copy()
    if ref_filled.empty:
        return pd.DataFrame()

    rows = []
    for asset, asset_log in per_asset_logs.items():
        if asset == reference_asset:
            continue

        other_filled = asset_log[asset_log["cancel_status"] == "filled"].copy()
        if other_filled.empty:
            continue

        row = {
            "reference": reference_asset,
            "comparison": asset,
            "ref_fill_count": len(ref_filled),
            "other_fill_count": len(other_filled),
        }

        # Slippage comparison
        ref_slip = _compute_slippage(ref_filled)
        other_slip = _compute_slippage(other_filled)
        if len(ref_slip) > 1 and len(other_slip) > 1:
            ks_stat, ks_p = stats.ks_2samp(ref_slip, other_slip)
            row["slippage_ks_stat"] = ks_stat
            row["slippage_ks_pvalue"] = ks_p
            row["ref_mean_slippage_bps"] = ref_slip.mean()
            row["other_mean_slippage_bps"] = other_slip.mean()
        else:
            row["slippage_ks_stat"] = np.nan
            row["slippage_ks_pvalue"] = np.nan

        # Adverse selection comparison
        ref_adv = _compute_adverse_selection(ref_filled)
        other_adv = _compute_adverse_selection(other_filled)
        if len(ref_adv) > 1 and len(other_adv) > 1:
            ks_stat, ks_p = stats.ks_2samp(ref_adv, other_adv)
            row["adverse_ks_stat"] = ks_stat
            row["adverse_ks_pvalue"] = ks_p
            row["ref_mean_adverse_bps"] = ref_adv.mean()
            row["other_mean_adverse_bps"] = other_adv.mean()
        else:
            row["adverse_ks_stat"] = np.nan
            row["adverse_ks_pvalue"] = np.nan

        # Realized PnL comparison
        ref_pnl = ref_filled["realized_pnl_at_horizon"].dropna().values
        other_pnl = other_filled["realized_pnl_at_horizon"].dropna().values
        if len(ref_pnl) > 1 and len(other_pnl) > 1:
            ks_stat, ks_p = stats.ks_2samp(ref_pnl, other_pnl)
            row["pnl_ks_stat"] = ks_stat
            row["pnl_ks_pvalue"] = ks_p
        else:
            row["pnl_ks_stat"] = np.nan
            row["pnl_ks_pvalue"] = np.nan

        similar = all(
            row.get(f"{m}_ks_pvalue", 0) > 0.05
            for m in ["slippage", "adverse", "pnl"]
            if not np.isnan(row.get(f"{m}_ks_pvalue", np.nan))
        )
        row["profiles_similar"] = similar

        rows.append(row)

    df = pd.DataFrame(rows)
    log.info("Fill profile comparison:\n%s", df.to_string(index=False) if not df.empty else "empty")
    return df


def _compute_slippage(filled: pd.DataFrame) -> np.ndarray:
    """Slippage in bps = (fill_price − simulated_fill_price) / simulated_fill_price × 10000."""
    valid = filled.dropna(subset=["fill_price", "simulated_fill_price"])
    if valid.empty:
        return np.array([])
    return (
        (valid["fill_price"] - valid["simulated_fill_price"])
        / valid["simulated_fill_price"]
        * 10_000
    ).values


def _compute_adverse_selection(filled: pd.DataFrame) -> np.ndarray:
    """Adverse selection in bps = −(midprice_after_15m − fill_price) / fill_price × 10000."""
    valid = filled.dropna(subset=["fill_price", "midprice_after_15m"])
    if valid.empty:
        return np.array([])
    return (
        -(valid["midprice_after_15m"] - valid["fill_price"])
        / valid["fill_price"]
        * 10_000
    ).values


# ── 3. Family weekly PnL ───────────────────────────────────────


def family_weekly_pnl(
    per_asset_logs: dict[str, pd.DataFrame],
    family_assets: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Aggregate per-asset daily/weekly PnL for a combined family portfolio.

    Returns
    -------
    (weekly_pnl_df, portfolio_stats)
        weekly_pnl_df: DataFrame with week, per-asset PnL columns, total PnL.
        portfolio_stats: dict with portfolio-level Sharpe, max drawdown, etc.
    """
    log.info("═══ Family weekly PnL aggregation ═══")

    if family_assets is None:
        family_assets = list(per_asset_logs.keys())

    daily_pnls = {}

    for asset in family_assets:
        asset_log = per_asset_logs.get(asset)
        if asset_log is None or asset_log.empty:
            continue

        filled = asset_log[asset_log["cancel_status"] == "filled"].copy()
        if filled.empty:
            continue

        # Parse timestamps
        filled["signal_ts"] = pd.to_datetime(filled["signal_timestamp"])
        filled["date"] = filled["signal_ts"].dt.date

        # Daily PnL = sum of realized_pnl_at_horizon
        daily = filled.groupby("date")["realized_pnl_at_horizon"].sum()
        daily_pnls[asset] = daily

    if not daily_pnls:
        log.warning("No filled trades across any asset")
        return pd.DataFrame(), {}

    # Merge into single DataFrame
    daily_df = pd.DataFrame(daily_pnls).fillna(0)
    daily_df["total"] = daily_df.sum(axis=1)
    daily_df.index = pd.DatetimeIndex(daily_df.index)

    # Resample to weekly
    weekly_df = daily_df.resample("W").sum()
    weekly_df.index.name = "week"

    # Portfolio stats
    total_daily = daily_df["total"]
    cumulative = total_daily.cumsum()
    drawdown = cumulative - cumulative.cummax()

    total_weekly = weekly_df["total"]

    portfolio_stats = {
        "total_pnl": float(total_daily.sum()),
        "mean_weekly_pnl": float(total_weekly.mean()) if len(total_weekly) > 0 else 0,
        "weekly_pnl_std": float(total_weekly.std()) if len(total_weekly) > 1 else 0,
        "weekly_sharpe": (
            float(total_weekly.mean() / total_weekly.std())
            if len(total_weekly) > 1 and total_weekly.std() > 0
            else 0.0
        ),
        "max_drawdown": float(drawdown.min()) if len(drawdown) > 0 else 0,
        "n_weeks": len(total_weekly),
        "n_positive_weeks": int((total_weekly > 0).sum()),
        "per_asset_contribution": {
            asset: float(weekly_df[asset].sum()) if asset in weekly_df.columns else 0
            for asset in family_assets
        },
    }

    log.info("Portfolio stats: %s", portfolio_stats)
    return weekly_df.reset_index(), portfolio_stats
