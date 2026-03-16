"""Live canary health-check monitor.

Evaluates realized paper-trade execution metrics against tolerance bands
defined in the exp007 config.  Provides:

  1. Per-lane health checks against thresholds
  2. Realized vs simulated lane-by-lane comparison
  3. Weekly execution-error stability scoring

All functions consume DataFrames in the paper-trade log schema produced
by paper_trade_logger.py — same schema for simulated and live data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.diagnostics.execution_quality import compute_all_metrics
from src.utils.logging import get_logger

log = get_logger("live_canary_monitor")


# ═══════════════════════════════════════════════════════════════════
#  Health check result
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BandCheck:
    """Result of checking one metric against its tolerance band."""
    metric: str
    value: float
    threshold: float
    direction: Literal["min", "max"]
    passed: bool
    lane: str = ""

    @property
    def status_icon(self) -> str:
        return "✅" if self.passed else "❌"

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "direction": self.direction,
            "passed": self.passed,
            "lane": self.lane,
        }


# ═══════════════════════════════════════════════════════════════════
#  Canary health checker
# ═══════════════════════════════════════════════════════════════════

class CanaryHealthCheck:
    """Checks execution metrics against tolerance bands from config."""

    # Mapping: config key → (metric key in compute_all_metrics, direction)
    BAND_MAP = {
        "min_realized_fill_rate":           ("realized_fill_rate",           "min"),
        "max_mean_slippage_bps":            ("mean_slippage_bps",           "max"),
        "max_cancel_rate":                  ("cancel_rate",                 "max"),
        "max_shortfall_vs_simulated_bps":   ("shortfall_vs_simulated_bps", "max"),
        "min_return_per_trade_bps":         ("mean_return_per_trade_bps",   "min"),
    }

    def __init__(self, tolerance_bands: dict) -> None:
        self.bands = tolerance_bands

    def check_lane(self, log_df: pd.DataFrame, lane_name: str = "") -> list[BandCheck]:
        """Run all tolerance checks on one lane's log data."""
        metrics = compute_all_metrics(log_df)
        checks: list[BandCheck] = []

        for config_key, (metric_key, direction) in self.BAND_MAP.items():
            threshold = self.bands.get(config_key)
            if threshold is None:
                continue

            value = metrics.get(metric_key, np.nan)
            if not np.isfinite(value):
                passed = False
            elif direction == "min":
                passed = value >= threshold
            else:
                # For absolute-value metrics like slippage, check abs
                passed = abs(value) <= threshold

            checks.append(BandCheck(
                metric=metric_key,
                value=value,
                threshold=threshold,
                direction=direction,
                passed=passed,
                lane=lane_name,
            ))

        return checks

    def check_all_lanes(
        self,
        log_df: pd.DataFrame,
        lane_col: str = "entry_mode",
    ) -> dict[str, list[BandCheck]]:
        """Run health checks per lane."""
        results: dict[str, list[BandCheck]] = {}

        if lane_col not in log_df.columns:
            results["all"] = self.check_lane(log_df, "all")
            return results

        for lane_name, lane_df in log_df.groupby(lane_col):
            results[str(lane_name)] = self.check_lane(lane_df, str(lane_name))

        return results

    def all_passed(self, checks: dict[str, list[BandCheck]]) -> bool:
        """True if every single check across all lanes passed."""
        return all(
            bc.passed
            for lane_checks in checks.values()
            for bc in lane_checks
        )

    def summary_table(self, checks: dict[str, list[BandCheck]]) -> pd.DataFrame:
        """Convert health checks to a DataFrame for reporting."""
        rows = []
        for lane_checks in checks.values():
            for bc in lane_checks:
                rows.append(bc.to_dict())
        return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
#  Realized vs simulated comparison
# ═══════════════════════════════════════════════════════════════════

def compare_realized_vs_simulated(
    simulated_log: pd.DataFrame,
    realized_log: pd.DataFrame,
    lane_col: str = "entry_mode",
) -> pd.DataFrame:
    """Compare lane-by-lane metrics between simulated and realized logs.

    Returns a DataFrame with columns:
        lane, metric, simulated_value, realized_value, delta, pct_change,
        degradation_flag (True if realized is structurally worse)
    """
    def _lane_metrics(log_df: pd.DataFrame) -> dict[str, dict]:
        result = {}
        if lane_col in log_df.columns:
            for lane, df in log_df.groupby(lane_col):
                result[str(lane)] = compute_all_metrics(df)
        else:
            result["all"] = compute_all_metrics(log_df)
        return result

    sim_metrics = _lane_metrics(simulated_log)
    real_metrics = _lane_metrics(realized_log)

    # Metrics where higher is better (min-direction thresholds)
    HIGHER_IS_BETTER = {"realized_fill_rate", "mean_return_per_trade_bps", "sharpe", "hit_rate"}

    rows = []
    all_lanes = set(sim_metrics.keys()) | set(real_metrics.keys())

    for lane in sorted(all_lanes):
        sim = sim_metrics.get(lane, {})
        real = real_metrics.get(lane, {})
        all_keys = set(sim.keys()) | set(real.keys())

        # Skip internal keys
        skip_keys = {"submitted", "filled", "missed_count"}

        for metric in sorted(all_keys - skip_keys):
            sim_val = sim.get(metric, np.nan)
            real_val = real.get(metric, np.nan)

            if not isinstance(sim_val, (int, float)) or not isinstance(real_val, (int, float)):
                continue

            delta = real_val - sim_val if np.isfinite(real_val) and np.isfinite(sim_val) else np.nan
            pct = (delta / abs(sim_val) * 100) if np.isfinite(delta) and abs(sim_val) > 1e-10 else np.nan

            # Flag degradation
            if metric in HIGHER_IS_BETTER:
                degraded = delta < 0 if np.isfinite(delta) else False
            else:
                degraded = delta > 0 if np.isfinite(delta) else False

            rows.append({
                "lane": lane,
                "metric": metric,
                "simulated": sim_val,
                "realized": real_val,
                "delta": delta,
                "pct_change": pct,
                "degradation_flag": degraded,
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
#  Weekly execution-error stability
# ═══════════════════════════════════════════════════════════════════

def weekly_execution_error(
    log_df: pd.DataFrame,
    timestamp_col: str = "signal_timestamp",
) -> pd.DataFrame:
    """Compute rolling weekly execution-error stability scores.

    Returns a DataFrame with one row per ISO week:
        week, mean_shortfall_bps, std_shortfall_bps, n_trades,
        execution_error_bps (= std of daily shortfall within the week)
    """
    df = log_df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    filled = df[df["cancel_status"].isin(["filled", "partial_fill"])].copy()

    if filled.empty or "realized_shortfall_vs_simulated" not in filled.columns:
        return pd.DataFrame(columns=[
            "week", "mean_shortfall_bps", "std_shortfall_bps",
            "n_trades", "execution_error_bps",
        ])

    filled["date"] = filled[timestamp_col].dt.date
    filled["week"] = filled[timestamp_col].dt.isocalendar().week.astype(int)

    # Daily shortfall
    daily = filled.groupby("date")["realized_shortfall_vs_simulated"].agg(
        ["mean", "count"]
    ).reset_index()
    daily.columns = ["date", "daily_mean_shortfall", "n_trades"]
    daily["week"] = pd.to_datetime(daily["date"]).dt.isocalendar().week.astype(int)

    # Weekly aggregation
    weekly = daily.groupby("week").agg(
        mean_shortfall_bps=("daily_mean_shortfall", "mean"),
        std_shortfall_bps=("daily_mean_shortfall", "std"),
        n_trades=("n_trades", "sum"),
    ).reset_index()
    weekly["execution_error_bps"] = weekly["std_shortfall_bps"]

    return weekly
