"""Spread PnL aggregation and side-by-side comparison for exp017.

Aggregates trade-level results into summary statistics and produces
the critical side-by-side table: "Max Excursion Opportunity" vs
"Realized Rule-Locked Net PnL" per pair.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from src.utils.logging import get_logger

log = get_logger("spread_pnl")


def build_trade_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trade-level results into summary per (pair, rule, spread_type, horizon).

    Returns
    -------
    DataFrame with: pair, rule, spread_type, horizon_h, trade_count, hit_rate,
    median_net_bps, mean_net_bps, mean_gross_bps, weekly_trade_count,
    max_drawdown_bps, sharpe_of_spread
    """
    if trades_df.empty:
        return pd.DataFrame()

    group_cols = ["pair", "rule", "spread_type", "horizon_h"]
    summary_rows = []

    for keys, group in trades_df.groupby(group_cols):
        pair, rule, spread_type, horizon_h = keys
        n = len(group)
        net_bps = group["net_spread_bps"].values
        gross_bps = group["gross_spread_bps"].values

        hits = np.sum(net_bps > 0)
        hit_rate = hits / n if n > 0 else 0.0

        # Estimate weekly trade count from time range
        if n > 1:
            ts = pd.to_datetime(group["entry_time"])
            time_span_days = (ts.max() - ts.min()).total_seconds() / 86400
            weeks = max(time_span_days / 7, 1)
            weekly_count = n / weeks
        else:
            weekly_count = 0.0

        # Running drawdown of cumulative net PnL
        cum_pnl = np.cumsum(net_bps)
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns = cum_pnl - running_max
        max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Sharpe-like ratio: mean / std of net bps
        mean_net = float(np.mean(net_bps))
        std_net = float(np.std(net_bps)) if n > 1 else np.nan
        sharpe = mean_net / std_net if (std_net and std_net > 1e-6) else np.nan

        summary_rows.append({
            "pair": pair,
            "rule": rule,
            "spread_type": spread_type,
            "horizon_h": horizon_h,
            "trade_count": n,
            "hit_rate": round(hit_rate, 3),
            "median_net_bps": round(float(np.median(net_bps)), 2),
            "mean_net_bps": round(mean_net, 2),
            "mean_gross_bps": round(float(np.mean(gross_bps)), 2),
            "weekly_trade_count": round(weekly_count, 1),
            "max_drawdown_bps": round(max_dd, 2),
            "sharpe_of_spread": round(sharpe, 3) if not np.isnan(sharpe) else None,
        })

    result = pd.DataFrame(summary_rows)
    if not result.empty:
        result = result.sort_values("median_net_bps", ascending=False)
    return result


def build_excursion_comparison(trades_df: pd.DataFrame,
                               spread_df: pd.DataFrame,
                               horizons: list[int]) -> pd.DataFrame:
    """Build side-by-side table: max excursion opportunity vs realized PnL.

    For each (pair, horizon), compute:
    - Max excursion opportunity: median of max(|upside|, |downside|) excursions
      (what exp016 measured — the best possible move)
    - Realized rule-locked net PnL: median of actual net_spread_bps from trades

    Parameters
    ----------
    trades_df : completed trades from trade_simulator.
    spread_df : spread panel with spread columns.
    horizons : list of horizon hours.

    Returns
    -------
    DataFrame with: pair, horizon_h, median_max_excursion_bps,
    median_realized_net_bps, excursion_capture_ratio
    """
    if trades_df.empty or spread_df.empty:
        return pd.DataFrame()

    comparison_rows = []

    for pair in trades_df["pair"].unique():
        pair_spread = spread_df[spread_df["pair"] == pair].copy()
        pair_spread = pair_spread.sort_values("timestamp").reset_index(drop=True)

        # Use raw_ratio spread for excursion calculation (comparable to exp016)
        if "spread_raw_ratio" not in pair_spread.columns:
            continue
        ratio_vals = pair_spread["spread_raw_ratio"].values

        for horizon in horizons:
            # Compute max excursion (exp016 methodology)
            n = len(ratio_vals)
            max_excursions = []
            for i in range(n - horizon):
                window = ratio_vals[i + 1: i + 1 + horizon]
                if len(window) == 0:
                    continue
                entry = ratio_vals[i]
                if abs(entry) < 1e-12:
                    continue
                up_bps = (np.max(window) / entry - 1.0) * 10000
                down_bps = abs((np.min(window) / entry - 1.0) * 10000)
                max_excursions.append(max(up_bps, down_bps))

            median_excursion = (float(np.median(max_excursions))
                                if max_excursions else np.nan)

            # Get realized PnL for this pair and horizon
            pair_trades = trades_df[
                (trades_df["pair"] == pair) &
                (trades_df["horizon_h"] == horizon)
            ]

            if pair_trades.empty:
                median_realized = np.nan
            else:
                median_realized = float(pair_trades["net_spread_bps"].median())

            # Capture ratio: what fraction of the opportunity is realized
            if not np.isnan(median_excursion) and median_excursion > 0:
                capture = median_realized / median_excursion if not np.isnan(median_realized) else np.nan
            else:
                capture = np.nan

            comparison_rows.append({
                "pair": pair,
                "horizon_h": horizon,
                "median_max_excursion_bps": round(median_excursion, 1)
                    if not np.isnan(median_excursion) else None,
                "median_realized_net_bps": round(median_realized, 1)
                    if not np.isnan(median_realized) else None,
                "excursion_capture_ratio": round(capture, 3)
                    if not np.isnan(capture) else None,
            })

    return pd.DataFrame(comparison_rows)


def evaluate_kill_gates(summary_df: pd.DataFrame,
                        gates_cfg: dict) -> dict:
    """Evaluate exp017 kill gates.

    Returns dict with pass/fail per criterion and overall verdict.
    """
    results = {}

    # Gate 1: at least min_pairs pairs with enough trades
    pairs_with_enough = 0
    if not summary_df.empty:
        pair_trade_counts = summary_df.groupby("pair")["trade_count"].sum()
        pairs_with_enough = int((pair_trade_counts >= gates_cfg["min_trades_per_pair"]).sum())

    results["pairs_with_enough_trades"] = {
        "value": pairs_with_enough,
        "threshold": gates_cfg["min_pairs"],
        "pass": pairs_with_enough >= gates_cfg["min_pairs"],
    }

    # Gate 2: at least min_trades_per_pair per qualifying pair
    results["min_trades_per_pair"] = {
        "value": int(pair_trade_counts.min()) if not summary_df.empty and len(pair_trade_counts) > 0 else 0,
        "threshold": gates_cfg["min_trades_per_pair"],
        "pass": pairs_with_enough >= gates_cfg["min_pairs"],
    }

    # Gate 3: median net spread > threshold (best combo per pair)
    best_medians = []
    if not summary_df.empty:
        for pair in summary_df["pair"].unique():
            pair_df = summary_df[summary_df["pair"] == pair]
            if not pair_df.empty:
                best_medians.append(pair_df["median_net_bps"].max())

    best_median = float(np.median(best_medians)) if best_medians else 0.0
    results["median_net_spread_bps"] = {
        "value": round(best_median, 2),
        "threshold": gates_cfg["min_median_net_spread_bps"],
        "pass": best_median >= gates_cfg["min_median_net_spread_bps"],
    }

    # Gate 4: hit rate
    best_hit_rates = []
    if not summary_df.empty:
        for pair in summary_df["pair"].unique():
            pair_df = summary_df[summary_df["pair"] == pair]
            if not pair_df.empty:
                best_hit_rates.append(pair_df["hit_rate"].max())

    median_hit = float(np.median(best_hit_rates)) if best_hit_rates else 0.0
    results["hit_rate"] = {
        "value": round(median_hit, 3),
        "threshold": gates_cfg["min_hit_rate"],
        "pass": median_hit >= gates_cfg["min_hit_rate"],
    }

    # Overall
    results["overall_pass"] = all(r["pass"] for k, r in results.items()
                                  if isinstance(r, dict) and "pass" in r)

    return results
