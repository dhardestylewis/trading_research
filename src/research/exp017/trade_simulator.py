"""Two-leg trade simulator for exp017.

Simulates the full lifecycle of each spread trade with:
- Entry at signal bar close
- Bar-by-bar tracking for stop/TP/horizon exit
- Per-leg fee and slippage
- Legging risk (delay between leg fills)
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from src.utils.logging import get_logger

log = get_logger("trade_simulator")


def _compute_spread_path(spread_df: pd.DataFrame, spread_col: str,
                         entry_idx: int, max_bars: int) -> np.ndarray:
    """Extract the spread path from entry_idx for max_bars forward."""
    end_idx = min(entry_idx + max_bars + 1, len(spread_df))
    return spread_df[spread_col].values[entry_idx:end_idx]


def simulate_trades(spread_df: pd.DataFrame,
                    signals_df: pd.DataFrame,
                    exec_cfg: dict) -> pd.DataFrame:
    """Simulate spread trades with two-leg execution model.

    Parameters
    ----------
    spread_df : DataFrame for a single pair with spread columns, timestamps,
                close_base, close_quote.
    signals_df : DataFrame of signals for this pair with entry_idx, direction, etc.
    exec_cfg : execution config dict from YAML.

    Returns
    -------
    DataFrame with one row per completed trade.
    """
    if signals_df.empty:
        return pd.DataFrame()

    stop_bps = exec_cfg["stop_loss_bps"]
    tp_bps = exec_cfg["take_profit_bps"]
    rt_cost_bps = exec_cfg["total_round_trip_bps"]
    fee_per_leg = exec_cfg["fee_bps_per_leg"]
    slip_per_leg = exec_cfg["slippage_bps_per_leg"]
    max_delay = exec_cfg["max_leg_delay_bars"]
    leg_slip_mult = exec_cfg["leg_slip_multiplier"]
    horizons = exec_cfg["holding_horizons_hours"]

    trade_rows = []

    for horizon in horizons:
        for _, sig in signals_df.iterrows():
            entry_idx = int(sig["entry_idx"])
            direction = int(sig["direction"])
            spread_type = sig["spread_type"]
            spread_col = f"spread_{spread_type}"

            if spread_col not in spread_df.columns:
                continue

            # Get forward spread path
            path = _compute_spread_path(spread_df, spread_col,
                                        entry_idx, horizon)
            if len(path) < 2:
                continue

            entry_val = path[0]
            if abs(entry_val) < 1e-12:
                continue

            # Track bar-by-bar for stop/TP
            exit_reason = "horizon"
            exit_bar = len(path) - 1
            exit_val = path[-1]

            for bar in range(1, len(path)):
                move_bps = (path[bar] / entry_val - 1.0) * direction * 10000.0

                if move_bps <= -stop_bps:
                    exit_reason = "stop_loss"
                    exit_bar = bar
                    exit_val = path[bar]
                    break
                elif move_bps >= tp_bps:
                    exit_reason = "take_profit"
                    exit_bar = bar
                    exit_val = path[bar]
                    break

            # Gross spread return
            gross_bps = (exit_val / entry_val - 1.0) * direction * 10000.0

            # Legging risk: model probabilistic extra slippage
            # Simple model: 30% chance of 1-2 bar delay on second leg
            rng_val = hash((sig["timestamp"], sig["rule_name"],
                            spread_type, horizon)) % 100
            if rng_val < 30:
                leg_penalty = leg_slip_mult * slip_per_leg
            else:
                leg_penalty = 0.0

            # Total friction: base round-trip + any legging penalty
            total_friction_bps = rt_cost_bps + leg_penalty

            # Net spread return
            net_bps = gross_bps - total_friction_bps

            # Per-leg slippage detail
            slippage_leg1 = slip_per_leg
            slippage_leg2 = slip_per_leg + leg_penalty

            # Get actual timestamp for exit
            exit_ts_idx = min(entry_idx + exit_bar, len(spread_df) - 1)

            trade_rows.append({
                "entry_time": sig["timestamp"],
                "exit_time": spread_df.iloc[exit_ts_idx]["timestamp"],
                "pair": sig["pair"],
                "rule": sig["rule_name"],
                "spread_type": spread_type,
                "horizon_h": horizon,
                "direction": direction,
                "direction_label": "long_spread" if direction == 1 else "short_spread",
                "entry_spread": entry_val,
                "exit_spread": exit_val,
                "gross_spread_bps": round(gross_bps, 2),
                "fee_per_leg_bps": fee_per_leg,
                "slippage_leg1_bps": round(slippage_leg1, 2),
                "slippage_leg2_bps": round(slippage_leg2, 2),
                "total_friction_bps": round(total_friction_bps, 2),
                "net_spread_bps": round(net_bps, 2),
                "holding_bars": exit_bar,
                "exit_reason": exit_reason,
                "close_base_entry": sig["close_base"],
                "close_quote_entry": sig["close_quote"],
            })

    result = pd.DataFrame(trade_rows)
    if not result.empty:
        pair_name = signals_df["pair"].iloc[0]
        log.info(f"Simulated {len(result)} trades for {pair_name}")
    return result
