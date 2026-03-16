"""Direction rules for exp017 — generate signed entry signals.

Each rule takes a spread series and returns +1 (long spread), -1 (short spread),
or 0 (no trade). Direction is locked ex ante from a mechanical rule,
NOT from ex-post excursion.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from src.utils.logging import get_logger

log = get_logger("direction_rules")


def _breakout_signals(spread: np.ndarray, window: int) -> np.ndarray:
    """Continuation: spread breaks above/below rolling range → enter in direction."""
    n = len(spread)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(window, n):
        rolling_high = np.nanmax(spread[i - window:i])
        rolling_low = np.nanmin(spread[i - window:i])
        if spread[i] > rolling_high:
            signals[i] = 1   # long spread (continuation up)
        elif spread[i] < rolling_low:
            signals[i] = -1  # short spread (continuation down)
    return signals


def _mean_reversion_signals(zscore: np.ndarray, threshold: float) -> np.ndarray:
    """Reversal: fade extreme z-score dislocations."""
    signals = np.zeros(len(zscore), dtype=np.int8)
    signals[zscore > threshold] = -1   # spread overbought → short spread
    signals[zscore < -threshold] = 1   # spread oversold → long spread
    return signals


def _vol_breakout_signals(spread: np.ndarray, compression_window: int,
                          expansion_mult: float) -> np.ndarray:
    """Volatility: ATR compression then expansion with signed filter."""
    n = len(spread)
    signals = np.zeros(n, dtype=np.int8)

    # ATR of the spread itself
    spread_diff = np.abs(np.diff(spread, prepend=spread[0]))
    # Rolling ATR
    atr = pd.Series(spread_diff).rolling(compression_window, min_periods=1).mean().values
    baseline_atr = pd.Series(atr).rolling(compression_window * 3, min_periods=1).mean().values

    for i in range(compression_window * 3, n):
        # Was compressed (recent ATR below baseline) and now expanding
        if atr[i] > expansion_mult * baseline_atr[i]:
            # Direction from the sign of the move
            move = spread[i] - spread[i - 1]
            if move > 0:
                signals[i] = 1
            elif move < 0:
                signals[i] = -1
    return signals


def _momentum_signals(spread: np.ndarray, fast_window: int,
                      slow_window: int) -> np.ndarray:
    """Momentum: fast MA vs slow MA crossover."""
    s = pd.Series(spread)
    fast_ma = s.rolling(fast_window, min_periods=fast_window).mean().values
    slow_ma = s.rolling(slow_window, min_periods=slow_window).mean().values

    signals = np.zeros(len(spread), dtype=np.int8)
    for i in range(slow_window, len(spread)):
        if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]):
            continue
        if fast_ma[i] > slow_ma[i]:
            signals[i] = 1
        elif fast_ma[i] < slow_ma[i]:
            signals[i] = -1
    return signals


def generate_direction_signals(spread_df: pd.DataFrame,
                               rules_cfg: list[dict],
                               spread_types: list[str],
                               event_mask: np.ndarray | None = None
                               ) -> pd.DataFrame:
    """Generate signed direction signals for each (pair, spread_type, rule).

    Parameters
    ----------
    spread_df : DataFrame for a single pair with spread_* columns and timestamp.
    rules_cfg : list of rule config dicts from YAML.
    spread_types : which spread columns to apply rules to.
    event_mask : optional boolean array, if provided only emit signals where True.

    Returns
    -------
    DataFrame of trade signals with columns:
        timestamp, pair, rule_name, spread_type, direction, entry_spread_value
    """
    signal_rows = []

    for spread_type in spread_types:
        col = f"spread_{spread_type}"
        if col not in spread_df.columns:
            continue
        spread_vals = spread_df[col].values

        for rule in rules_cfg:
            rule_name = rule["name"]
            rule_type = rule["type"]

            if rule_type == "breakout":
                raw_signals = _breakout_signals(spread_vals, rule["window"])
            elif rule_type == "mean_reversion":
                if spread_type != "zscore":
                    # Mean-reversion rule only applies to z-score spread
                    continue
                raw_signals = _mean_reversion_signals(
                    spread_vals, rule["zscore_threshold"])
            elif rule_type == "vol_breakout":
                raw_signals = _vol_breakout_signals(
                    spread_vals, rule["compression_window"],
                    rule["expansion_multiplier"])
            elif rule_type == "momentum":
                raw_signals = _momentum_signals(
                    spread_vals, rule["fast_window"], rule["slow_window"])
            else:
                log.warning(f"Unknown rule type: {rule_type}")
                continue

            # Apply event mask if provided
            if event_mask is not None:
                raw_signals = raw_signals * event_mask.astype(np.int8)

            # Extract non-zero signals
            active = np.where(raw_signals != 0)[0]
            for idx in active:
                signal_rows.append({
                    "timestamp": spread_df.iloc[idx]["timestamp"],
                    "pair": spread_df.iloc[idx]["pair"],
                    "rule_name": rule_name,
                    "spread_type": spread_type,
                    "direction": int(raw_signals[idx]),
                    "entry_spread_value": float(spread_vals[idx]),
                    "entry_idx": int(idx),
                    "close_base": float(spread_df.iloc[idx]["close_base"]),
                    "close_quote": float(spread_df.iloc[idx]["close_quote"]),
                })

    result = pd.DataFrame(signal_rows)
    if not result.empty:
        log.info(f"Generated {len(result)} signals for pair={spread_df['pair'].iloc[0]}")
    return result
