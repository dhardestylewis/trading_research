"""Spread definitions for RV pair analysis.

Computes 4 spread representations from a ratio panel:
1. raw_ratio   — P_A / P_B
2. log_spread  — log(P_A) - log(P_B)
3. beta_adjusted — P_A - β·P_B via rolling OLS
4. zscore      — (spread - μ) / σ over trailing window
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from src.utils.logging import get_logger

log = get_logger("spread_definitions")


def _rolling_beta(base_prices: np.ndarray, quote_prices: np.ndarray,
                  lookback: int) -> np.ndarray:
    """Compute rolling OLS beta: base_ret ~ β * quote_ret."""
    n = len(base_prices)
    betas = np.full(n, np.nan)

    base_ret = np.diff(np.log(base_prices))
    quote_ret = np.diff(np.log(quote_prices))

    for i in range(lookback, len(base_ret)):
        y = base_ret[i - lookback:i]
        x = quote_ret[i - lookback:i]
        x_mean = x.mean()
        y_mean = y.mean()
        cov_xy = np.sum((x - x_mean) * (y - y_mean))
        var_x = np.sum((x - x_mean) ** 2)
        if var_x > 1e-12:
            betas[i + 1] = cov_xy / var_x
        else:
            betas[i + 1] = 1.0

    return betas


def compute_spreads(panel: pd.DataFrame, pairs: list[list[str]],
                    spread_types: list[str],
                    beta_lookback: int = 72,
                    zscore_lookback: int = 72) -> pd.DataFrame:
    """Compute spread definitions for each RV pair.

    Parameters
    ----------
    panel : OHLCV panel with 'asset' and 'timestamp' columns.
    pairs : list of [base, quote] symbol pairs.
    spread_types : which spread definitions to compute.
    beta_lookback : rolling window for OLS beta.
    zscore_lookback : rolling window for z-score.

    Returns
    -------
    DataFrame with columns: timestamp, pair, close_base, close_quote,
    and one column per spread type.
    """
    log.info(f"Computing spreads for {len(pairs)} pairs, types={spread_types}")

    # Pivot panel to get per-asset close prices
    col_key = "asset" if "asset" in panel.columns else "symbol"
    pvt = panel.pivot(index="timestamp", columns=col_key)

    all_dfs = []
    for pair in pairs:
        base, quote = pair
        base_label = (f"{base}-USD" if f"{base}-USD" in pvt["close"].columns
                      else f"{base}-USDT")
        quote_label = (f"{quote}-USD" if f"{quote}-USD" in pvt["close"].columns
                       else f"{quote}-USDT")

        if base_label not in pvt["close"].columns or quote_label not in pvt["close"].columns:
            log.warning(f"Missing {base_label} or {quote_label} in panel")
            continue

        base_close = pvt["close"][base_label].values
        quote_close = pvt["close"][quote_label].values
        timestamps = pvt.index.values

        # Build result frame
        df = pd.DataFrame({"timestamp": timestamps,
                           "pair": f"{base}/{quote}",
                           "close_base": base_close,
                           "close_quote": quote_close})

        # Also grab OHLC for the base for ATR computation
        for col in ["open", "high", "low"]:
            if col in pvt.columns:
                df[f"{col}_base"] = pvt[col][base_label].values
                df[f"{col}_quote"] = pvt[col][quote_label].values

        # --- Spread definitions ---
        if "raw_ratio" in spread_types:
            df["spread_raw_ratio"] = base_close / quote_close

        if "log_spread" in spread_types:
            with np.errstate(divide="ignore", invalid="ignore"):
                df["spread_log_spread"] = np.log(base_close) - np.log(quote_close)

        if "beta_adjusted" in spread_types:
            betas = _rolling_beta(base_close, quote_close, beta_lookback)
            df["rolling_beta"] = betas
            df["spread_beta_adjusted"] = base_close - betas * quote_close

        if "zscore" in spread_types:
            # Z-score on log spread
            log_spread = np.log(base_close) - np.log(quote_close)
            s = pd.Series(log_spread)
            rolling_mean = s.rolling(zscore_lookback, min_periods=zscore_lookback).mean()
            rolling_std = s.rolling(zscore_lookback, min_periods=zscore_lookback).std()
            df["spread_zscore"] = ((s - rolling_mean) / rolling_std).values

        df = df.dropna(subset=[c for c in df.columns if c.startswith("spread_")],
                       how="all")
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    log.info(f"Computed {len(result)} rows across {result['pair'].nunique()} pairs")
    return result
