"""Regression-based direction rule for exp018.

Replaces the four heuristic rules from exp017 with a rolling OLS
regression predicting spread change from lagged features.
Sign of prediction gives direction (+1 / -1).
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from src.utils.logging import get_logger

log = get_logger("regression_direction")


def _build_features(spread_vals: np.ndarray,
                    lags: list[int]) -> tuple[np.ndarray, np.ndarray, int]:
    """Build lagged feature matrix and target for spread regression.

    Features:
    - Lagged spread changes at specified lags
    - Current spread z-score (trailing 48-bar)
    - Trailing volatility ratio (recent vs baseline)

    Returns
    -------
    X : feature matrix (n_valid, n_features)
    y : target (1-bar-ahead spread change)
    start_idx : index in original array where valid data starts
    """
    max_lag = max(lags)
    n = len(spread_vals)

    if n < max_lag + 50:
        return np.array([]), np.array([]), 0

    # Spread changes
    spread_diff = np.diff(spread_vals, prepend=spread_vals[0])

    # Trailing z-score
    s = pd.Series(spread_vals)
    roll_mean = s.rolling(48, min_periods=24).mean().values
    roll_std = s.rolling(48, min_periods=24).std().values
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = np.where(roll_std > 1e-12,
                          (spread_vals - roll_mean) / roll_std, 0.0)

    # Trailing vol ratio
    recent_vol = pd.Series(np.abs(spread_diff)).rolling(12, min_periods=6).mean().values
    baseline_vol = pd.Series(np.abs(spread_diff)).rolling(72, min_periods=36).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_ratio = np.where(baseline_vol > 1e-12,
                             recent_vol / baseline_vol, 1.0)

    # Build matrix
    start_idx = max_lag + 1
    n_valid = n - start_idx - 1  # -1 for the forward target

    if n_valid < 10:
        return np.array([]), np.array([]), 0

    n_features = len(lags) + 2  # lagged changes + zscore + vol_ratio
    X = np.zeros((n_valid, n_features))
    y = np.zeros(n_valid)

    for row in range(n_valid):
        idx = start_idx + row
        # Lagged spread changes
        for j, lag in enumerate(lags):
            X[row, j] = spread_diff[idx - lag]
        # Z-score
        X[row, len(lags)] = zscore[idx]
        # Vol ratio
        X[row, len(lags) + 1] = vol_ratio[idx]
        # Target: next-bar spread change
        y[row] = spread_diff[idx + 1]

    return X, y, start_idx


def _rolling_ols_predict(X: np.ndarray, y: np.ndarray,
                         lookback: int, min_obs: int) -> np.ndarray:
    """Rolling OLS with ridge regularization.

    For each point t, fit on [t-lookback:t], predict at t.

    Returns
    -------
    predictions : array of predicted values (NaN where insufficient data)
    """
    n = len(y)
    predictions = np.full(n, np.nan)
    ridge_lambda = 1e-4

    for t in range(lookback, n):
        X_train = X[t - lookback:t]
        y_train = y[t - lookback:t]

        if len(y_train) < min_obs:
            continue

        # Ridge regression: (X'X + λI)^{-1} X'y
        XtX = X_train.T @ X_train + ridge_lambda * np.eye(X_train.shape[1])
        Xty = X_train.T @ y_train

        try:
            beta = np.linalg.solve(XtX, Xty)
            predictions[t] = X[t] @ beta
        except np.linalg.LinAlgError:
            continue

    return predictions


def generate_regression_signals(spread_df: pd.DataFrame,
                                spread_types: list[str],
                                regression_cfg: dict,
                                event_mask: np.ndarray | None = None
                                ) -> pd.DataFrame:
    """Generate direction signals from rolling spread regression.

    Parameters
    ----------
    spread_df : DataFrame for a single pair with spread_* columns.
    spread_types : which spread columns to apply regression to.
    regression_cfg : config dict with lookback, min_obs, lags, threshold.
    event_mask : optional boolean mask (only signal where True).

    Returns
    -------
    DataFrame of signals matching exp017 format:
        timestamp, pair, rule_name, spread_type, direction, entry_spread_value
    """
    lookback = regression_cfg["lookback"]
    min_obs = regression_cfg["min_obs"]
    lags = regression_cfg["lags"]
    threshold = regression_cfg["prediction_threshold"]

    signal_rows = []

    for spread_type in spread_types:
        col = f"spread_{spread_type}"
        if col not in spread_df.columns:
            continue

        spread_vals = spread_df[col].values
        X, y, start_idx = _build_features(spread_vals, lags)

        if len(X) == 0:
            continue

        predictions = _rolling_ols_predict(X, y, lookback, min_obs)

        # Convert predictions to signals
        for i in range(len(predictions)):
            if np.isnan(predictions[i]):
                continue

            abs_pred = abs(predictions[i])
            if abs_pred <= threshold:
                continue

            original_idx = start_idx + i
            if original_idx >= len(spread_df):
                continue

            # Apply event mask
            if event_mask is not None and not event_mask[original_idx]:
                continue

            direction = 1 if predictions[i] > 0 else -1

            signal_rows.append({
                "timestamp": spread_df.iloc[original_idx]["timestamp"],
                "pair": spread_df.iloc[original_idx]["pair"],
                "rule_name": "spread_regression",
                "spread_type": spread_type,
                "direction": direction,
                "entry_spread_value": float(spread_vals[original_idx]),
                "entry_idx": int(original_idx),
                "close_base": float(spread_df.iloc[original_idx]["close_base"]),
                "close_quote": float(spread_df.iloc[original_idx]["close_quote"]),
                "prediction": float(predictions[i]),
            })

    result = pd.DataFrame(signal_rows)
    if not result.empty:
        pair = spread_df["pair"].iloc[0]
        n_long = (result["direction"] == 1).sum()
        n_short = (result["direction"] == -1).sum()
        log.info(f"Regression signals for {pair}: {len(result)} total "
                 f"({n_long} long, {n_short} short)")
    return result
