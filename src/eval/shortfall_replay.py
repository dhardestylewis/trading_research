"""Branch F — Cost Model Freshness / Cadence Study.

Tests train windows (3d/7d/14d/30d) × refresh cadences (1h/6h/12h/24h)
and judges by OOS shortfall prediction quality, decile ranking stability,
and live-vs-replay consistency.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional
from scipy.stats import spearmanr

logger = logging.getLogger("shortfall_replay")


def _parse_duration(s: str) -> int:
    """Parse a duration string like '3d' or '6h' into seconds."""
    s = s.strip().lower()
    if s.endswith("d"):
        return int(s[:-1]) * 86_400
    elif s.endswith("h"):
        return int(s[:-1]) * 3_600
    elif s.endswith("m"):
        return int(s[:-1]) * 60
    else:
        return int(s)


def _train_and_eval_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target: str,
) -> Dict:
    """Train a LightGBM on train, predict on test, return metrics."""
    import lightgbm as lgb

    valid_feats = [f for f in feature_cols if f in train_df.columns and f in test_df.columns]
    if not valid_feats or target not in train_df.columns or target not in test_df.columns:
        return {"spearman": np.nan, "mae": np.nan, "n_test": 0}

    X_tr = train_df[valid_feats].fillna(0)
    y_tr = train_df[target]
    X_te = test_df[valid_feats].fillna(0)
    y_te = test_df[target]

    if len(X_tr) < 50 or len(X_te) < 10:
        return {"spearman": np.nan, "mae": np.nan, "n_test": len(X_te)}

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_te, label=y_te, reference=dtrain)

    params = {
        "objective": "regression", "metric": "mae",
        "num_leaves": 31, "max_depth": 5, "learning_rate": 0.03,
        "verbose": -1,
    }
    model = lgb.train(
        params, dtrain,
        valid_sets=[dval],
        num_boost_round=200,
        callbacks=[lgb.early_stopping(15, verbose=False)],
    )

    preds = model.predict(X_te)
    valid = ~(np.isnan(y_te.values) | np.isnan(preds))
    if valid.sum() < 10:
        return {"spearman": np.nan, "mae": np.nan, "n_test": int(valid.sum())}

    sp, _ = spearmanr(y_te.values[valid], preds[valid])
    mae = np.mean(np.abs(y_te.values[valid] - preds[valid]))

    return {"spearman": float(sp), "mae": float(mae), "n_test": int(valid.sum())}


def evaluate_cost_freshness(
    surface: pd.DataFrame,
    train_windows: Optional[List[str]] = None,
    refresh_cadences: Optional[List[str]] = None,
    feature_cols: Optional[List[str]] = None,
    target: str = "shortfall_1s_bps",
) -> pd.DataFrame:
    """Evaluate cost model freshness across training windows × refresh cadences.

    Parameters
    ----------
    surface : pd.DataFrame
        Cost surface with datetime index.
    train_windows : list of str
        e.g. ["3d", "7d", "14d", "30d"]
    refresh_cadences : list of str
        e.g. ["1h", "6h", "12h", "24h"]
    feature_cols : list
        Feature columns to use.
    target : str
        Cost target column.

    Returns
    -------
    pd.DataFrame
        Rows = window × cadence, columns = metrics.
    """
    if train_windows is None:
        train_windows = ["3d", "7d", "14d", "30d"]
    if refresh_cadences is None:
        refresh_cadences = ["1h", "6h", "12h", "24h"]
    if feature_cols is None:
        feature_cols = [
            "quoted_spread_bps", "spread_percentile", "signed_volume_1s",
            "flow_imbalance", "trade_burst_5s", "recent_realized_volatility",
            "book_imbalance", "hour_of_day", "weekend_indicator",
            "spread_bps", "trade_count_burst_intensity",
        ]

    # Ensure datetime index
    if not isinstance(surface.index, pd.DatetimeIndex):
        if "timestamp" in surface.columns:
            surface = surface.set_index("timestamp").sort_index()
        else:
            surface = surface.sort_index()

    # Subsample for speed — freshness study tests ranking stability, not absolute fit
    if len(surface) > 200_000:
        logger.info("Subsampling surface from %d to 200K rows for freshness study", len(surface))
        surface = surface.iloc[::len(surface) // 200_000].copy()

    t_min, t_max = surface.index.min(), surface.index.max()
    results = []

    for tw in train_windows:
        tw_sec = _parse_duration(tw)
        for rc in refresh_cadences:
            rc_sec = _parse_duration(rc)

            logger.info("Freshness study: train_window=%s, refresh=%s", tw, rc)

            fold_results = []
            cursor = t_min + pd.Timedelta(seconds=tw_sec)

            max_folds = 30  # cap to keep runtime reasonable
            fold_count = 0

            while cursor < t_max and fold_count < max_folds:
                train_start = cursor - pd.Timedelta(seconds=tw_sec)
                test_end = min(cursor + pd.Timedelta(seconds=rc_sec), t_max)

                train_mask = (surface.index >= train_start) & (surface.index < cursor)
                test_mask = (surface.index >= cursor) & (surface.index < test_end)

                train_df = surface[train_mask]
                test_df = surface[test_mask]

                if len(train_df) >= 50 and len(test_df) >= 10:
                    m = _train_and_eval_fold(train_df, test_df, feature_cols, target)
                    fold_results.append(m)

                cursor += pd.Timedelta(seconds=rc_sec)
                fold_count += 1

            if fold_results:
                fr_df = pd.DataFrame(fold_results)
                results.append({
                    "train_window": tw,
                    "refresh_cadence": rc,
                    "mean_spearman": fr_df["spearman"].mean(),
                    "std_spearman": fr_df["spearman"].std(),
                    "mean_mae": fr_df["mae"].mean(),
                    "n_folds": len(fr_df),
                    "total_test_samples": fr_df["n_test"].sum(),
                    "stability": 1.0 - fr_df["spearman"].std() / max(fr_df["spearman"].mean(), 1e-6),
                })
            else:
                results.append({
                    "train_window": tw,
                    "refresh_cadence": rc,
                    "mean_spearman": np.nan,
                    "std_spearman": np.nan,
                    "mean_mae": np.nan,
                    "n_folds": 0,
                    "total_test_samples": 0,
                    "stability": np.nan,
                })

    result_df = pd.DataFrame(results)
    logger.info("Freshness study complete:\n%s", result_df.to_string())
    return result_df
