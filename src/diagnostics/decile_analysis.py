"""Score-decile and tail-quantile analysis.

For each model, partition predictions into score buckets and compute
per-bucket trading metrics to assess score ordering and tail tradability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.metrics import trading_metrics
from src.utils.logging import get_logger

log = get_logger("decile_analysis")


def assign_deciles(
    preds: pd.DataFrame,
    prob_col: str = "y_pred_prob",
    n_buckets: int = 10,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Add ``score_decile`` column (1 = lowest, N = highest).

    If *group_cols* is provided, quantile cuts are computed within each
    group (e.g. per fold) to avoid look-ahead bias.
    """
    df = preds.copy()
    if group_cols:
        df["score_decile"] = df.groupby(group_cols)[prob_col].transform(
            lambda s: pd.qcut(s, n_buckets, labels=False, duplicates="drop") + 1
        )
    else:
        df["score_decile"] = pd.qcut(df[prob_col], n_buckets, labels=False, duplicates="drop") + 1
    return df


def decile_metrics(
    preds: pd.DataFrame,
    ret_col: str = "fwd_ret_1h",
    cost_bps: float = 15.0,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """Compute per-decile summary metrics.

    Returns one row per (model_name, score_decile) with:
    mean_gross_return, mean_net_return, median_net_return, hit_rate,
    trade_count, asset_composition, avg_realized_vol, avg_drawdown.
    """
    cost = cost_bps / 10_000.0
    df = assign_deciles(preds, n_buckets=n_buckets, group_cols=["model_name", "fold_id"])

    rows: list[dict] = []
    for (model, decile), grp in df.groupby(["model_name", "score_decile"]):
        gross = grp[ret_col].values
        # Approximate net: each trade incurs entry + exit cost
        net = gross - 2 * cost
        r: dict = {
            "model_name": model,
            "score_decile": decile,
            "mean_gross_return": np.nanmean(gross),
            "mean_net_return": np.nanmean(net),
            "median_net_return": np.nanmedian(net),
            "hit_rate": (gross > 0).mean() if len(gross) > 0 else np.nan,
            "trade_count": len(grp),
        }
        # Asset composition
        if "asset" in grp.columns:
            comp = grp["asset"].value_counts(normalize=True).to_dict()
            r["asset_composition"] = comp
        # Average regime descriptors (if available)
        for col in ("realized_vol_24h", "drawdown_168h", "drawdown_24h"):
            if col in grp.columns:
                r[f"avg_{col}"] = grp[col].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def tail_quantile_metrics(
    preds: pd.DataFrame,
    quantiles: list[float] = (0.80, 0.90, 0.95, 0.975),
    ret_col: str = "fwd_ret_1h",
    cost_bps: float = 15.0,
) -> pd.DataFrame:
    """Compute metrics for top-quantile buckets (top 20%, 10%, 5%, 2.5%).

    Returns one row per (model_name, quantile_threshold).
    """
    cost = cost_bps / 10_000.0
    rows: list[dict] = []

    for model, mgrp in preds.groupby("model_name"):
        for q in quantiles:
            threshold = mgrp["y_pred_prob"].quantile(q)
            tail = mgrp[mgrp["y_pred_prob"] >= threshold]
            if len(tail) == 0:
                continue
            gross = tail[ret_col].values
            net = gross - 2 * cost
            r: dict = {
                "model_name": model,
                "tail_quantile": q,
                "tail_label": f"top_{100 * (1 - q):.1f}%",
                "prob_threshold": threshold,
                "mean_gross_return": np.nanmean(gross),
                "mean_net_return": np.nanmean(net),
                "hit_rate": (gross > 0).mean(),
                "trade_count": len(tail),
                "fold_count": tail["fold_id"].nunique(),
            }
            # Fold stability for this tail
            fold_returns = tail.groupby("fold_id")[ret_col].mean()
            r["fold_profitability"] = (fold_returns > 0).mean()
            # Asset concentration
            if "asset" in tail.columns:
                comp = tail["asset"].value_counts(normalize=True)
                r["max_asset_share"] = comp.max()
                r["dominant_asset"] = comp.idxmax()
            rows.append(r)
    return pd.DataFrame(rows)


def check_monotonicity(decile_df: pd.DataFrame, metric: str = "mean_net_return") -> pd.DataFrame:
    """Check whether per-decile metric is non-decreasing for each model.

    Returns one row per model with monotonicity flag and Spearman rank
    correlation of (decile → metric).
    """
    from scipy.stats import spearmanr

    rows = []
    for model, grp in decile_df.groupby("model_name"):
        sorted_g = grp.sort_values("score_decile")
        vals = sorted_g[metric].values
        is_mono = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
        rho, pval = spearmanr(sorted_g["score_decile"].values, vals)
        rows.append({
            "model_name": model,
            "is_monotonic": is_mono,
            "spearman_rho": rho,
            "spearman_pval": pval,
        })
    return pd.DataFrame(rows)
