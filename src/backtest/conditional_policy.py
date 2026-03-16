"""Conditional policy functions (Branch D).

Four policy types:
  D1  baseline_threshold   — same as exp001
  D2  tail_threshold       — trade only if score rank >= quantile
  D3  regime_threshold     — trade only if regime flag = 1
  D4  hybrid_threshold     — combine tail + regime
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.diagnostics.regime_labeller import REGIME_COLS
from src.utils.logging import get_logger

log = get_logger("conditional_policy")


def baseline_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    """D1: w_t = 1[p_t > τ]."""
    return (probs > threshold).astype(float)


def tail_threshold(
    probs: np.ndarray,
    threshold: float,
    quantile_rank: np.ndarray,
    q: float,
) -> np.ndarray:
    """D2: w_t = 1[p_t > τ] · 1[rank_t >= q].

    *quantile_rank* should be a per-fold rank (0–1) of the probability.
    """
    return ((probs > threshold) & (quantile_rank >= q)).astype(float)


def regime_threshold(
    probs: np.ndarray,
    threshold: float,
    regime_flags: np.ndarray,
) -> np.ndarray:
    """D3: w_t = 1[p_t > τ] · 1[regime_eligible].

    *regime_flags* is a binary array (1 = eligible).
    """
    return ((probs > threshold) & (regime_flags == 1)).astype(float)


def hybrid_threshold(
    probs: np.ndarray,
    threshold: float,
    quantile_rank: np.ndarray,
    q: float,
    regime_flags: np.ndarray,
) -> np.ndarray:
    """D4: w_t = 1[p_t > τ] · 1[rank_t >= q] · 1[regime_eligible]."""
    return (
        (probs > threshold)
        & (quantile_rank >= q)
        & (regime_flags == 1)
    ).astype(float)


def compute_quantile_ranks(
    preds: pd.DataFrame,
    prob_col: str = "y_pred_prob",
    group_cols: list[str] = ("model_name", "fold_id"),
) -> pd.Series:
    """Compute within-group quantile rank (0–1) for each prediction."""
    return preds.groupby(list(group_cols))[prob_col].rank(pct=True)


def evaluate_policies(
    preds: pd.DataFrame,
    thresholds: list[float] = (0.50, 0.55, 0.60, 0.65),
    tail_quantiles: list[float] = (0.80, 0.90, 0.95, 0.975),
    regime_filters: list[str] = ("high_vol", "high_liquidity", "drawdown_rebound", "trend_up"),
    hybrid_thresholds: list[float] = (0.55, 0.60, 0.65),
    hybrid_tail_quantiles: list[float] = (0.90, 0.95),
    hybrid_regime_filters: list[str] = ("high_vol", "high_liquidity", "drawdown_rebound"),
    ret_col: str = "fwd_ret_1h",
    cost_bps: float = 15.0,
    bars_per_year: float = 365.25 * 24,
) -> pd.DataFrame:
    """Evaluate all four policy types and return a comparison table."""
    cost = cost_bps / 10_000.0
    af = bars_per_year

    # Pre-compute quantile ranks
    preds = preds.copy()
    preds["_qrank"] = compute_quantile_ranks(preds)

    regime_col_map = {
        "high_vol": "regime_vol_high",
        "high_liquidity": "regime_liquidity_high",
        "drawdown_rebound": "regime_rebound",
        "trend_up": "regime_trend_up",
        "non_weekend": None,  # special: regime_weekend == 0
        "weekend": "regime_weekend",
    }

    def _compute_metrics(positions, gross_rets, fold_ids, assets, label):
        active = positions > 0
        if active.sum() < 5:
            return None
        net = gross_rets[active] - 2 * cost
        mean_n = np.nanmean(net)
        std_n = np.nanstd(net)
        sharpe = (mean_n / std_n * np.sqrt(af)) if std_n > 0 else 0.0
        cum = (1 + net).prod() - 1
        fold_df = pd.DataFrame({"fold_id": fold_ids[active], "ret": gross_rets[active]})
        fold_prof = (fold_df.groupby("fold_id")["ret"].mean() > 0).mean()
        asset_s = pd.Series(assets[active])
        max_asset = asset_s.value_counts(normalize=True).max() if len(asset_s) > 0 else np.nan
        return {
            **label,
            "sharpe": sharpe,
            "cumulative_return": cum,
            "hit_rate": (gross_rets[active] > 0).mean(),
            "trade_count": int(active.sum()),
            "fold_profitability": fold_prof,
            "max_asset_share": max_asset,
        }

    rows: list[dict] = []

    for model, mgrp in preds.groupby("model_name"):
        probs = mgrp["y_pred_prob"].values
        gross = mgrp[ret_col].values
        folds = mgrp["fold_id"].values
        assets = mgrp["asset"].values if "asset" in mgrp.columns else np.full(len(mgrp), "unknown")
        qranks = mgrp["_qrank"].values

        # D1: Baseline
        for tau in thresholds:
            pos = baseline_threshold(probs, tau)
            r = _compute_metrics(pos, gross, folds, assets,
                                 {"policy": "baseline", "model_name": model, "threshold": tau,
                                  "tail_quantile": None, "regime_filter": None})
            if r:
                rows.append(r)

        # D2: Tail-only
        for tau in thresholds:
            for q in tail_quantiles:
                pos = tail_threshold(probs, tau, qranks, q)
                r = _compute_metrics(pos, gross, folds, assets,
                                     {"policy": "tail_only", "model_name": model, "threshold": tau,
                                      "tail_quantile": q, "regime_filter": None})
                if r:
                    rows.append(r)

        # D3: Regime-gated
        for tau in thresholds:
            for rf_name in regime_filters:
                rcol = regime_col_map.get(rf_name, f"regime_{rf_name}")
                if rf_name == "non_weekend" and "regime_weekend" in mgrp.columns:
                    flags = (mgrp["regime_weekend"] == 0).astype(float).values
                elif rcol in mgrp.columns:
                    flags = mgrp[rcol].values
                else:
                    continue
                pos = regime_threshold(probs, tau, flags)
                r = _compute_metrics(pos, gross, folds, assets,
                                     {"policy": "regime_gated", "model_name": model, "threshold": tau,
                                      "tail_quantile": None, "regime_filter": rf_name})
                if r:
                    rows.append(r)

        # D4: Hybrid
        for tau in hybrid_thresholds:
            for q in hybrid_tail_quantiles:
                for rf_name in hybrid_regime_filters:
                    rcol = regime_col_map.get(rf_name, f"regime_{rf_name}")
                    if rcol not in mgrp.columns:
                        continue
                    flags = mgrp[rcol].values
                    pos = hybrid_threshold(probs, tau, qranks, q, flags)
                    r = _compute_metrics(pos, gross, folds, assets,
                                         {"policy": "hybrid", "model_name": model, "threshold": tau,
                                          "tail_quantile": q, "regime_filter": rf_name})
                    if r:
                        rows.append(r)

    return pd.DataFrame(rows)
