"""Branch D — Execution-aware alpha prediction.

Build models that predict net alpha conditional on expected execution
cost, not raw directional return.

Sweeps over slippage assumptions (0–20 bps), trains LightGBM on
adjusted targets, compares to raw-target baseline under identical policy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.sparse_policy import threshold_separation_policy
from src.models.train_lightgbm import train as train_lightgbm
from src.utils.logging import get_logger

log = get_logger("execution_aware_alpha")


def build_execution_adjusted_targets(
    merged: pd.DataFrame,
    slippage_bps: float,
    horizons: list[int] = (1,),
) -> pd.DataFrame:
    """Construct execution-adjusted label columns.

    For each horizon, create:
      fwd_net_alpha_{h}h = fwd_ret_{h}h - (2 * slippage / 10000)
      fwd_net_profitable_{h}h = 1[fwd_net_alpha_{h}h > 0]

    The '2×' accounts for round-trip cost (entry + exit slippage).
    """
    df = merged.copy()
    slip = slippage_bps / 10_000.0

    for h in horizons:
        ret_col = f"fwd_ret_{h}h"
        if ret_col not in df.columns:
            continue
        df[f"fwd_net_alpha_{h}h"] = df[ret_col] - 2 * slip
        df[f"fwd_net_profitable_{h}h"] = (df[f"fwd_net_alpha_{h}h"] > 0).astype(float)

    return df


def _train_and_evaluate(
    merged: pd.DataFrame,
    fold_df: pd.DataFrame,
    feat_cols: list[str],
    target_col: str,
    ret_col: str,
    *,
    target_asset: str = "SOL-USD",
    threshold: float = 0.55,
    sep_gap: int = 3,
    cost_bps: float = 15.0,
    label: str = "raw",
    bars_per_year: float = 365.25 * 24,
) -> dict:
    """Train LightGBM on given target, deploy on target asset, evaluate."""
    fold_df = fold_df.copy()
    for col in ("start", "end"):
        if fold_df[col].dt.tz is not None:
            fold_df[col] = fold_df[col].dt.tz_localize(None)

    cost = cost_bps / 10_000.0
    all_preds: list[pd.DataFrame] = []

    for fold_id in sorted(fold_df["fold_id"].unique()):
        fold_rows = fold_df[fold_df["fold_id"] == fold_id]
        train_r = fold_rows[fold_rows["split"] == "train"]
        test_r = fold_rows[fold_rows["split"] == "test"]
        if train_r.empty or test_r.empty:
            continue
        train_r = train_r.iloc[0]
        test_r = test_r.iloc[0]

        ts = merged["timestamp"]
        train_mask = (ts >= train_r["start"]) & (ts < train_r["end"])
        test_mask = (ts >= test_r["start"]) & (ts < test_r["end"])

        train_df = merged[train_mask].copy()
        test_df = merged[test_mask & (merged["asset"] == target_asset)].copy()

        required = feat_cols + [target_col, ret_col]
        for df_ in (train_df, test_df):
            valid = df_[required].notna().all(axis=1)
            df_.drop(df_[~valid].index, inplace=True)

        if len(train_df) < 50 or len(test_df) < 5:
            continue

        tm = train_lightgbm(
            train_df[feat_cols], train_df[target_col],
            config_path="configs/models/lightgbm_v1.yaml",
            feature_names=feat_cols,
        )
        probs = tm.predict_proba(test_df)

        pred_df = test_df[["asset", "timestamp", ret_col]].copy()
        pred_df["y_pred_prob"] = probs
        pred_df["fold_id"] = fold_id
        all_preds.append(pred_df)

    if not all_preds:
        return {
            "target_variant": label,
            "trade_count": 0,
            "sharpe": np.nan,
            "mean_net_bps": np.nan,
            "hit_rate": np.nan,
            "fold_profitability": np.nan,
        }

    preds = pd.concat(all_preds, ignore_index=True).sort_values("timestamp")
    probs = preds["y_pred_prob"].values
    positions = threshold_separation_policy(probs, threshold, sep_gap)
    active = positions > 0

    if active.sum() < 3:
        return {
            "target_variant": label,
            "trade_count": 0,
            "sharpe": np.nan,
            "mean_net_bps": np.nan,
            "hit_rate": np.nan,
            "fold_profitability": np.nan,
        }

    gross = preds[ret_col].values[active]
    net = gross - 2 * cost
    mean_net = np.nanmean(net)
    std_net = np.nanstd(net)
    sharpe = (mean_net / std_net * np.sqrt(bars_per_year)) if std_net > 0 else 0.0

    fold_ids = preds["fold_id"].values[active]
    fold_rets = pd.DataFrame({"fold_id": fold_ids, "ret": gross}).groupby("fold_id")["ret"].mean()
    fold_prof = (fold_rets > 0).mean() if len(fold_rets) > 0 else 0.0

    return {
        "target_variant": label,
        "trade_count": int(active.sum()),
        "sharpe": sharpe,
        "cumulative_return": (1 + net).prod() - 1,
        "mean_net_bps": mean_net * 10_000,
        "hit_rate": (gross > 0).mean(),
        "fold_profitability": fold_prof,
    }


def compare_target_variants(
    merged: pd.DataFrame,
    fold_df: pd.DataFrame,
    feat_cols: list[str],
    slippage_scenarios_bps: list[float] = (0, 5, 10, 15, 20),
    *,
    target_asset: str = "SOL-USD",
    threshold: float = 0.55,
    sep_gap: int = 3,
    cost_bps: float = 15.0,
) -> pd.DataFrame:
    """Compare raw vs execution-adjusted targets across slippage levels."""
    rows: list[dict] = []

    # Baseline: raw target
    log.info("  Training with raw target (fwd_profitable_1h)")
    raw_row = _train_and_evaluate(
        merged, fold_df, feat_cols,
        target_col="fwd_profitable_1h",
        ret_col="fwd_ret_1h",
        target_asset=target_asset,
        threshold=threshold,
        sep_gap=sep_gap,
        cost_bps=cost_bps,
        label="raw_fwd_profitable_1h",
    )
    raw_row["slippage_bps"] = 0
    rows.append(raw_row)

    # Execution-aware variants
    for slip in slippage_scenarios_bps:
        if slip == 0:
            continue  # already covered by raw baseline
        label = f"exec_aware_{slip}bps"
        log.info("  Training with execution-aware target (slippage=%d bps)", slip)

        adjusted = build_execution_adjusted_targets(merged, slip, horizons=[1])
        target_col = "fwd_net_profitable_1h"

        if target_col not in adjusted.columns:
            continue

        row = _train_and_evaluate(
            adjusted, fold_df, feat_cols,
            target_col=target_col,
            ret_col="fwd_ret_1h",
            target_asset=target_asset,
            threshold=threshold,
            sep_gap=sep_gap,
            cost_bps=cost_bps,
            label=label,
        )
        row["slippage_bps"] = slip
        rows.append(row)

    return pd.DataFrame(rows)


def execution_aware_study(
    merged: pd.DataFrame,
    fold_df: pd.DataFrame,
    feat_cols: list[str],
    *,
    target_asset: str = "SOL-USD",
    slippage_scenarios_bps: list[float] = (0, 5, 10, 15, 20),
    threshold: float = 0.55,
    sep_gap: int = 3,
    cost_bps: float = 15.0,
) -> pd.DataFrame:
    """Run the full execution-aware alpha study.

    Returns DataFrame with one row per target variant × slippage scenario.
    """
    log.info("═══ Branch D: Execution-aware alpha ═══")

    result = compare_target_variants(
        merged, fold_df, feat_cols,
        slippage_scenarios_bps=list(slippage_scenarios_bps),
        target_asset=target_asset,
        threshold=threshold,
        sep_gap=sep_gap,
        cost_bps=cost_bps,
    )

    if not result.empty:
        raw_bps = result.loc[result["target_variant"] == "raw_fwd_profitable_1h", "mean_net_bps"].values
        raw_bps = raw_bps[0] if len(raw_bps) > 0 else 0.0
        result["improvement_vs_raw_bps"] = result["mean_net_bps"] - raw_bps

        for _, row in result.iterrows():
            log.info("  %s (slip=%d): %d trades, Sharpe=%.2f, net=%.1f bps, delta=%.1f bps",
                     row["target_variant"], row.get("slippage_bps", 0),
                     row["trade_count"], row.get("sharpe", 0),
                     row.get("mean_net_bps", 0), row.get("improvement_vs_raw_bps", 0))

    return result
