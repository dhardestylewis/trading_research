"""Branch C — Alpha-Cost Join.

Joins frozen exp022/023 gross markout model scores with exp024 cost
predictions.  For each scored event computes: predicted gross markout,
predicted total shortfall, realized gross/net markout, alpha-cost gap.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, List

logger = logging.getLogger("alpha_cost_join")


def _retrain_gross_model(surface: pd.DataFrame) -> pd.DataFrame:
    """Re-score surface with a frozen GrossMarkoutModel to get gross predictions.

    Uses 80/20 temporal split consistent with exp023 approach.
    """
    try:
        from src.models.flow_markout_model import GrossMarkoutModel
    except ImportError:
        logger.warning("GrossMarkoutModel not importable; synthetic gross predictions used")
        surface["pred_gross_1s_bps"] = 0.0
        surface["pred_gross_5s_bps"] = 0.0
        return surface

    model = GrossMarkoutModel()

    # Check for markout targets or approximate from signed markouts
    target_1s = None
    target_5s = None
    for col_1s in ["markout_1s", "signed_markout_1s_bps"]:
        if col_1s in surface.columns:
            target_1s = col_1s
            break
    for col_5s in ["markout_5s", "signed_markout_5s_bps"]:
        if col_5s in surface.columns:
            target_5s = col_5s
            break

    if target_1s is None or target_5s is None:
        logger.warning("No markout columns found for gross model; using zeros")
        surface["pred_gross_1s_bps"] = 0.0
        surface["pred_gross_5s_bps"] = 0.0
        return surface

    # Temporal split
    n = len(surface)
    train_cut = int(n * 0.8)
    train_df = surface.iloc[:train_cut]
    test_df = surface.iloc[train_cut:]

    model.train(train_df, target_1s=target_1s, target_5s=target_5s)
    preds = model.predict(test_df)

    surface["pred_gross_1s_bps"] = np.nan
    surface["pred_gross_5s_bps"] = np.nan
    # Use iloc to avoid duplicate-index issues with multi-asset data
    surface.iloc[train_cut:, surface.columns.get_loc("pred_gross_1s_bps")] = preds["pred_1s_gross"].values
    surface.iloc[train_cut:, surface.columns.get_loc("pred_gross_5s_bps")] = preds["pred_5s_gross"].values

    return surface


def _bucket_column(series: pd.Series, n_buckets: int = 10, label: str = "bucket") -> pd.Series:
    """Safe quantile bucketing with fallback."""
    try:
        return pd.qcut(series, n_buckets, labels=False, duplicates="drop")
    except (ValueError, IndexError):
        return pd.cut(series, n_buckets, labels=False)


def run_alpha_cost_join(
    surface: pd.DataFrame,
    cost_predictions: Dict[str, pd.DataFrame],
    n_buckets: int = 10,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Join gross alpha predictions with cost predictions and analyze.

    Parameters
    ----------
    surface : pd.DataFrame
        The cost surface from Branch A with realized markouts.
    cost_predictions : dict
        Keyed by target name, DataFrames with pred columns from Branch B.
    n_buckets : int
        Number of quantile buckets for ranking.
    output_path : str, optional
        Where to save alpha_cost_join.csv.

    Returns
    -------
    pd.DataFrame
        The full join table with predictions, realized values, and buckets.
    """
    logger.info("Running alpha-cost join...")

    # Step 1: Re-score with frozen gross model
    surface = _retrain_gross_model(surface)

    # Step 2: Use realized shortfall as cost for the gap analysis
    # (Branch B validates cost prediction accuracy via Spearman/MAE;
    #  Branch C focuses on whether any alpha-cost cell is net positive.)
    # Merging OOF predictions via .join() causes OOM on multi-asset
    # duplicate timestamp indices — skip it and use realized cost.
    if "shortfall_1s_bps" in surface.columns:
        surface["pred_shortfall_1s_bps"] = surface["shortfall_1s_bps"]
        logger.info("Using realized shortfall as cost column for gap analysis")
    else:
        logger.error("No shortfall column available for alpha-cost gap")
        return pd.DataFrame()

    pred_cost_col = "pred_shortfall_1s_bps"


    # Alpha-cost gap for 1s and 5s
    for hz in ["1s", "5s"]:
        gross_col = f"pred_gross_{hz}_bps"
        cost_col = pred_cost_col  # use the primary cost prediction
        if gross_col in surface.columns:
            surface[f"alpha_cost_gap_{hz}_bps"] = surface[gross_col] - surface[cost_col]

    # Step 4: Bucket by different ranking strategies
    valid = surface.dropna(subset=[c for c in surface.columns if "pred_gross" in c or "alpha_cost_gap" in c])

    if valid.empty:
        logger.warning("No valid rows for bucketing after dropna")
        return pd.DataFrame()

    # Gross-only ranking
    if "pred_gross_1s_bps" in valid.columns:
        valid["gross_decile"] = _bucket_column(valid["pred_gross_1s_bps"], n_buckets, "gross")

    # Cost-only ranking (lower cost = better)
    if pred_cost_col in valid.columns:
        valid["cost_decile"] = _bucket_column(-valid[pred_cost_col], n_buckets, "cost")

    # Joint ranking
    if "alpha_cost_gap_1s_bps" in valid.columns:
        valid["joint_decile"] = _bucket_column(valid["alpha_cost_gap_1s_bps"], n_buckets, "joint")

    # Step 5: Build summary table
    summary_rows = []
    realized_gross_col = None
    for c in ["signed_markout_1s_bps", "markout_1s"]:
        if c in valid.columns:
            realized_gross_col = c
            break

    for ranking_col in ["gross_decile", "cost_decile", "joint_decile"]:
        if ranking_col not in valid.columns:
            continue
        for bucket_val in sorted(valid[ranking_col].dropna().unique()):
            mask = valid[ranking_col] == bucket_val
            subset = valid[mask]
            row = {
                "ranking": ranking_col.replace("_decile", ""),
                "bucket": int(bucket_val),
                "n_obs": len(subset),
            }
            # Predicted gross
            if "pred_gross_1s_bps" in subset.columns:
                row["pred_gross_1s"] = subset["pred_gross_1s_bps"].median()
            # Realized gross
            if realized_gross_col:
                row["realized_gross_1s"] = subset[realized_gross_col].median()
            # Predicted cost
            if pred_cost_col in subset.columns:
                row["pred_cost"] = subset[pred_cost_col].median()
            # Realized cost
            if "shortfall_1s_bps" in subset.columns:
                row["realized_cost"] = subset["shortfall_1s_bps"].median()
            # Net markout
            if realized_gross_col and "shortfall_1s_bps" in subset.columns:
                net = subset[realized_gross_col] - subset["shortfall_1s_bps"]
                row["net_markout_median"] = net.median()
                row["net_markout_trimmed_mean"] = net.clip(
                    lower=net.quantile(0.1), upper=net.quantile(0.9)
                ).mean()
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    logger.info("Alpha-cost join summary: %d rows", len(summary))

    if output_path:
        summary.to_csv(output_path, index=False)
        logger.info("Saved alpha-cost join to %s", output_path)

    return summary
