"""Extract interpretable cells from latent states.

For each discovered cluster, computes conditional net expectancy,
ranks clusters by conservative metrics, and fits shallow decision
trees to produce human-readable rule descriptions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from src.utils.logging import get_logger

log = get_logger("exp019.cell_extractor")


def _cluster_economics(
    combined: pd.DataFrame,
    labels: np.ndarray,
    exec_cfg: dict,
) -> pd.DataFrame:
    """Compute per-cluster economic statistics."""
    df = combined.copy()
    df["cluster"] = labels

    records: list[dict] = []
    for cid in sorted(df["cluster"].unique()):
        mask = df["cluster"] == cid
        cluster = df[mask]

        # Use OOF predictions where available, fall back to actuals
        net_col = "actual_net_move_bps"
        gross_col = "actual_gross_move_bps"
        if net_col not in cluster.columns:
            net_col = "pred_net_move_bps"
            gross_col = "pred_gross_move_bps"

        net = cluster[net_col].dropna()
        gross = cluster[gross_col].dropna() if gross_col in cluster.columns else pd.Series(dtype=float)

        if len(net) < 10:
            continue

        # Stress test: subtract additional friction
        stress_extra = exec_cfg.get("round_trip_bps", 30) * (
            exec_cfg.get("stress_multiplier", 1.5) - 1
        )
        net_stressed = net - stress_extra

        # Trimmed mean (10% each side)
        trim_frac = 0.10
        n_trim = max(1, int(len(net) * trim_frac))
        sorted_net = net.sort_values().values
        trimmed = sorted_net[n_trim:-n_trim] if len(sorted_net) > 2 * n_trim else sorted_net
        trimmed_mean = trimmed.mean()

        # Asset breakdown
        n_assets = cluster["asset"].nunique() if "asset" in cluster.columns else 0
        asset_counts = cluster["asset"].value_counts() if "asset" in cluster.columns else pd.Series(dtype=int)
        top_asset_frac = asset_counts.iloc[0] / len(cluster) if len(asset_counts) > 0 else 1.0

        records.append({
            "cluster_id": cid,
            "n_samples": len(net),
            "median_net_bps": round(net.median(), 2),
            "mean_net_bps": round(net.mean(), 2),
            "trimmed_mean_net_bps": round(trimmed_mean, 2),
            "std_net_bps": round(net.std(), 2),
            "pct_positive": round((net > 0).mean() * 100, 1),
            "median_gross_bps": round(gross.median(), 2) if len(gross) > 0 else np.nan,
            "stressed_median_bps": round(net_stressed.median(), 2),
            "stressed_mean_bps": round(net_stressed.mean(), 2),
            "n_assets": n_assets,
            "top_asset_frac": round(top_asset_frac, 4),
        })

    return pd.DataFrame(records).sort_values(
        "trimmed_mean_net_bps", ascending=False
    ).reset_index(drop=True)


def _fit_interpretable_tree(
    features: pd.DataFrame,
    feature_cols: list[str],
    labels: np.ndarray,
    target_cluster: int,
    max_depth: int,
) -> str:
    """Fit a shallow decision tree to describe a cluster in human-readable rules."""
    X = features[feature_cols].values.copy()
    X[np.isnan(X)] = 0
    y = (labels == target_cluster).astype(int)

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=50,
        class_weight="balanced",
    )
    tree.fit(X, y)

    rule_text = export_text(tree, feature_names=feature_cols, max_depth=max_depth)
    return rule_text


def _get_prototypes(
    features: pd.DataFrame,
    labels: np.ndarray,
    target_cluster: int,
    n_prototypes: int,
) -> pd.DataFrame:
    """Get representative examples from a cluster."""
    mask = labels == target_cluster
    cluster_features = features[mask]

    if len(cluster_features) <= n_prototypes:
        return cluster_features[["asset", "timestamp"]].copy()

    # Sample evenly across time
    idx = np.linspace(0, len(cluster_features) - 1, n_prototypes, dtype=int)
    return cluster_features.iloc[idx][["asset", "timestamp"]].copy()


def extract_cells(
    features: pd.DataFrame,
    combined_predictions: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list[str],
    exec_cfg: dict,
    extraction_cfg: dict,
) -> dict:
    """Extract and rank interpretable cells from discovered latent states.

    Parameters
    ----------
    features : extended feature DataFrame
    combined_predictions : OOF predictions DataFrame
    labels : cluster assignments
    feature_cols : feature column names
    exec_cfg : execution config
    extraction_cfg : cell extraction config from YAML

    Returns
    -------
    dict with:
        cell_economics : DataFrame ranked by conservative net expectancy
        cell_cards : list of dicts with rule descriptions and prototypes
    """
    max_depth = extraction_cfg.get("max_tree_depth", 3)
    n_prototypes = extraction_cfg.get("n_prototypes", 5)
    top_n = extraction_cfg.get("top_cells", 10)

    # Compute economics per cluster
    cell_economics = _cluster_economics(combined_predictions, labels, exec_cfg)
    log.info("Cell economics computed for %d clusters", len(cell_economics))

    # Build cell cards for top clusters
    top_clusters = cell_economics.head(top_n)
    cell_cards: list[dict] = []

    for _, row in top_clusters.iterrows():
        cid = int(row["cluster_id"])

        # Fit interpretable tree
        rule_text = _fit_interpretable_tree(
            features, feature_cols, labels, cid, max_depth
        )

        # Get prototypes
        prototypes = _get_prototypes(features, labels, cid, n_prototypes)

        card = {
            "cluster_id": cid,
            "n_samples": int(row["n_samples"]),
            "median_net_bps": row["median_net_bps"],
            "mean_net_bps": row["mean_net_bps"],
            "trimmed_mean_net_bps": row["trimmed_mean_net_bps"],
            "pct_positive": row["pct_positive"],
            "n_assets": int(row["n_assets"]),
            "stressed_median_bps": row["stressed_median_bps"],
            "rule_description": rule_text,
            "prototypes": prototypes.to_dict("records"),
        }
        cell_cards.append(card)

        log.info(
            "  Cell %d: n=%d, median_net=%.1f, trimmed_mean=%.1f, "
            "assets=%d, pct_pos=%.1f%%",
            cid, card["n_samples"], card["median_net_bps"],
            card["trimmed_mean_net_bps"], card["n_assets"],
            card["pct_positive"],
        )

    return {
        "cell_economics": cell_economics,
        "cell_cards": cell_cards,
    }
