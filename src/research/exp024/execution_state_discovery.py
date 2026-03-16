"""Branch D — Low-Cost State Discovery.

Uses clustering and shallow trees on execution-bearing features to discover
interpretable "cheap execution cells."  This replaces hand-authored execution
rules with data-driven regime identification.
"""
import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, List
from pathlib import Path

logger = logging.getLogger("execution_state_discovery")


def _cluster_execution_states(
    surface: pd.DataFrame,
    feature_cols: List[str],
    n_clusters: int = 8,
    method: str = "kmeans",
) -> pd.Series:
    """Cluster timestamps into execution regimes."""
    from sklearn.preprocessing import StandardScaler

    valid_feats = [f for f in feature_cols if f in surface.columns]
    if not valid_feats:
        logger.warning("No valid features for clustering; returning zeros")
        return pd.Series(0, index=surface.index, name="exec_cluster")

    X = surface[valid_feats].copy()
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == "kmeans":
        from sklearn.cluster import KMeans
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    else:
        from sklearn.mixture import GaussianMixture
        clusterer = GaussianMixture(n_components=n_clusters, random_state=42)

    labels = clusterer.fit_predict(X_scaled)
    return pd.Series(labels, index=surface.index, name="exec_cluster")


def _train_cost_classifier(
    surface: pd.DataFrame,
    feature_cols: List[str],
    cost_col: str = "shortfall_1s_bps",
    low_cost_quantile: float = 0.25,
    max_depth: int = 4,
) -> Dict:
    """Train a shallow decision tree to classify low-cost states."""
    from sklearn.tree import DecisionTreeClassifier, export_text

    valid_feats = [f for f in feature_cols if f in surface.columns]
    if not valid_feats or cost_col not in surface.columns:
        logger.warning("Insufficient data for cost classifier")
        return {"tree_rules": "N/A", "accuracy": 0.0}

    threshold = surface[cost_col].quantile(low_cost_quantile)
    y = (surface[cost_col] <= threshold).astype(int)

    X = surface[valid_feats].fillna(surface[valid_feats].median())

    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42, min_samples_leaf=100)
    tree.fit(X, y)

    rules = export_text(tree, feature_names=valid_feats, max_depth=max_depth)
    accuracy = tree.score(X, y)

    logger.info("Cost classifier accuracy: %.3f", accuracy)
    logger.info("Tree rules:\n%s", rules)

    return {
        "tree": tree,
        "tree_rules": rules,
        "accuracy": accuracy,
        "threshold_bps": float(threshold),
        "feature_names": valid_feats,
    }


def _build_state_cards(
    surface: pd.DataFrame,
    cluster_col: str = "exec_cluster",
    cost_col: str = "shortfall_1s_bps",
) -> pd.DataFrame:
    """Build interpretable state cards for each discovered cluster."""
    if cluster_col not in surface.columns or cost_col not in surface.columns:
        return pd.DataFrame()

    cards = []
    for cid in sorted(surface[cluster_col].unique()):
        mask = surface[cluster_col] == cid
        subset = surface[mask]
        card = {
            "cluster": cid,
            "sample_size": len(subset),
            "median_shortfall_bps": subset[cost_col].median(),
            "mean_shortfall_bps": subset[cost_col].mean(),
            "p25_shortfall_bps": subset[cost_col].quantile(0.25),
            "p75_shortfall_bps": subset[cost_col].quantile(0.75),
        }

        # Asset mix
        if "asset" in subset.columns:
            asset_pcts = subset["asset"].value_counts(normalize=True)
            card["asset_mix"] = "; ".join(
                f"{a}: {p:.0%}" for a, p in asset_pcts.items()
            )
            card["dominant_asset"] = asset_pcts.index[0]
            card["dominant_asset_pct"] = float(asset_pcts.iloc[0])

        # Spread regime
        if "quoted_spread_bps" in subset.columns:
            card["median_spread_bps"] = subset["quoted_spread_bps"].median()

        # Hour distribution
        if "hour_of_day" in subset.columns:
            card["median_hour"] = int(subset["hour_of_day"].median())

        # Weekend fraction
        if "weekend_indicator" in subset.columns:
            card["weekend_frac"] = subset["weekend_indicator"].mean()

        # Prototype timestamps (5 representative)
        if isinstance(subset.index, pd.DatetimeIndex):
            sample_idx = np.linspace(0, len(subset) - 1, min(5, len(subset)), dtype=int)
            card["prototype_timestamps"] = "; ".join(
                str(subset.index[i]) for i in sample_idx
            )

        cards.append(card)

    cards_df = pd.DataFrame(cards).sort_values("median_shortfall_bps")
    return cards_df


def discover_execution_states(
    surface: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    n_clusters: int = 8,
    output_path: Optional[str] = None,
) -> Dict:
    """Main entry: discover low-cost execution states.

    Parameters
    ----------
    surface : pd.DataFrame
        Cost surface from Branch A.
    feature_cols : list, optional
        Execution-bearing features for clustering.
    n_clusters : int
        Number of states to discover.
    output_path : str, optional
        Path to save execution_cells.csv.

    Returns
    -------
    dict with keys: state_cards (DataFrame), tree_classifier (dict),
    surface_with_clusters (DataFrame)
    """
    if feature_cols is None:
        feature_cols = [
            "quoted_spread_bps", "spread_percentile", "signed_volume_1s",
            "flow_imbalance", "trade_burst_5s", "recent_realized_volatility",
            "book_imbalance", "hour_of_day", "weekend_indicator",
        ]

    logger.info("Discovering execution states with %d clusters...", n_clusters)

    # Step 1: Cluster
    surface["exec_cluster"] = _cluster_execution_states(
        surface, feature_cols, n_clusters=n_clusters
    )

    # Step 2: Shallow cost classifier
    tree_result = _train_cost_classifier(surface, feature_cols)

    # Step 3: State cards
    state_cards = _build_state_cards(surface)
    logger.info("State cards:\n%s", state_cards.to_string())

    if output_path:
        state_cards.to_csv(output_path, index=False)
        logger.info("Saved execution cells to %s", output_path)

    return {
        "state_cards": state_cards,
        "tree_classifier": tree_result,
        "surface_with_clusters": surface,
    }
