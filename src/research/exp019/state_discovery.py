"""Latent state discovery via clustering.

Discovers market states from the extended feature matrix using
Gaussian Mixture Models or KMeans. Profiles each cluster with
mean feature values, temporal distribution, and asset composition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from src.utils.logging import get_logger

log = get_logger("exp019.state_discovery")


def _normalize_features(
    features: pd.DataFrame, feature_cols: list[str],
) -> tuple[np.ndarray, StandardScaler]:
    """Z-score normalize features, handling NaN by filling with 0."""
    scaler = StandardScaler()
    X = features[feature_cols].values.copy()
    # Fill NaN with column median, then z-score
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def _fit_cluster_model(
    X: np.ndarray, method: str, n_clusters: int, random_state: int,
):
    """Fit a clustering model and return labels + model."""
    if method == "gmm":
        model = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            n_init=3,
            random_state=random_state,
            max_iter=200,
        )
        model.fit(X)
        labels = model.predict(X)
        score = model.bic(X)  # lower is better for GMM
    elif method == "kmeans":
        model = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=random_state,
            max_iter=300,
        )
        model.fit(X)
        labels = model.labels_
        score = model.inertia_  # lower is better
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return labels, model, score


def _profile_clusters(
    features: pd.DataFrame, feature_cols: list[str],
    labels: np.ndarray, min_cluster_frac: float,
) -> pd.DataFrame:
    """Build cluster profiles with mean features, size, and asset mix."""
    df = features.copy()
    df["cluster"] = labels
    n_total = len(df)

    profiles: list[dict] = []
    for cid in sorted(df["cluster"].unique()):
        mask = df["cluster"] == cid
        cluster_df = df[mask]
        n = len(cluster_df)
        frac = n / n_total

        profile = {
            "cluster_id": cid,
            "n_samples": n,
            "frac_of_total": round(frac, 4),
            "viable": frac >= min_cluster_frac,
        }

        # Mean of each feature
        for col in feature_cols:
            profile[f"mean_{col}"] = cluster_df[col].mean()

        # Asset breakdown
        if "asset" in df.columns:
            asset_counts = cluster_df["asset"].value_counts()
            profile["n_assets"] = len(asset_counts)
            profile["top_asset"] = asset_counts.index[0]
            profile["top_asset_frac"] = round(
                asset_counts.iloc[0] / n, 4
            )

        profiles.append(profile)

    return pd.DataFrame(profiles)


def discover_states(
    features: pd.DataFrame, cfg: dict,
) -> dict:
    """Run latent state discovery.

    Parameters
    ----------
    features : DataFrame with [asset, timestamp, ...feature_cols...]
    cfg : state_discovery config dict from YAML

    Returns
    -------
    dict with keys:
        best_labels : np.ndarray of cluster assignments
        best_n_clusters : int
        best_score : float
        best_model : fitted model
        scaler : StandardScaler
        profiles : DataFrame of cluster profiles
        sweep_results : list of dicts with n_clusters, score
        feature_cols : list of feature column names
    """
    method = cfg.get("method", "gmm")
    n_clusters_sweep = cfg.get("n_clusters", [8, 12, 16])
    min_cluster_frac = cfg.get("min_cluster_frac", 0.02)
    do_normalize = cfg.get("normalize", True)
    random_state = cfg.get("random_state", 42)
    stop_early = cfg.get("stop_if_all_viable", True)

    # Identify feature columns (everything except asset, timestamp)
    meta_cols = {"asset", "timestamp"}
    feature_cols = [c for c in features.columns if c not in meta_cols]

    # Normalize
    if do_normalize:
        X, scaler = _normalize_features(features, feature_cols)
    else:
        X = features[feature_cols].values.copy()
        X[np.isnan(X)] = 0
        scaler = None

    # Sweep over cluster counts
    sweep_results: list[dict] = []
    best_score = np.inf
    best_labels = None
    best_model = None
    best_n = None

    for n_cl in n_clusters_sweep:
        log.info("Fitting %s with n_clusters=%d ...", method, n_cl)
        labels, model, score = _fit_cluster_model(X, method, n_cl, random_state)

        # Check how many clusters meet minimum size
        unique, counts = np.unique(labels, return_counts=True)
        viable_count = sum(c / len(labels) >= min_cluster_frac for c in counts)

        sweep_results.append({
            "n_clusters": n_cl,
            "score": score,
            "viable_clusters": viable_count,
            "total_clusters": len(unique),
        })
        log.info(
            "  n=%d → score=%.1f, viable=%d/%d clusters",
            n_cl, score, viable_count, len(unique),
        )

        if score < best_score:
            best_score = score
            best_labels = labels
            best_model = model
            best_n = n_cl

        # Early stopping: if all clusters are viable, no need for more granularity
        if stop_early and viable_count == len(unique):
            log.info("  Early stop: all %d clusters viable at n=%d, skipping larger n", viable_count, n_cl)
            break

    # Build profiles for best model
    profiles = _profile_clusters(features, feature_cols, best_labels, min_cluster_frac)

    log.info(
        "Best clustering: %s with n=%d (score=%.1f, %d viable clusters)",
        method, best_n, best_score,
        profiles["viable"].sum(),
    )

    return {
        "best_labels": best_labels,
        "best_n_clusters": best_n,
        "best_score": best_score,
        "best_model": best_model,
        "scaler": scaler,
        "profiles": profiles,
        "sweep_results": sweep_results,
        "feature_cols": feature_cols,
    }
