"""Train logistic regression model."""
from __future__ import annotations

import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.predict import TrainedModel
from src.utils.logging import get_logger

log = get_logger("train_logistic")


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    config_path: str | None = None,
    feature_names: list[str] | None = None,
) -> TrainedModel:
    """Train a logistic regression classifier and return a TrainedModel."""
    params = {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 1000}
    do_standardize = True

    if config_path:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        params.update(cfg.get("params", {}))
        do_standardize = cfg.get("standardize", True)

    feat_cols = feature_names or [c for c in X_train.columns if c not in ("asset", "timestamp")]
    X = X_train[feat_cols].values
    y = y_train.values

    # Drop rows with NaN
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    scaler = None
    if do_standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    clf = LogisticRegression(**params)
    clf.fit(X, y)
    log.info("Logistic trained on %d samples, classes=%s", len(y), np.unique(y).tolist())

    return TrainedModel(name="logistic_regression", model=clf, scaler=scaler, feature_names=feat_cols)
