"""Train LightGBM classifier."""
from __future__ import annotations

import yaml
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.models.predict import TrainedModel
from src.utils.logging import get_logger

log = get_logger("train_lightgbm")


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    config_path: str | None = None,
    feature_names: list[str] | None = None,
) -> TrainedModel:
    """Train a LightGBM binary classifier with early stopping and return a TrainedModel."""
    params = {
        "num_leaves": 31,
        "max_depth": 4,
        "learning_rate": 0.03,
        "n_estimators": 500,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
    }
    early_stopping_rounds = 50

    if config_path:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        params.update(cfg.get("params", {}))
        early_stopping_rounds = cfg.get("early_stopping_rounds", 50)

    feat_cols = feature_names or [c for c in X_train.columns if c not in ("asset", "timestamp")]

    Xt = X_train[feat_cols].values
    yt = y_train.values
    mask_t = np.isfinite(Xt).all(axis=1) & np.isfinite(yt)
    Xt, yt = Xt[mask_t], yt[mask_t]

    clf = lgb.LGBMClassifier(**params)

    fit_kwargs: dict = {}
    if X_val is not None and y_val is not None:
        Xv = X_val[feat_cols].values
        yv = y_val.values
        mask_v = np.isfinite(Xv).all(axis=1) & np.isfinite(yv)
        Xv, yv = Xv[mask_v], yv[mask_v]
        fit_kwargs["eval_set"] = [(Xv, yv)]
        fit_kwargs["callbacks"] = [lgb.early_stopping(early_stopping_rounds, verbose=False)]

    clf.fit(Xt, yt, **fit_kwargs)
    log.info("LightGBM trained on %d samples, best_iteration=%s", len(yt), getattr(clf, "best_iteration_", "N/A"))

    return TrainedModel(name="lightgbm", model=clf, scaler=None, feature_names=feat_cols)
