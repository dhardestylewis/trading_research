"""Branch B — Cost Prediction Model.

Trains lightweight models (LightGBM, CatBoost, quantile baselines) to predict
short-horizon execution shortfall via strict rolling walk-forward validation.

Targets: shortfall_1s_bps, shortfall_5s_bps, adverse_markout_1s/5s_bps
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional
from scipy.stats import spearmanr

logger = logging.getLogger("cost_surface_model")

# ── Feature list (execution-bearing only) ────────────────────────
DEFAULT_FEATURES = [
    "quoted_spread_bps", "spread_percentile", "signed_volume_1s",
    "signed_volume_5s", "flow_imbalance", "trade_burst_5s",
    "vwap_dislocation", "recent_realized_volatility", "trade_count",
    "top_of_book_size", "book_imbalance", "hour_of_day",
    "weekend_indicator",
    # from raw flow bar columns
    "spread_bps", "trade_count_burst_intensity",
    "buyer_maker_seller_maker_imbalance",
]

DEFAULT_TARGETS = [
    "shortfall_1s_bps",
    "shortfall_5s_bps",
    "adverse_markout_1s_bps",
    "adverse_markout_5s_bps",
]


class QuantileBaselineModel:
    """Per-asset × hour conditional quantile baseline."""

    def __init__(self):
        self.lookup: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        df = X[["asset", "hour_of_day"]].copy() if "asset" in X.columns else X.copy()
        df["_target"] = y.values

        if "asset" in df.columns and "hour_of_day" in df.columns:
            grouped = df.groupby(["asset", "hour_of_day"])["_target"].median()
            self.lookup = grouped.to_dict()
        else:
            self.lookup = {"__global__": df["_target"].median()}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "asset" in X.columns and "hour_of_day" in X.columns:
            return np.array([
                self.lookup.get((row.get("asset"), row.get("hour_of_day")),
                               np.nanmedian(list(self.lookup.values())))
                for _, row in X[["asset", "hour_of_day"]].iterrows()
            ])
        global_val = self.lookup.get("__global__", 0.0)
        return np.full(len(X), global_val)


def _get_valid_features(df: pd.DataFrame, feature_list: List[str]) -> List[str]:
    """Return features that actually exist in the dataframe."""
    return [f for f in feature_list if f in df.columns]


def _train_lgbm(X_train, y_train, X_val, y_val, params: dict):
    """Train a LightGBM regressor with early stopping."""
    import lightgbm as lgb

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    lgb_params = {
        "objective": "regression",
        "metric": "mae",
        "num_leaves": params.get("num_leaves", 31),
        "max_depth": params.get("max_depth", 5),
        "learning_rate": params.get("learning_rate", 0.03),
        "min_child_samples": params.get("min_child_samples", 100),
        "subsample": params.get("subsample", 0.8),
        "colsample_bytree": params.get("colsample_bytree", 0.7),
        "reg_alpha": params.get("reg_alpha", 0.1),
        "reg_lambda": params.get("reg_lambda", 1.0),
        "verbose": -1,
    }

    model = lgb.train(
        lgb_params, dtrain,
        valid_sets=[dval],
        num_boost_round=params.get("n_estimators", 300),
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )
    return model


def _train_catboost(X_train, y_train, X_val, y_val, params: dict):
    """Train a CatBoost regressor."""
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        logger.warning("CatBoost not installed; falling back to LightGBM")
        return _train_lgbm(X_train, y_train, X_val, y_val, params)

    cat_features = [c for c in X_train.columns if X_train[c].dtype == "object"]
    model = CatBoostRegressor(
        iterations=params.get("n_estimators", 300),
        learning_rate=params.get("learning_rate", 0.03),
        depth=params.get("max_depth", 5),
        verbose=0,
        early_stopping_rounds=20,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val),
              cat_features=cat_features if cat_features else None)
    return model


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute Spearman correlation, MAE, and decile separation."""
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() < 20:
        return {"spearman": np.nan, "mae": np.nan, "decile_sep": np.nan}

    yt, yp = y_true[valid], y_pred[valid]
    sp, _ = spearmanr(yt, yp)
    mae = np.mean(np.abs(yt - yp))

    # Top-decile vs bottom-decile realised cost separation
    try:
        deciles = pd.qcut(yp, 10, labels=False, duplicates="drop")
        top = yt[deciles == deciles.max()]
        bot = yt[deciles == deciles.min()]
        decile_sep = float(np.median(top) - np.median(bot))
    except (ValueError, IndexError):
        decile_sep = np.nan

    return {"spearman": float(sp), "mae": float(mae), "decile_sep": decile_sep}


class CostSurfaceModelTrainer:
    """Orchestrates walk-forward training of cost prediction models."""

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        targets: Optional[List[str]] = None,
        model_classes: Optional[List[str]] = None,
    ):
        self.feature_cols = feature_cols or DEFAULT_FEATURES
        self.targets = targets or DEFAULT_TARGETS
        self.model_classes = model_classes or ["LightGBM"]
        self.results: Dict[str, Any] = {}

    def train_walk_forward(
        self,
        surface: pd.DataFrame,
        train_window_seconds: int = 259200,  # 3 days
        forward_slice_seconds: int = 3600,    # 1 hour
        model_params: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Run strict rolling walk-forward for all targets × model classes.

        Returns dict keyed by target×model with OOS predictions, metrics,
        and feature importances.
        """
        if model_params is None:
            model_params = {}

        feats = _get_valid_features(surface, self.feature_cols)
        if not feats:
            logger.error("No valid features found in surface columns: %s", list(surface.columns)[:20])
            return {}

        logger.info("Walk-forward: %d features, %d targets, %d model classes",
                     len(feats), len(self.targets), len(self.model_classes))

        # Ensure datetime index
        if not isinstance(surface.index, pd.DatetimeIndex):
            if "timestamp" in surface.columns:
                surface = surface.set_index("timestamp").sort_index()
            else:
                surface = surface.sort_index()

        all_results = {}

        for target in self.targets:
            if target not in surface.columns:
                logger.warning("Target %s not found — skipping", target)
                continue

            valid = surface.dropna(subset=[target] + feats)
            if len(valid) < 1000:
                logger.warning("Insufficient data (%d rows) for target %s", len(valid), target)
                continue

            # Time boundaries
            t_min, t_max = valid.index.min(), valid.index.max()
            total_seconds = (t_max - t_min).total_seconds()

            if total_seconds < train_window_seconds + forward_slice_seconds:
                logger.warning("Insufficient time span for walk-forward on %s", target)
                continue

            for model_class in self.model_classes:
                key = f"{target}__{model_class}"
                logger.info("Training %s ...", key)

                oof_indices = []
                oof_preds = []
                fold_metrics = []
                feature_importances = []
                fold = 0

                cursor = t_min + pd.Timedelta(seconds=train_window_seconds)
                while cursor < t_max:
                    train_end = cursor
                    train_start = cursor - pd.Timedelta(seconds=train_window_seconds)
                    test_end = min(cursor + pd.Timedelta(seconds=forward_slice_seconds), t_max)

                    train_mask = (valid.index >= train_start) & (valid.index < train_end)
                    test_mask = (valid.index >= train_end) & (valid.index < test_end)

                    X_tr = valid.loc[train_mask, feats]
                    y_tr = valid.loc[train_mask, target]
                    X_te = valid.loc[test_mask, feats]
                    y_te = valid.loc[test_mask, target]

                    if len(X_tr) < 100 or len(X_te) < 10:
                        cursor += pd.Timedelta(seconds=forward_slice_seconds)
                        continue

                    if model_class == "LightGBM":
                        model = _train_lgbm(X_tr, y_tr, X_te, y_te, model_params)
                        preds = model.predict(X_te)
                        fi = pd.Series(
                            model.feature_importance(importance_type="gain"),
                            index=feats,
                        )
                        feature_importances.append(fi)
                    elif model_class == "CatBoost":
                        model = _train_catboost(X_tr, y_tr, X_te, y_te, model_params)
                        preds = model.predict(X_te)
                    elif model_class == "quantile_baseline":
                        qb = QuantileBaselineModel()
                        qb.fit(X_tr, y_tr)
                        preds = qb.predict(X_te)
                    else:
                        logger.warning("Unknown model class: %s", model_class)
                        cursor += pd.Timedelta(seconds=forward_slice_seconds)
                        continue

                    oof_indices.extend(X_te.index.tolist())
                    oof_preds.extend(preds.tolist())

                    metrics = _compute_metrics(y_te.values, preds)
                    metrics["fold"] = fold
                    metrics["n_train"] = len(X_tr)
                    metrics["n_test"] = len(X_te)
                    fold_metrics.append(metrics)

                    cursor += pd.Timedelta(seconds=forward_slice_seconds)
                    fold += 1

                if not oof_preds:
                    logger.warning("No folds completed for %s", key)
                    continue

                # Aggregate
                oof_df = pd.DataFrame({
                    "timestamp": oof_indices,
                    f"pred_{target}": oof_preds,
                }).set_index("timestamp")

                fold_df = pd.DataFrame(fold_metrics)
                avg_metrics = {
                    "mean_spearman": fold_df["spearman"].mean(),
                    "mean_mae": fold_df["mae"].mean(),
                    "mean_decile_sep": fold_df["decile_sep"].mean(),
                    "n_folds": len(fold_df),
                }

                fi_agg = None
                if feature_importances:
                    fi_agg = pd.concat(feature_importances, axis=1).mean(axis=1).sort_values(ascending=False)

                all_results[key] = {
                    "oof_predictions": oof_df,
                    "fold_metrics": fold_df,
                    "aggregate_metrics": avg_metrics,
                    "feature_importance": fi_agg,
                }
                logger.info(
                    "%s — Spearman: %.4f, MAE: %.2f, decile_sep: %.2f (%d folds)",
                    key, avg_metrics["mean_spearman"], avg_metrics["mean_mae"],
                    avg_metrics["mean_decile_sep"], avg_metrics["n_folds"],
                )

        self.results = all_results
        return all_results
