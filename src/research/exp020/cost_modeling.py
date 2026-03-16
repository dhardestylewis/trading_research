"""Core execution cost modeling logic for exp020."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, List

from src.utils.logging import get_logger

log = get_logger("cost_modeling")

def train_cost_model(
    features: pd.DataFrame, 
    target_series: pd.Series, 
    feature_cols: List[str], 
    model_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Train a LightGBM model to predict execution cost / toxicity.
    Uses walk-forward validation.
    """
    log.info("Training Execution Cost Model on %d features...", len(feature_cols))
    
    wf_cfg = model_cfg["walk_forward"]
    min_train_bars = wf_cfg["min_train_bars"]
    val_bars = wf_cfg["val_bars"]
    step_bars = wf_cfg["step_bars"]

    # We assume features DataFrame is sorted temporally.
    unique_times = features["timestamp"].sort_values().unique()
    
    oof_predictions = np.full(len(features), np.nan)
    models = []
    
    train_end_idx = min_train_bars
    fold_idx = 0
    
    while train_end_idx < len(unique_times):
        val_end_idx = min(train_end_idx + val_bars, len(unique_times))
        
        train_times = unique_times[:train_end_idx]
        val_times = unique_times[train_end_idx:val_end_idx]
        
        train_mask = features["timestamp"].isin(train_times)
        val_mask = features["timestamp"].isin(val_times)
        
        X_train = features.loc[train_mask, feature_cols]
        y_train = target_series.loc[train_mask]
        
        X_val = features.loc[val_mask, feature_cols]
        y_val = target_series.loc[val_mask]
        
        if len(y_val) == 0:
            break
            
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        clf = lgb.train(
            model_cfg["params"],
            train_data,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(model_cfg["early_stopping_rounds"], verbose=False)]
        )
        
        preds = clf.predict(X_val)
        oof_predictions[val_mask] = preds
        models.append(clf)
        
        train_end_idx += step_bars
        fold_idx += 1
        
    log.info("Completed %d walk-forward folds.", fold_idx)
    
    # Calculate global correlation
    valid_mask = ~np.isnan(oof_predictions) & ~np.isnan(target_series)
    corr = np.corrcoef(oof_predictions[valid_mask], target_series[valid_mask])[0, 1]
    log.info("Global OOF Correlation for Exec Loss: %.4f", corr)
    
    # Feature Importance (average across folds)
    if models:
        importance_df = pd.DataFrame()
        for i, m in enumerate(models):
            imp = m.feature_importance(importance_type='gain')
            importance_df[f'fold_{i}'] = imp
        importance_df['mean_gain'] = importance_df.mean(axis=1)
        importance_df.index = feature_cols
        top_features = importance_df.sort_values('mean_gain', ascending=False).head(10)
        log.info("Top 5 Features by Gain:\n%s", top_features.head(5)[['mean_gain']])
    else:
        importance_df = pd.DataFrame()

    return {
        "models": models,
        "oof_predictions": oof_predictions,
        "correlation": corr,
        "feature_importance": importance_df
    }

def build_cost_surface(features: pd.DataFrame, oof_preds: np.ndarray) -> pd.DataFrame:
    """Build a categorized surface of execution costs based on predictions and time regimes."""
    surface_df = features[["asset", "timestamp"]].copy()
    surface_df["pred_exec_loss_bps"] = oof_preds
    
    # Create toxicity regimes (Deciles of predicted execution loss)
    non_nan_preds = surface_df["pred_exec_loss_bps"].dropna()
    if len(non_nan_preds) > 0:
        surface_df["toxicity_decile"] = pd.qcut(surface_df["pred_exec_loss_bps"], 10, labels=False, duplicates='drop')
    
    # Time regimes
    surface_df["hour"] = pd.to_datetime(surface_df["timestamp"]).dt.hour
    
    return surface_df
