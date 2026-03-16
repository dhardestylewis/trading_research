"""Multi-horizon and multi-target modeling for exp025."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge, LogisticRegression
from src.utils.logging import get_logger

log = get_logger("multi_horizon_model")

class MultiHorizonModel:
    """Trains multi-horizon and multi-target models across folds."""
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.horizons = cfg["horizons"]
        self.model_cfg = cfg["model"]
        self.exec_cfg = cfg["execution"]
        
        # We will store models per horizon and per target
        self.models = {}
        
    def fit(self, X_train: pd.DataFrame, horizons_data: dict[str, pd.DataFrame]):
        """
        X_train: features
        horizons_data: dict mapping horizon -> df with targets
        """
        for h in self.horizons:
            h_str = h["label"]
            h_bars = h["bars"]
            y_df = horizons_data[h_str]
            
            self.models[h_str] = {}
            
            # Gross move (Regression)
            if "lightgbm" in self.model_cfg["type"].lower():
                lgb_params = self.model_cfg["params"].copy()
                lgb_params["objective"] = "regression"
                lgb_params["metric"] = "rmse"
                
                model_gross = lgb.LGBMRegressor(**lgb_params)
                y_gross = y_df[f"fwd_ret_{h_bars}"]
                
                # Drop NaNs
                valid_idx = ~y_gross.isna()
                model_gross.fit(X_train[valid_idx], y_gross[valid_idx])
                self.models[h_str]["gross"] = model_gross
                
                # Probability tails (Classification)
                for tail in [25, 50, 100]:
                    tail_col = f"prob_tail_{tail}_{h_bars}"
                    if tail_col in y_df.columns:
                        clf_params = self.model_cfg["params"].copy()
                        clf_params["objective"] = "binary"
                        clf_params["metric"] = "binary_logloss"
                        model_clf = lgb.LGBMClassifier(**clf_params)
                        model_clf.fit(X_train[valid_idx], y_df[tail_col][valid_idx])
                        self.models[h_str][f"prob_{tail}"] = model_clf
            
            # Linear Baseline (Regression)
            rc = Ridge(alpha=1.0)
            valid_idx = ~y_df[f"fwd_ret_{h_bars}"].isna()
            X_filled = X_train[valid_idx].fillna(0)
            rc.fit(X_filled, y_df[f"fwd_ret_{h_bars}"][valid_idx])
            self.models[h_str]["linear_gross"] = rc
            
    def predict(self, X_test: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Returns predictions per horizon."""
        preds = {}
        cost_bps = self.exec_cfg["round_trip_bps"]
        
        for h in self.horizons:
            h_str = h["label"]
            h_preds = pd.DataFrame(index=X_test.index)
            
            if h_str in self.models:
                if "gross" in self.models[h_str]:
                    # Convert to bps
                    h_preds["pred_gross_bps"] = self.models[h_str]["gross"].predict(X_test) * 10000
                    h_preds["pred_net_bps"] = h_preds["pred_gross_bps"] - cost_bps
                    
                if "linear_gross" in self.models[h_str]:
                    X_filled = X_test.fillna(0)
                    h_preds["pred_linear_gross_bps"] = self.models[h_str]["linear_gross"].predict(X_filled) * 10000
                    
                for tail in [25, 50, 100]:
                    k = f"prob_{tail}"
                    if k in self.models[h_str]:
                        # Predict prob of class 1
                        probs = self.models[h_str][k].predict_proba(X_test)
                        if probs.shape[1] > 1:
                            h_preds[f"pred_prob_{tail}"] = probs[:, 1]
                        else:
                            h_preds[f"pred_prob_{tail}"] = probs[:, 0]
                            
            preds[h_str] = h_preds
            
        return preds
