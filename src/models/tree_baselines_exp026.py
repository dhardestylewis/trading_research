"""Tree baseline models (Ridge, LightGBM, CatBoost) for exp026."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from src.utils.logging import get_logger

log = get_logger("tree_baselines_exp026")

class TreeBaselinesExp026:
    """Trains Ridge, LightGBM, and CatBoost baselines per horizon."""
    
    def __init__(self, horizons: list[int]):
        self.horizons = horizons
        self.models = {}
        
    def fit(self, X_train: pd.DataFrame, horizons_df: pd.DataFrame):
        """
        X_train: features dataframe
        horizons_df: dataframe containing targets like gross_move_bps_{h}
        """
        for h in self.horizons:
            self.models[h] = {}
            target_col = f"gross_move_bps_{h}"
            
            if target_col not in horizons_df.columns:
                continue
                
            y_train = horizons_df[target_col]
            valid_idx = ~y_train.isna()
            
            X_valid = X_train[valid_idx]
            y_valid = y_train[valid_idx]
            
            # Fill NAs for linear models & Catboost if needed
            X_filled = X_valid.fillna(0)
            
            # 1. Ridge
            log.info(f"Fitting Ridge for {h} bars...")
            ridge = Ridge(alpha=10.0)
            ridge.fit(X_filled, y_valid)
            self.models[h]["Ridge"] = ridge
            
            # 2. LightGBM
            log.info(f"Fitting LightGBM for {h} bars...")
            lgb_model = lgb.LGBMRegressor(
                num_leaves=31,
                max_depth=5,
                learning_rate=0.03,
                n_estimators=150,
                verbose=-1,
                random_state=42
            )
            lgb_model.fit(X_valid, y_valid)
            self.models[h]["LightGBM"] = lgb_model
            
            # 3. CatBoost
            log.info(f"Fitting CatBoost for {h} bars...")
            cat_model = CatBoostRegressor(
                iterations=150,
                depth=5,
                learning_rate=0.03,
                verbose=False,
                random_seed=42,
                allow_writing_files=False
            )
            cat_model.fit(X_filled, y_valid)
            self.models[h]["CatBoost"] = cat_model
            
    def predict(self, X_test: pd.DataFrame) -> dict[int, pd.DataFrame]:
        """Returns predictions per horizon."""
        preds = {}
        X_filled = X_test.fillna(0)
        
        for h in self.horizons:
            if h not in self.models:
                continue
                
            h_preds = pd.DataFrame(index=X_test.index)
            
            if "Ridge" in self.models[h]:
                h_preds["pred_Ridge"] = self.models[h]["Ridge"].predict(X_filled)
                
            if "LightGBM" in self.models[h]:
                h_preds["pred_LightGBM"] = self.models[h]["LightGBM"].predict(X_test)
                
            if "CatBoost" in self.models[h]:
                h_preds["pred_CatBoost"] = self.models[h]["CatBoost"].predict(X_filled)
                
            preds[h] = h_preds
            
        return preds
