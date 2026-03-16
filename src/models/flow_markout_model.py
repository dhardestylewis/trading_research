import pandas as pd
import lightgbm as lgb
import logging

logger = logging.getLogger("flow_markout_model")

class GrossMarkoutModel:
    """
    Lightweight tabular model trained offline to predict 1s and 5s
    gross markout edge from short-horizon flow features.
    """
    def __init__(self, model_type="LightGBM"):
        self.model_type = model_type
        self.model_1s = None
        self.model_5s = None
        self.features = [
            'signed_volume_1s', 'flow_imbalance', 'buyer_maker_seller_maker_imbalance',
            'trade_count_burst_intensity', 'vwap_dislocation', 'spread_bps'
        ]
        logger.info(f"Initialized GrossMarkoutModel with {model_type}")

    def train(self, X_train: pd.DataFrame, target_1s="markout_1s", target_5s="markout_5s"):
        logger.info(f"Training offline models for 1s and 5s targets using {self.model_type}")
        valid_cols = [c for c in self.features if c in X_train.columns]
        
        if valid_cols and target_1s in X_train.columns:
            d_train_1s = lgb.Dataset(X_train[valid_cols], label=X_train[target_1s])
            params = {'objective': 'regression', 'verbose': -1, 'learning_rate': 0.05}
            self.model_1s = lgb.train(params, d_train_1s, num_boost_round=100)
            
        if valid_cols and target_5s in X_train.columns:
            d_train_5s = lgb.Dataset(X_train[valid_cols], label=X_train[target_5s])
            params = {'objective': 'regression', 'verbose': -1, 'learning_rate': 0.05}
            self.model_5s = lgb.train(params, d_train_5s, num_boost_round=100)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fast scoring function for executing during live paper validation."""
        valid_cols = [c for c in self.features if c in X.columns]
        
        pred_1s = self.model_1s.predict(X[valid_cols]) if self.model_1s else [0.0]*len(X)
        pred_5s = self.model_5s.predict(X[valid_cols]) if self.model_5s else [0.0]*len(X)
        
        return pd.DataFrame({'pred_1s_gross': pred_1s, 'pred_5s_gross': pred_5s}, index=X.index)

class TrivialBaselines:
    """Implementations of non-ML filter baselines required for Gate 1."""
    @staticmethod
    def flow_sign_baseline(flow_imbalance_1s: pd.Series) -> pd.Series:
        return flow_imbalance_1s.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    @staticmethod
    def signed_volume_shock_baseline(signed_vol: pd.Series, threshold: float) -> pd.Series:
        return signed_vol.apply(lambda x: 1 if x > threshold else (-1 if x < -threshold else 0))

    @staticmethod
    def vwap_dislocation_baseline(vwap_disloc: pd.Series, threshold: float) -> pd.Series:
        return vwap_disloc.apply(lambda x: 1 if x < -threshold else (-1 if x > threshold else 0))
