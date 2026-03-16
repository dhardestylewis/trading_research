import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging
from src.data.build_rich_perp_state_features import build_rich_perp_state_features

logger = logging.getLogger(__name__)

class FoundationFeatureBuilder:
    """
    Constructs the feature sets explicitly bound to the constraints required
    by foundation models (like TabPFN v0.1.9).
    
    1. Loads the rich state from exp026.
    2. Enforces PCA reduction to <100 features.
    3. Truncates training views to N <= 1000 dynamically during `get_train_slice`.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pca_components = config['features'].get('pca_components', 90)
        self.sc_X = StandardScaler()
        self.pca = PCA(n_components=self.pca_components, random_state=42)
        
        # State tracking for walk-forward persistence
        self.is_fit = False
        self.feature_cols = []
        
    def build_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Build standard rich features, but instead of returning 200+ raw columns,
        it fits/transforms them down to the precise foundation limits (90 PCA dims).
        """
        logger.info("Extracting rich base features for Foundation mapping...")
        horizons = self.config.get('targets', {}).get('horizons', [8])
        df_rich = build_rich_perp_state_features(df, horizons=horizons)
        
        # Get feature columns (exclude targets, metadata)
        exclude_cols = ['timestamp', 'asset', 'symbol']
        for h in horizons:
            exclude_cols.extend([
                f'fwd_ret_{h}', f'gross_move_bps_{h}', 
                f'prob_tail_25_{h}', f'prob_tail_50_{h}', f'prob_tail_100_{h}'
            ])
            
        raw_feats = [c for c in df_rich.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_rich[c])]
        
        logger.info(f"Generated {len(raw_feats)} raw rich features.")
        return df_rich, raw_feats
        
    def fit_transform_pca(self, X_train: pd.DataFrame) -> np.ndarray:
        """Fit scaler and PCA on training slice, return compressed vectors."""
        logger.info(f"Fitting PCA (target={self.pca_components}) to training set of shape {X_train.shape}")
        
        # Protect against NaN and Inf before PCA
        X_train_clean = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        # Adjust PCA components if we have fewer features than requested
        n_features = X_train_clean.shape[1]
        n_samples = X_train_clean.shape[0]
        actual_components = min(self.pca_components, n_features, n_samples)
        
        if self.pca.n_components != actual_components:
            logger.warning(f"Adjusting PCA components from {self.pca.n_components} to {actual_components} due to matrix limits.")
            self.pca = PCA(n_components=actual_components, random_state=42)
            self.pca_components = actual_components
            
        X_scaled = self.sc_X.fit_transform(X_train_clean)
        X_pca = self.pca.fit_transform(X_scaled)
        
        self.is_fit = True
        return X_pca
        
    def transform_pca(self, X_test: pd.DataFrame) -> np.ndarray:
        """Transform test slice using previously fitted PCA."""
        if not self.is_fit:
            raise ValueError("Must call fit_transform_pca before transform_pca")
            
        X_test_clean = X_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_scaled = self.sc_X.transform(X_test_clean)
        X_pca = self.pca.transform(X_scaled)
        
        return X_pca

    def enforce_n_constraint(self, train_df: pd.DataFrame, max_n: int = 1000) -> pd.DataFrame:
        """
        Foundation models (TabPFN v0.1.9) hard crash if N > 1024. 
        We downsample the history gracefully, prioritizing recent events 
        and high-volatility events within the N limit.
        """
        if len(train_df) <= max_n:
            return train_df.copy()
            
        logger.info(f"Downsampling training frame from {len(train_df)} to {max_n} rows for foundation constraint.")
        
        # Strategy: Keep the most recent 50% of quota. 
        # Randomly sample the remaining 50% of quota from the older history to maintain diversity.
        recent_quota = int(max_n * 0.5)
        older_quota = max_n - recent_quota
        
        recent_cutoff_idx = len(train_df) - recent_quota
        
        # The most recent rows
        df_recent = train_df.iloc[recent_cutoff_idx:]
        
        # The historical rows
        df_older = train_df.iloc[:recent_cutoff_idx].sample(n=older_quota, random_state=42)
        
        # Re-sort chronologically
        df_constrained = pd.concat([df_older, df_recent]).sort_index()
        return df_constrained
