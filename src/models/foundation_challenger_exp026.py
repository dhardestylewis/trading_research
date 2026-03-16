"""Foundation Tabular Challenger for exp026."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# PyTorch >= 2.0 Monkey Patch for TabPFN 0.1.9
import torch
import torch.nn.modules.transformer
import typing

# Spoof typing and module exports stripped in PyTorch 2.0
torch.nn.modules.transformer.Optional = typing.Optional
torch.nn.modules.transformer.Tensor = torch.Tensor
torch.nn.modules.transformer.Module = torch.nn.Module
torch.nn.modules.transformer.Linear = torch.nn.Linear
torch.nn.modules.transformer.Dropout = torch.nn.Dropout
torch.nn.modules.transformer.LayerNorm = torch.nn.LayerNorm
torch.nn.modules.transformer.MultiheadAttention = torch.nn.MultiheadAttention

if not hasattr(torch.nn.modules.transformer, '_get_activation_fn'):
    def _get_activation_fn(activation: str):
        if activation == "relu":
            return torch.nn.functional.relu
        elif activation == "gelu":
            return torch.nn.functional.gelu
        raise RuntimeError(f"activation should be relu/gelu, not {activation}")
    torch.nn.modules.transformer._get_activation_fn = _get_activation_fn

# Scikit-learn >= 1.6 Monkey Patch for TabPFN 0.1.9
import sklearn.utils.validation
_orig_check_X_y = sklearn.utils.validation.check_X_y
def _patched_check_X_y(*args, **kwargs):
    if 'force_all_finite' in kwargs:
        kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
    return _orig_check_X_y(*args, **kwargs)
sklearn.utils.validation.check_X_y = _patched_check_X_y

_orig_check_array = sklearn.utils.validation.check_array
def _patched_check_array(*args, **kwargs):
    if 'force_all_finite' in kwargs:
        kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
    return _orig_check_array(*args, **kwargs)
sklearn.utils.validation.check_array = _patched_check_array

from tabpfn import TabPFNClassifier
from src.utils.logging import get_logger

log = get_logger("foundation_challenger_exp026")

class FoundationChallengerExp026:
    """Trainer and wrapper for TabPFNClassifier per horizon using continuous EV prediction."""
    def __init__(self, horizons: list[int], num_bins: int=5):
        self.horizons = horizons
        self.num_bins = num_bins
        self.models = {}
        self.bin_edges = {}
        self.bin_centers = {}
        self.pca = {}
        self.features = []
        
    def fit(self, X_train: pd.DataFrame, horizons_df: pd.DataFrame):
        self.features = [c for c in X_train.columns]
        
        for h in self.horizons:
            target_col = f"gross_move_bps_{h}"
            if target_col not in horizons_df.columns:
                continue
                
            y_train = horizons_df[target_col]
            valid_idx = ~y_train.isna()
            
            X_valid = X_train[valid_idx].copy()
            y_valid = y_train[valid_idx].copy()
            
            # TabPFN 0.1.9 strictly limits training size to 1024 rows.
            # Subsample to 1000 max.
            if len(X_valid) > 1000:
                sample_idx = np.random.choice(X_valid.index, size=1000, replace=False)
                X_valid = X_valid.loc[sample_idx]
                y_valid = y_valid.loc[sample_idx]

            # TabPFN v0.1.9 limits input to 100 features max
            if X_valid.shape[1] > 90:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_valid)
                pca = PCA(n_components=90, random_state=42)
                X_valid = pd.DataFrame(pca.fit_transform(X_scaled), index=X_valid.index)
                self.pca[h] = (scaler, pca)
            else:
                self.pca[h] = None

            # 1. Bucketize continuous target sequence into classification bins (quantile mapping)
            y_binned, bins = pd.qcut(y_valid, q=self.num_bins, retbins=True, labels=False, duplicates='drop')
            
            # 2. Extract bin centers for expected-value conversion
            # Using median of the original values within each bin as the center representation
            centers = []
            for i in range(len(bins)-1):
                mask = (y_valid >= bins[i]) & (y_valid <= bins[i+1])
                centers.append(y_valid[mask].median() if mask.any() else np.mean([bins[i], bins[i+1]]))
            
            self.bin_edges[h] = bins
            self.bin_centers[h] = np.array(centers)
                
            # TabPFN 0.1.9 classification model
            model = TabPFNClassifier(device="cpu", N_ensemble_configurations=1) # Auto-fallback + ensemble downscaling for speed
            model.fit(X_valid, y_binned)
            
            self.models[h] = model
            log.info(f"Fitted TabPFN Classifier for {h} bars. Continuous values mapped onto {len(centers)} bins via PCA compression.")

    def predict(self, X_test: pd.DataFrame) -> dict[int, pd.DataFrame]:
        preds = {}
        for h, model in self.models.items():
            X_raw = X_test[self.features].copy()
            
            # Apply PCA projection if fitted during training
            if self.pca[h] is not None:
                scaler, pca = self.pca[h]
                X_raw = pd.DataFrame(pca.transform(scaler.transform(X_raw)), index=X_raw.index)

            # Get probabilistic output across all bins
            probas = model.predict_proba(X_raw)
            
            # Reconstruct expected continuous value: Sum[ P(Class i) * ValueCenter(Class i) ]
            centers = self.bin_centers[h]
            
            # Alignment check: Map proba array dynamically to learned classes (in case empty bins were dropped)
            expected_values = np.zeros(len(X_test))
            for proba_idx, true_class in enumerate(model.classes_):
                expected_values += probas[:, proba_idx] * centers[true_class]
            
            preds_df = pd.DataFrame(index=X_test.index)
            # Use same column name to integrate seamlessly with the pipeline evaluator
            preds_df["pred_TabPFNRegressor"] = expected_values
            preds[h] = preds_df
            
        return preds
