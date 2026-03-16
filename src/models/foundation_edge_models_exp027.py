import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple
import xgboost as xgb

# ---------------------------------------------------------------------------
# TabPFN / Torch / Sklearn Monkey Patches from exp026
# ---------------------------------------------------------------------------
# PyTorch >= 2.0 Monkey Patch for TabPFN 0.1.9
import torch
import torch.nn.modules.transformer
import typing
import sklearn.utils.validation

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
# ---------------------------------------------------------------------------


logger = logging.getLogger(__name__)

class TabPFNTopDecile:
    """
    Directly isolates the TabPFN classifier mapping established in exp026.
    We bin continuous targets to 10 classes and predict expected value.
    This model receives the pre-compressed 90-dim PCA matrix and truncated N rows.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = TabPFNClassifier(device='cpu', N_ensemble_configurations=16) # CPU explicitly for local
        self.bin_edges = None
        self.bin_centers = None
        
    def _digitize_target(self, y_train: np.ndarray, num_bins: int = 10) -> np.ndarray:
        logger.info(f"Binning continuous target into {num_bins} uniform bins for TabPFN...")
        quantiles = np.linspace(0, 1, num_bins + 1)
        self.bin_edges = np.quantile(y_train, quantiles)
        
        # Enforce strict outer edges to catch all outliers
        self.bin_edges[0] = -np.inf
        self.bin_edges[-1] = np.inf
        
        y_binned = np.digitize(y_train, self.bin_edges[1:-1])
        
        self.bin_centers = np.zeros(num_bins)
        for i in range(num_bins):
            mask = (y_binned == i)
            if np.any(mask):
                self.bin_centers[i] = y_train[mask].mean()
            else:
                self.bin_centers[i] = 0.0
                
        return y_binned

    def fit(self, X_train: np.ndarray, y_train: pd.Series):
        if X_train.shape[1] > 100:
             raise ValueError(f"TabPFN violates feature limit: {X_train.shape[1]}")
        if X_train.shape[0] > 1024:
             raise ValueError(f"TabPFN violates sample limit: {X_train.shape[0]}")
             
        y_train_np = y_train.values
        y_binned = self._digitize_target(y_train_np)
        
        logger.info(f"Fitting TabPFN on {X_train.shape[0]} rows and {X_train.shape[1]} dims")
        self.model.fit(X_train, y_binned)
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        logger.info("Predicting using TabPFN...")
        probas = self.model.predict_proba(X_test)
        num_expected_classes = len(self.bin_centers)
        
        if probas.shape[1] < num_expected_classes:
            padded_probas = np.zeros((probas.shape[0], num_expected_classes))
            for i, c in enumerate(self.model.classes_):
                if c < num_expected_classes:
                    padded_probas[:, c] = probas[:, i]
            probas = padded_probas
            
        expected_values = np.dot(probas, self.bin_centers)
        return expected_values

class XGBoostCounterBaseline:
    """
    Max-capacity gradient boosting baseline. 
    It intentionally receives the exact same constrained PCA inputs to prove whether 
    the TabPFN edge is just "good tuning" or a structural foundation advantage.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=6,             # Deeper than default baselines to compete with Foundation
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
    def fit(self, X_train: np.ndarray, y_train: pd.Series):
        logger.info("Fitting heavy XGBoost counter-baseline...")
        self.model.fit(X_train, y_train.values)
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        logger.info("Predicting using heavy XGBoost...")
        return self.model.predict(X_test)

def get_challenger(model_name: str, config: Dict[str, Any]) -> Any:
    if model_name == "TabPFNTopDecile":
        return TabPFNTopDecile(config)
    elif model_name == "XGBoostCounterBaseline":
        return XGBoostCounterBaseline(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")
