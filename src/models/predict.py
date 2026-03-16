"""Unified prediction interface for all model types."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.utils.logging import get_logger

log = get_logger("predict")


@dataclass
class TrainedModel:
    """Container for a trained model + optional scaler."""
    name: str
    model: Any
    scaler: StandardScaler | None = None
    feature_names: list[str] = field(default_factory=list)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(positive class) for each row."""
        Xm = X[self.feature_names].values if self.feature_names else X.values
        if self.scaler is not None:
            Xm = self.scaler.transform(Xm)

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(Xm)[:, 1]
        # fallback for models that only predict 0/1
        return self.model.predict(Xm).astype(float)


class NaiveMomentumModel:
    """Baseline: predict profitable if most recent 1h return was positive."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if "ret_1h" in X.columns:
            return (X["ret_1h"] > 0).astype(float).values
        return np.full(len(X), 0.5)
