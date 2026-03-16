"""Signal-to-position mapping (threshold policy)."""
from __future__ import annotations
import numpy as np


def long_flat_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Return position array: 1 where prob > threshold, else 0."""
    return (probabilities > threshold).astype(float)
