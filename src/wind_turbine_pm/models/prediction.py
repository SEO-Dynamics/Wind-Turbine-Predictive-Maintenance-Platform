"""Pure scoring helpers.

These functions contain no IO and no artifact loading, which makes them trivial
to unit-test and safe to reuse from the service, the training scripts and the
evaluation scripts alike.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wind_turbine_pm.features.failure_features import align_to_feature_order


class FeatureContractError(ValueError):
    """Raised when a feature matrix does not match the model's contract."""


def validate_feature_frame(features: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    """Check and align a feature frame against the model's feature contract.

    Args:
        features: Candidate feature frame.
        expected: Feature names in the order the model expects.

    Returns:
        The frame reindexed to exactly ``expected``.

    Raises:
        FeatureContractError: If features are missing or the frame is empty.
    """
    if features.empty:
        raise FeatureContractError("Feature frame is empty")
    try:
        return align_to_feature_order(features, expected)
    except ValueError as exc:
        raise FeatureContractError(str(exc)) from exc


def predict_proba(estimator: Any, features: pd.DataFrame, expected: list[str]) -> np.ndarray:
    """Score a feature frame, returning positive-class probabilities.

    Args:
        estimator: Fitted estimator exposing ``predict_proba``.
        features: Feature frame.
        expected: Feature names in model order.

    Returns:
        A 1-D array of probabilities in ``[0, 1]``.

    Raises:
        FeatureContractError: If the frame does not match the contract.
    """
    aligned = validate_feature_frame(features, expected)
    probabilities = estimator.predict_proba(aligned)[:, 1]
    return np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)


def apply_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Convert probabilities to binary predictions.

    Args:
        probabilities: Positive-class probabilities.
        threshold: Decision threshold.

    Returns:
        An ``int`` array of 0/1 predictions.

    Raises:
        ValueError: If the threshold is outside ``[0, 1]``.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must lie in [0, 1], got {threshold}")
    return (np.asarray(probabilities) >= threshold).astype(int)
