"""Candidate model definitions.

Every candidate is returned as a complete :class:`~sklearn.pipeline.Pipeline`
so that imputation and scaling are fitted on the training split only and travel
with the model into the artifact.  This removes the most common leakage bug in
production ML: a scaler fitted on the full dataset.

Gradient boosting is provided by scikit-learn's ``HistGradientBoostingClassifier``
rather than XGBoost or LightGBM.  Both of those require a platform-specific
OpenMP runtime that is not reliably present on macOS or slim Docker images, and
the histogram-based scikit-learn implementation is the same algorithm family
with no extra dependency.  This is the "dependency stability" trade-off the
project brief allows; the decision is recorded in the README.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from wind_turbine_pm.config import Config
from wind_turbine_pm.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Candidate:
    """A named model candidate together with its fitting requirements."""

    name: str
    estimator: Pipeline
    supports_sample_weight: bool
    positive_class_weight: str | float | None = None

    @property
    def algorithm(self) -> str:
        """Class name of the final estimator in the pipeline."""
        return type(self.estimator.steps[-1][1]).__name__


def _params(cfg: Config, name: str) -> dict[str, Any]:
    params = cfg.get(f"training.candidates.{name}.params")
    return dict(params) if params is not None else {}


def _dummy(cfg: Config, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("model", DummyClassifier(random_state=seed, **_params(cfg, "dummy"))),
        ]
    )


def _logistic_regression(cfg: Config, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler(quantile_range=(5.0, 95.0))),
            ("model", LogisticRegression(random_state=seed, **_params(cfg, "logistic_regression"))),
        ]
    )


def _random_forest(cfg: Config, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(random_state=seed, **_params(cfg, "random_forest"))),
        ]
    )


def _hist_gradient_boosting(cfg: Config, seed: int) -> Pipeline:
    # HistGradientBoosting handles NaN natively, so no imputer is needed; the
    # missingness pattern itself carries signal (a dead sensor is informative).
    return Pipeline(
        [
            (
                "model",
                HistGradientBoostingClassifier(
                    random_state=seed, **_params(cfg, "hist_gradient_boosting")
                ),
            )
        ]
    )


_BUILDERS = {
    "dummy": (_dummy, False),
    "logistic_regression": (_logistic_regression, True),
    "random_forest": (_random_forest, True),
    "hist_gradient_boosting": (_hist_gradient_boosting, True),
}


def build_candidates(cfg: Config) -> list[Candidate]:
    """Instantiate every enabled candidate model.

    Args:
        cfg: Merged configuration.

    Returns:
        The enabled candidates, in configuration order.

    Raises:
        ValueError: If configuration names an unknown candidate, or none are
            enabled.
    """
    seed = int(cfg.get("random_seed", 42))
    configured = cfg.get("training.candidates") or {}
    candidates: list[Candidate] = []

    for name in configured:
        if not bool(cfg.get(f"training.candidates.{name}.enabled", False)):
            logger.info("Candidate disabled by configuration", extra={"candidate": name})
            continue
        if name not in _BUILDERS:
            raise ValueError(
                f"Unknown model candidate {name!r}; known candidates: {sorted(_BUILDERS)}"
            )
        builder, supports_weight = _BUILDERS[name]
        candidates.append(
            Candidate(
                name=name,
                estimator=builder(cfg, seed),
                supports_sample_weight=supports_weight,
                positive_class_weight=cfg.get(f"training.candidates.{name}.positive_class_weight"),
            )
        )

    if not candidates:
        raise ValueError("No model candidates are enabled in training.candidates")
    return candidates


def compute_sample_weights(
    y: np.ndarray, positive_class_weight: str | float | None
) -> np.ndarray | None:
    """Build per-sample weights that up-weight the rare positive class.

    Args:
        y: Binary label array.
        positive_class_weight: ``"auto"`` for the negative/positive ratio, a
            number for an explicit weight, or ``None`` for no weighting.

    Returns:
        A weight array, or ``None`` when no weighting is requested.

    Raises:
        ValueError: If ``positive_class_weight`` is an unrecognised string.
    """
    if positive_class_weight is None:
        return None

    if isinstance(positive_class_weight, str):
        if positive_class_weight != "auto":
            raise ValueError(
                f"positive_class_weight must be 'auto' or numeric, got {positive_class_weight!r}"
            )
        n_positive = float((y == 1).sum())
        n_negative = float((y == 0).sum())
        weight = n_negative / n_positive if n_positive > 0 else 1.0
    else:
        weight = float(positive_class_weight)

    weights = np.ones_like(y, dtype=float)
    weights[y == 1] = weight
    return weights
