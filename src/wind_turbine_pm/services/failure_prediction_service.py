"""The single prediction entry point for the Failure Prediction Module.

Both the FastAPI layer and the Streamlit dashboard use this class.  Prediction
logic - feature building, contract validation, thresholding, risk banding,
explanation and advisory generation - exists here and nowhere else, so the two
front ends can never drift apart.

The service supports both documented input modes:

* :meth:`predict_from_window` - the **preferred** mode.  Give it a window of raw
  observations and it computes the features itself using the same
  :func:`~wind_turbine_pm.features.failure_features.build_failure_features`
  function used in training.
* :meth:`predict_from_features` - the fallback.  Give it a precomputed feature
  dictionary matching the saved schema.

Artifacts are loaded lazily on first use and cached, so importing this module
never touches the filesystem and never triggers training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from wind_turbine_pm.config import Config, get_config
from wind_turbine_pm.constants import TIMESTAMP, TURBINE_ID, RiskLevel
from wind_turbine_pm.contracts.metadata import ModelMetadata
from wind_turbine_pm.contracts.observations import TurbineWindow
from wind_turbine_pm.contracts.predictions import (
    BatchFailurePrediction,
    FailurePrediction,
    RiskFactor,
)
from wind_turbine_pm.data.preprocessing import preprocess
from wind_turbine_pm.explainability.narratives import build_advisory, describe_factor
from wind_turbine_pm.explainability.shap_explainer import (
    ExplainerUnavailableError,
    FailureExplainer,
    LocalAttribution,
)
from wind_turbine_pm.features.failure_features import build_failure_features
from wind_turbine_pm.logging_config import get_logger
from wind_turbine_pm.models.persistence import (
    ModelBundle,
    bundle_available,
    check_runtime_compatibility,
    load_bundle,
)
from wind_turbine_pm.models.prediction import FeatureContractError, apply_threshold, predict_proba
from wind_turbine_pm.models.threshold import map_risk_level
from wind_turbine_pm.utils.io import ArtifactNotFoundError

logger = get_logger(__name__)


class ServiceNotReadyError(RuntimeError):
    """Raised when the service is asked to predict without usable artifacts."""

    def __init__(self, message: str, hint: str) -> None:
        """Store the failure reason and its remediation command.

        Args:
            message: What is missing.
            hint: Shell command that fixes it.
        """
        self.hint = hint
        super().__init__(f"{message} Run: {hint}")


class InsufficientHistoryError(ValueError):
    """Raised when a prediction window is too short for the feature windows."""


@dataclass(frozen=True)
class ServiceStatus:
    """Health information about the service and its artifacts."""

    model_loaded: bool
    model_version: str | None
    algorithm: str | None
    n_features: int
    threshold: float | None
    explainer_method: str | None
    detail: str
    runtime_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "model_loaded": self.model_loaded,
            "model_version": self.model_version,
            "algorithm": self.algorithm,
            "n_features": self.n_features,
            "threshold": self.threshold,
            "explainer_method": self.explainer_method,
            "detail": self.detail,
            "runtime_warnings": list(self.runtime_warnings),
        }


class FailurePredictionService:
    """Loads the published model and turns turbine data into advisory output."""

    def __init__(self, cfg: Config | None = None, eager: bool = False) -> None:
        """Create the service.

        Args:
            cfg: Merged configuration; loaded from disk when omitted.
            eager: Load artifacts immediately instead of on first use. Useful in
                tests; the API deliberately stays lazy so that a missing model
                does not prevent the process from starting.
        """
        self._cfg = cfg if cfg is not None else get_config()
        self._bundle: ModelBundle | None = None
        self._explainer: FailureExplainer | None = None
        self._load_error: str | None = None
        if eager:
            self._ensure_loaded()

    # -- Configuration accessors -------------------------------------------
    @property
    def config(self) -> Config:
        """The configuration this service was built with."""
        return self._cfg

    @property
    def horizon_hours(self) -> int:
        """The prediction horizon in hours."""
        return int(self._cfg.require("target.horizon_hours"))

    @property
    def risk_bands(self) -> tuple[float, float]:
        """The ``(low_max, medium_max)`` risk band cut-points."""
        return (
            float(self._cfg.require("risk_levels.low_max")),
            float(self._cfg.require("risk_levels.medium_max")),
        )

    # -- Artifact loading --------------------------------------------------
    def _ensure_loaded(self) -> ModelBundle:
        """Load and cache the model bundle, raising a helpful error on failure."""
        if self._bundle is not None:
            return self._bundle
        try:
            self._bundle = load_bundle(self._cfg)
        except ArtifactNotFoundError as exc:
            self._load_error = str(exc)
            raise ServiceNotReadyError("Model artifacts are not available.", exc.hint) from exc
        self._load_error = None
        return self._bundle

    @property
    def is_ready(self) -> bool:
        """Whether artifacts are present and loadable without raising."""
        if self._bundle is not None:
            return True
        if not bundle_available(self._cfg):
            return False
        try:
            self._ensure_loaded()
        except ServiceNotReadyError:
            return False
        return True

    @property
    def metadata(self) -> ModelMetadata:
        """The loaded model's metadata document."""
        return self._ensure_loaded().metadata

    @property
    def feature_names(self) -> list[str]:
        """Feature names in the order the model expects them."""
        return self._ensure_loaded().features

    @property
    def threshold(self) -> float:
        """The published decision threshold."""
        return self._ensure_loaded().threshold

    @property
    def model_version(self) -> str:
        """The published model version."""
        return self._ensure_loaded().version

    def status(self) -> ServiceStatus:
        """Report readiness without raising, for the ``/health`` endpoint.

        Returns:
            The current :class:`ServiceStatus`.
        """
        if not self.is_ready:
            hint = "python scripts/run_failure_pipeline.py"
            return ServiceStatus(
                model_loaded=False,
                model_version=None,
                algorithm=None,
                n_features=0,
                threshold=None,
                explainer_method=None,
                detail=self._load_error or f"Model artifacts not found. Run: {hint}",
            )
        metadata = self.metadata
        return ServiceStatus(
            runtime_warnings=check_runtime_compatibility(metadata),
            model_loaded=True,
            model_version=metadata.model_version,
            algorithm=metadata.algorithm,
            n_features=metadata.n_features,
            threshold=self.threshold,
            explainer_method=self._explainer.method if self._explainer else "not initialised",
            detail="Model artifacts loaded.",
        )

    def _get_explainer(self) -> FailureExplainer | None:
        """Build the explainer lazily; return ``None`` if it cannot be built."""
        if not bool(self._cfg.get("explainability.enabled", True)):
            return None
        if self._explainer is not None:
            return self._explainer
        bundle = self._ensure_loaded()
        background = _load_background(self._cfg, tuple(bundle.features))
        self._explainer = FailureExplainer(
            estimator=bundle.estimator,
            feature_names=bundle.features,
            background=background,
            max_background=int(self._cfg.get("explainability.background_samples", 200)),
            seed=int(self._cfg.get("random_seed", 42)),
        )
        return self._explainer

    # -- Core prediction ---------------------------------------------------
    def _build_predictions(
        self,
        features: pd.DataFrame,
        turbine_ids: list[str],
        timestamps: list[datetime],
    ) -> list[FailurePrediction]:
        """Score an aligned feature frame and assemble full prediction objects."""
        bundle = self._ensure_loaded()
        probabilities = predict_proba(bundle.estimator, features, bundle.features)
        predictions = apply_threshold(probabilities, bundle.threshold)
        low_max, medium_max = self.risk_bands
        top_k = int(self._cfg.get("explainability.top_k_factors", 5))

        attributions = self._attributions(features, top_k)
        explainer_method = self._explainer.method if self._explainer else "unavailable"

        outputs: list[FailurePrediction] = []
        for position, (turbine_id, timestamp) in enumerate(zip(turbine_ids, timestamps)):
            probability = float(probabilities[position])
            risk_level = RiskLevel(map_risk_level(probability, low_max, medium_max))
            row_attributions = attributions[position] if attributions else []
            advisory = build_advisory(row_attributions, probability, risk_level, explainer_method)
            outputs.append(
                FailurePrediction(
                    turbine_id=turbine_id,
                    timestamp=timestamp,
                    failure_probability=round(probability, 6),
                    prediction=int(predictions[position]),
                    risk_level=risk_level,
                    threshold=round(bundle.threshold, 6),
                    horizon_hours=self.horizon_hours,
                    top_risk_factors=[
                        RiskFactor(
                            feature=attribution.feature,
                            impact=round(attribution.impact, 6),
                            direction=attribution.direction,
                            value=None
                            if attribution.value is None or not np.isfinite(attribution.value)
                            else round(attribution.value, 6),
                            description=describe_factor(attribution),
                        )
                        for attribution in row_attributions
                    ],
                    explanation=advisory.explanation,
                    recommendation=advisory.recommendation,
                    model_version=bundle.version,
                )
            )
        return outputs

    def _attributions(self, features: pd.DataFrame, top_k: int) -> list[list[LocalAttribution]]:
        """Compute local attributions, degrading to empty lists on failure."""
        explainer = self._get_explainer()
        if explainer is None:
            return []
        aligned = features.loc[:, self.feature_names]
        try:
            return explainer.local_attributions(aligned, top_k=top_k)
        except (ExplainerUnavailableError, ValueError) as exc:
            # An explanation failure must not break a prediction: the narrative
            # layer says explicitly when no attribution was available.
            logger.warning("Local attribution failed", extra={"error": str(exc)})
            return []

    # -- Public prediction APIs -------------------------------------------
    def predict_from_features(
        self,
        features: dict[str, float] | pd.DataFrame,
        turbine_id: str,
        timestamp: datetime,
    ) -> FailurePrediction:
        """Predict from a precomputed feature vector matching the saved schema.

        Args:
            features: A mapping of feature name to value, or a single-row frame.
            turbine_id: Turbine identifier for the output.
            timestamp: Observation timestamp for the output.

        Returns:
            The :class:`FailurePrediction`.

        Raises:
            ServiceNotReadyError: If model artifacts are unavailable.
            FeatureContractError: If required features are missing.
        """
        frame = pd.DataFrame([features]) if isinstance(features, dict) else features.copy()
        if len(frame) != 1:
            raise FeatureContractError(f"Expected exactly one row of features, got {len(frame)}")
        aligned = frame.reindex(columns=self.feature_names)
        missing = [name for name in self.feature_names if name not in frame.columns]
        if missing:
            raise FeatureContractError(
                f"Missing {len(missing)} required feature(s) out of {len(self.feature_names)}: "
                f"{missing[:10]}" + (" ..." if len(missing) > 10 else "")
            )
        aligned = aligned.astype(np.float32)
        return self._build_predictions(aligned, [turbine_id], [timestamp])[0]

    def predict_from_window(self, window: TurbineWindow) -> FailurePrediction:
        """Predict from a window of raw observations (preferred mode).

        The window is preprocessed and passed through the production feature
        pipeline; the prediction corresponds to its **last** timestamp.

        Args:
            window: A validated :class:`TurbineWindow`.

        Returns:
            The :class:`FailurePrediction` for the final observation.

        Raises:
            ServiceNotReadyError: If model artifacts are unavailable.
            InsufficientHistoryError: If the window is shorter than the longest
                configured feature window.
        """
        self._ensure_loaded()
        frame = window.to_frame()
        required_hours = float(self._cfg.get("serving.min_history_hours", 72))
        span_hours = (
            (frame[TIMESTAMP].iloc[-1] - frame[TIMESTAMP].iloc[0]) / pd.Timedelta("1h")
            if len(frame) > 1
            else 0.0
        )
        if span_hours < required_hours:
            raise InsufficientHistoryError(
                f"Turbine {window.turbine_id}: window spans {span_hours:.1f}h but at least "
                f"{required_hours:.0f}h of history is required so the 48-hour rolling features "
                f"are defined. Supply more observations."
            )

        prepared, _ = preprocess(frame, self._cfg)
        if len(prepared) < 2:
            raise InsufficientHistoryError(
                f"Turbine {window.turbine_id}: fewer than two usable observations after cleaning."
            )
        features, _ = build_failure_features(prepared, self._cfg)
        last = features.index[-1]
        row = features.loc[[last]]
        return self._build_predictions(
            row,
            [str(prepared.loc[last, TURBINE_ID])],
            [pd.Timestamp(prepared.loc[last, TIMESTAMP]).to_pydatetime()],
        )[0]

    def predict_batch_from_windows(self, windows: list[TurbineWindow]) -> BatchFailurePrediction:
        """Predict for several turbines at once.

        Args:
            windows: One window per turbine.

        Returns:
            A :class:`BatchFailurePrediction`.

        Raises:
            ServiceNotReadyError: If model artifacts are unavailable.
            ValueError: If the batch exceeds ``serving.max_batch_turbines``.
        """
        limit = int(self._cfg.get("serving.max_batch_turbines", 200))
        if len(windows) > limit:
            raise ValueError(
                f"Batch of {len(windows)} exceeds the configured limit of {limit} turbines"
            )
        predictions = [self.predict_from_window(window) for window in windows]
        return BatchFailurePrediction(
            predictions=predictions,
            model_version=self.model_version,
            threshold=round(self.threshold, 6),
            count=len(predictions),
        )

    def predict_frame(
        self, frame: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Score a prepared multi-turbine frame, one row per observation.

        This is the bulk path used by the dashboard's fleet view, where features
        have usually already been computed by the pipeline.

        Args:
            frame: Frame carrying ``turbine_id`` and ``timestamp``.
            features: Precomputed feature matrix aligned to ``frame``; computed
                from ``frame`` when omitted.

        Returns:
            A frame with ``turbine_id``, ``timestamp``, ``failure_probability``,
            ``prediction`` and ``risk_level``.

        Raises:
            ServiceNotReadyError: If model artifacts are unavailable.
        """
        bundle = self._ensure_loaded()
        if features is None:
            prepared, _ = preprocess(frame, self._cfg)
            features, _ = build_failure_features(prepared, self._cfg)
            frame = prepared
        probabilities = predict_proba(bundle.estimator, features.loc[frame.index], bundle.features)
        low_max, medium_max = self.risk_bands
        return pd.DataFrame(
            {
                TURBINE_ID: frame[TURBINE_ID].to_numpy(),
                TIMESTAMP: pd.to_datetime(frame[TIMESTAMP]).to_numpy(),
                "failure_probability": probabilities,
                "prediction": apply_threshold(probabilities, bundle.threshold),
                "risk_level": [map_risk_level(p, low_max, medium_max) for p in probabilities],
            },
            index=frame.index,
        )

    def explain_observation(
        self, features: pd.DataFrame, top_k: int | None = None
    ) -> list[LocalAttribution]:
        """Return local attributions for a single-row feature frame.

        Args:
            features: A one-row feature frame.
            top_k: Number of factors; defaults to configuration.

        Returns:
            Ranked attributions, empty when no explainer is available.
        """
        k = top_k if top_k is not None else int(self._cfg.get("explainability.top_k_factors", 5))
        attributions = self._attributions(features, k)
        return attributions[0] if attributions else []

    def example_feature_payload(self) -> dict[str, float]:
        """Build a valid example payload for the prepared-features endpoint.

        Values come from the median of the persisted background sample when it
        is available, so the example is representative rather than all zeros.

        Returns:
            A mapping of every required feature name to a numeric value.
        """
        names = self.feature_names
        background = _load_background(self._cfg, tuple(names))
        if background is not None and not background.empty:
            medians = background[names].median()
            return {name: float(np.nan_to_num(medians[name], nan=0.0)) for name in names}
        return dict.fromkeys(names, 0.0)

    def minimum_history_hours(self) -> float:
        """Minimum window length in hours accepted by the raw-window endpoint.

        Returns:
            The configured minimum history in hours.
        """
        return float(self._cfg.get("serving.min_history_hours", 72))


# ---------------------------------------------------------------------------
# Background sample (used for explanations and example payloads)
# ---------------------------------------------------------------------------
def background_path(cfg: Config):
    """Path of the persisted SHAP background sample.

    Args:
        cfg: Merged configuration.

    Returns:
        Absolute path to the background parquet file.
    """
    from pathlib import Path

    from wind_turbine_pm.utils.paths import resolve

    return resolve(Path(str(cfg.require("paths.artifacts_models"))) / "shap_background.parquet")


@lru_cache(maxsize=4)
def _load_background(cfg: Config, feature_names: tuple[str, ...]) -> pd.DataFrame | None:
    """Load the persisted background sample, or ``None`` when it is absent."""
    path = background_path(cfg)
    if not path.is_file():
        logger.warning(
            "SHAP background sample not found; explanations may be degraded",
            extra={"path": str(path)},
        )
        return None
    frame = pd.read_parquet(path)
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        logger.warning(
            "Background sample does not match the model feature set; ignoring it",
            extra={"missing": len(missing)},
        )
        return None
    return frame[list(feature_names)]


@lru_cache(maxsize=1)
def get_service() -> FailurePredictionService:
    """Return the process-wide service instance.

    Cached so the FastAPI dependency and the dashboard share one loaded model
    rather than deserialising it per request.

    Returns:
        The shared :class:`FailurePredictionService`.
    """
    return FailurePredictionService()
