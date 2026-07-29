"""Request and response schemas for the HTTP API.

Wire schemas live here; the domain contracts they build on live in
:mod:`wind_turbine_pm.contracts`.  Keeping them separate means a future change
to the HTTP surface (pagination, envelopes, versioned payloads) does not force a
change to the shared platform contracts that other modules depend on.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from wind_turbine_pm.contracts.observations import TurbineObservation, TurbineWindow
from wind_turbine_pm.contracts.predictions import FailurePrediction


class HealthResponse(BaseModel):
    """Response of ``GET /health``."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    status: Literal["ok", "degraded"] = Field(
        description=(
            "'ok' when every mounted module's artifacts are loaded, 'degraded' when at least "
            "one is not. See `modules` for which."
        )
    )
    api_version: str
    model_loaded: bool = Field(
        description=(
            "Whether the Failure Prediction model is loaded. Retained as a stable top-level "
            "field for existing probes; per-module status lives in `modules`."
        )
    )
    model_version: str | None = None
    model_artifact_status: str = Field(description="Human-readable artifact status summary.")
    timestamp: datetime
    detail: str = ""
    modules: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Artifact status per mounted module, keyed by module name. Each module is "
            "independent: one having no artifacts does not stop the others from serving."
        ),
    )
    runtime_warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Non-empty when the runtime's library versions differ from those that "
            "produced a model artifact; output may be unreliable."
        ),
    )


class ModelInfoResponse(BaseModel):
    """Response of ``GET /failure/model-info``."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str
    model_version: str
    algorithm: str
    training_date: datetime
    target: str
    horizon_hours: int
    n_features: int
    threshold: float
    threshold_method: str
    risk_levels: dict[str, float]
    dataset: dict[str, Any]
    is_synthetic: bool
    selection_rationale: str
    library_versions: dict[str, str]
    advisory_only: Literal[True] = True


class FeatureListResponse(BaseModel):
    """Response of ``GET /failure/features``."""

    model_config = ConfigDict(extra="forbid")

    n_features: int
    features: list[str] = Field(description="Feature names in the exact order the model expects.")
    feature_groups: dict[str, list[str]]
    note: str = Field(
        default=(
            "Order matters. When posting to /failure/predict/prepared, every listed feature "
            "must be present; extra keys are ignored."
        )
    )


class MetricsResponse(BaseModel):
    """Response of ``GET /failure/metrics``."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_version: str
    threshold: float
    metrics: dict[str, Any]
    is_synthetic: bool
    disclaimer: str = Field(
        default=(
            "Metrics are computed on synthetic data unless stated otherwise and do not "
            "represent validated real-world performance."
        )
    )


class ExampleResponse(BaseModel):
    """Response of ``GET /failure/example``."""

    model_config = ConfigDict(extra="forbid")

    window_request: dict[str, Any] = Field(description="A ready-to-post /failure/predict body.")
    prepared_request: dict[str, Any] = Field(
        description="A ready-to-post /failure/predict/prepared body."
    )
    min_history_hours: float


class WindowPredictRequest(BaseModel):
    """Request body of ``POST /failure/predict`` (preferred, raw-history mode)."""

    model_config = ConfigDict(extra="forbid")

    turbine_id: str = Field(min_length=1, max_length=32)
    observations: list[TurbineObservation] = Field(
        min_length=2,
        description=(
            "Chronologically ordered recent history for one turbine. The prediction is made for "
            "the last observation. At least `min_history_hours` of history is required."
        ),
    )

    def to_window(self) -> TurbineWindow:
        """Convert to the shared :class:`TurbineWindow` contract.

        Returns:
            The validated window.
        """
        return TurbineWindow(turbine_id=self.turbine_id, observations=self.observations)


class BatchPredictRequest(BaseModel):
    """Request body of ``POST /failure/predict/batch``."""

    model_config = ConfigDict(extra="forbid")

    windows: list[WindowPredictRequest] = Field(min_length=1, max_length=200)


class PreparedPredictRequest(BaseModel):
    """Request body of ``POST /failure/predict/prepared`` (fallback mode)."""

    model_config = ConfigDict(extra="forbid")

    turbine_id: str = Field(min_length=1, max_length=32)
    timestamp: datetime
    features: dict[str, float] = Field(
        min_length=1,
        description="Precomputed features matching the schema from GET /failure/features.",
    )


class BatchPredictResponse(BaseModel):
    """Response of ``POST /failure/predict/batch``."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    predictions: list[FailurePrediction]
    count: int
    model_version: str
    threshold: float
    advisory_only: Literal[True] = True


# ---------------------------------------------------------------------------
# Turbine Health Monitoring
# ---------------------------------------------------------------------------
class HealthWindowRequest(BaseModel):
    """Request body of ``POST /health-monitoring/assess``.

    Deliberately the same shape as :class:`WindowPredictRequest`: a caller that
    can already post a window to the Failure Prediction Module can post the
    identical body here, which is the point of the shared observation contract.
    """

    model_config = ConfigDict(extra="forbid")

    turbine_id: str = Field(min_length=1, max_length=32)
    observations: list[TurbineObservation] = Field(
        min_length=2,
        description=(
            "Chronologically ordered recent history for one turbine. The assessment is made for "
            "the last observation. At least `min_history_hours` of history is required so the "
            "trailing condition indicators and the drift baselines are defined."
        ),
    )

    def to_window(self) -> TurbineWindow:
        """Convert to the shared :class:`TurbineWindow` contract.

        Returns:
            The validated window.
        """
        return TurbineWindow(turbine_id=self.turbine_id, observations=self.observations)


class BatchHealthRequest(BaseModel):
    """Request body of ``POST /health-monitoring/assess/batch``."""

    model_config = ConfigDict(extra="forbid")

    windows: list[HealthWindowRequest] = Field(min_length=1, max_length=200)


class HealthModelInfoResponse(BaseModel):
    """Response of ``GET /health-monitoring/model-info``."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str
    model_version: str
    algorithm: str
    training_date: datetime
    target: str
    target_source: str = Field(
        description="Column the ground-truth condition came from. On a real fleet this must be "
        "independent of the SCADA channels the features are built from."
    )
    n_features: int
    health_classes: dict[str, float] = Field(description="Score cut-points for the four bands.")
    drift: dict[str, Any] = Field(description="Drift detector settings and fitted calibration.")
    dataset: dict[str, Any]
    is_synthetic: bool
    selection_rationale: str
    library_versions: dict[str, str]
    advisory_only: Literal[True] = True


class HealthMetricsResponse(BaseModel):
    """Response of ``GET /health-monitoring/metrics``."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_version: str
    metrics: dict[str, Any]
    is_synthetic: bool
    disclaimer: str = Field(
        default=(
            "Metrics are computed on synthetic data unless stated otherwise and do not "
            "represent validated real-world performance. The health target is derived from the "
            "simulator's degradation state, not from inspection outcomes."
        )
    )


class SensorRulesResponse(BaseModel):
    """Response of ``GET /health-monitoring/sensor-rules``."""

    model_config = ConfigDict(extra="forbid")

    n_rules: int
    rules: list[dict[str, Any]] = Field(
        description="One record per sensor rule, including its provenance and rationale."
    )
    note: str = Field(
        default=(
            "Hard limits mirror the platform's physical ranges (a reading outside them is an "
            "instrument fault). Warning and alarm limits are operating-envelope limits: values "
            "that are physically possible but indicate operation outside the healthy envelope. "
            "These are advisory thresholds for a synthetic fleet and must be re-derived from the "
            "operator's own turbine type and alarm history before operational use."
        )
    )


class HealthExampleResponse(BaseModel):
    """Response of ``GET /health-monitoring/example``."""

    model_config = ConfigDict(extra="forbid")

    window_request: dict[str, Any] = Field(
        description="A ready-to-post /health-monitoring/assess body."
    )
    min_history_hours: float


class ErrorResponse(BaseModel):
    """Structured error body returned for every handled failure."""

    model_config = ConfigDict(extra="forbid")

    error: str = Field(description="Machine-readable error code.")
    detail: str = Field(description="Human-readable description.")
    hint: str | None = Field(default=None, description="Command or action that resolves the error.")
