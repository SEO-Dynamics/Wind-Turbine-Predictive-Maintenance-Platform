"""Anomaly Detection output and artifact contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from wind_turbine_pm.constants import OperatingRegime
from wind_turbine_pm.contracts.predictions import BasePrediction, RiskFactor

AnomalySeverity = Literal["normal", "warning", "alarm", "not_applicable"]
AssessmentState = Literal["scored", "not_applicable"]


class AnomalyPrediction(BasePrediction):
    """One novelty assessment at the latest point in a turbine window."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    assessment_state: AssessmentState = "scored"
    raw_anomaly_score: float | None = None
    anomaly_score: float | None = Field(default=None, ge=0.0, le=1.0)
    is_anomaly: bool | None = None
    severity: AnomalySeverity = "normal"
    warning_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    alarm_threshold: float = Field(default=0.99, ge=0.0, le=1.0)
    operating_regime: OperatingRegime
    data_quality: float = Field(default=1.0, ge=0.0, le=1.0)
    contributing_signals: list[RiskFactor] = Field(default_factory=list)
    explanation: str = ""


class BatchAnomalyPrediction(BaseModel):
    """Batch anomaly response."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    predictions: list[AnomalyPrediction]
    model_version: str
    count: int = Field(ge=0)
    advisory_only: Literal[True] = True


class AnomalyCalibration(BaseModel):
    """Empirical mapping from raw novelty scores to comparable percentiles."""

    model_config = ConfigDict(extra="forbid")

    reference_scores: list[float]
    warning_raw: float
    alarm_raw: float
    warning_percentile: float = 0.95
    alarm_percentile: float = 0.99
    achieved_warning_rate: float = Field(ge=0.0, le=1.0)
    achieved_alarm_rate: float = Field(ge=0.0, le=1.0)


class AnomalyModelMetadata(BaseModel):
    """Validated metadata stored beside the anomaly estimator."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str = "anomaly_detection"
    model_version: str
    algorithm: str
    training_date: datetime
    features: list[str]
    feature_groups: dict[str, list[str]] = Field(default_factory=dict)
    metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    selection_rationale: str
    dataset: dict[str, Any]
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    library_versions: dict[str, str] = Field(default_factory=dict)
    advisory_only: Literal[True] = True

    @property
    def n_features(self) -> int:
        return len(self.features)
