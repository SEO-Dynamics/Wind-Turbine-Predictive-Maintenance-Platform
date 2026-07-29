"""Turbine Health Monitoring output contracts.

These are **additive** platform contracts: nothing here changes the existing
observation, prediction or metadata schemas.  :class:`HealthAssessment`
subclasses :class:`~wind_turbine_pm.contracts.predictions.BasePrediction`, so
anything that can render a failure prediction can render a health assessment -
the shared fields (``turbine_id``, ``timestamp``, ``model_version``,
``advisory_only``) mean the same thing in both.

Input is the shared :class:`~wind_turbine_pm.contracts.observations.TurbineWindow`;
the health module deliberately does not define its own ingestion schema.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wind_turbine_pm.constants import (
    DriftDirection,
    DriftSeverity,
    HealthClass,
    OperatingRegime,
)
from wind_turbine_pm.contracts.metadata import DatasetSummary
from wind_turbine_pm.contracts.predictions import BasePrediction, RiskFactor


class SensorRuleViolation(BaseModel):
    """One sensor breaching its configured operating envelope."""

    model_config = ConfigDict(extra="forbid")

    sensor: str = Field(description="Sensor channel name.")
    rule: str = Field(description="Which check fired, e.g. 'alarm_above' or 'stuck'.")
    severity: DriftSeverity = Field(description="'warning' or 'alarm'.")
    observed: float | None = Field(default=None, description="Value that triggered the rule.")
    limit: float | None = Field(default=None, description="Configured limit that was crossed.")
    fraction_of_window: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Share of the assessed window in violation."
    )
    description: str = Field(default="", description="Human-readable phrasing.")


class SensorDriftSignal(BaseModel):
    """Drift verdict for one sensor of one turbine.

    ``statistic`` and ``control_limit`` are reported in the detector's own
    units (sigma for CUSUM, standard errors for EWMA) so an engineer can see
    *how far* past the limit the signal is rather than only that it fired.
    """

    model_config = ConfigDict(extra="forbid")

    sensor: str
    method: Literal["cusum", "ewma", "isolation_forest"]
    detected: bool
    severity: DriftSeverity = DriftSeverity.NONE
    direction: DriftDirection = DriftDirection.NONE
    statistic: float = Field(description="Detector statistic at the assessed timestamp.")
    control_limit: float = Field(gt=0.0, description="Limit the statistic is compared against.")
    first_detected: datetime | None = Field(
        default=None, description="Earliest timestamp in the window at which the limit was crossed."
    )
    description: str = ""


class ComponentHealth(BaseModel):
    """Health roll-up for one turbine component."""

    model_config = ConfigDict(extra="forbid")

    component: str = Field(description="Component key, e.g. 'gearbox'.")
    score: float = Field(ge=0.0, le=100.0, description="0-100, higher is healthier.")
    health_class: HealthClass
    sensors: list[str] = Field(default_factory=list, description="Sensors backing this score.")
    drivers: list[str] = Field(
        default_factory=list, description="Short phrases explaining the deduction."
    )


class HealthAssessment(BasePrediction):
    """Turbine Health Monitoring output for one turbine at one timestamp.

    ``health_score`` is the published number: the model's estimate of condition
    (``raw_health_score``) after the drift penalty is applied.  Both are
    reported so the deduction is always auditable.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    health_score: float = Field(ge=0.0, le=100.0, description="Published 0-100 health score.")
    health_class: HealthClass = Field(description="Band the published score falls into.")
    raw_health_score: float = Field(
        ge=0.0, le=100.0, description="Model output before the drift penalty."
    )
    drift_penalty: float = Field(
        default=0.0, ge=0.0, description="Points deducted for detected sensor drift."
    )
    operating_regime: OperatingRegime = Field(description="Operating point of the assessment.")
    data_quality: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Share of the assessed window that passed the sensor validation rules.",
    )
    component_health: list[ComponentHealth] = Field(default_factory=list)
    drift_signals: list[SensorDriftSignal] = Field(
        default_factory=list, description="Only signals that fired; empty means no drift detected."
    )
    rule_violations: list[SensorRuleViolation] = Field(default_factory=list)
    top_factors: list[RiskFactor] = Field(
        default_factory=list,
        description="Largest condition deviations, most important first.",
    )
    explanation: str = Field(default="", description="Human-readable summary of the assessment.")
    recommendation: str = Field(default="", description="Advisory maintenance message.")

    @model_validator(mode="after")
    def _check_penalty_consistency(self) -> HealthAssessment:
        """Guarantee the published score is the raw score minus the penalty."""
        expected = max(self.raw_health_score - self.drift_penalty, 0.0)
        if abs(expected - self.health_score) > 1e-6:
            raise ValueError(
                f"health_score ({self.health_score}) must equal raw_health_score "
                f"({self.raw_health_score}) minus drift_penalty ({self.drift_penalty}), "
                f"floored at zero"
            )
        return self


class BatchHealthAssessment(BaseModel):
    """Container for a batch of health assessments."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    assessments: list[HealthAssessment]
    model_version: str
    count: int = Field(ge=0)
    advisory_only: Literal[True] = True


class FleetHealthSummary(BaseModel):
    """Fleet-level roll-up of the most recent assessment per turbine."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    as_of: datetime | None = Field(default=None, description="Cut-off the snapshot describes.")
    n_turbines: int = Field(ge=0)
    mean_health_score: float = Field(ge=0.0, le=100.0)
    min_health_score: float = Field(ge=0.0, le=100.0)
    class_counts: dict[str, int] = Field(
        default_factory=dict, description="Turbine count per health class."
    )
    regime_counts: dict[str, int] = Field(
        default_factory=dict, description="Turbine count per operating regime."
    )
    n_drift_alerts: int = Field(default=0, ge=0)
    worst_turbines: list[dict[str, Any]] = Field(
        default_factory=list, description="Lowest-scoring turbines, worst first."
    )
    model_version: str = ""
    advisory_only: Literal[True] = True


class HealthClassBands(BaseModel):
    """Score cut-points for the four health classes."""

    model_config = ConfigDict(extra="forbid")

    healthy_min: float = Field(gt=0.0, le=100.0)
    monitor_min: float = Field(gt=0.0, le=100.0)
    degraded_min: float = Field(ge=0.0, le=100.0)

    @model_validator(mode="after")
    def _check_order(self) -> HealthClassBands:
        if not self.degraded_min < self.monitor_min < self.healthy_min:
            raise ValueError(
                "Health class bands must satisfy degraded_min < monitor_min < healthy_min, got "
                f"{self.degraded_min}, {self.monitor_min}, {self.healthy_min}"
            )
        return self


class HealthModelMetadata(BaseModel):
    """Full description of a trained, published health-score model.

    Deliberately a sibling of
    :class:`~wind_turbine_pm.contracts.metadata.ModelMetadata` rather than a
    reuse of it: that contract is built around a binary classifier with a
    decision threshold and probability risk bands, neither of which applies to a
    0-100 regression score.  Provenance is shared - the same
    :class:`DatasetSummary` records what the model was trained on - so the two
    artifacts stay comparable.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str
    model_version: str
    algorithm: str = Field(description="Concrete estimator class that was selected.")
    module: str = Field(default="turbine_health_monitoring")
    training_date: datetime
    target: str = Field(description="Name of the regression target.")
    target_source: str = Field(description="Column the ground-truth health state came from.")

    features: list[str] = Field(description="Feature names in the exact order the model expects.")
    feature_groups: dict[str, list[str]] = Field(default_factory=dict)

    health_classes: HealthClassBands
    drift: dict[str, Any] = Field(
        default_factory=dict, description="Drift detector configuration in force at training time."
    )
    metrics: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Metrics keyed by split name."
    )
    dataset: DatasetSummary
    selection_rationale: str = Field(default="")
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    library_versions: dict[str, str] = Field(default_factory=dict)
    advisory_only: bool = True

    @property
    def n_features(self) -> int:
        """Number of features the model consumes."""
        return len(self.features)
