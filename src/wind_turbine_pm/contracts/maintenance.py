"""Unified risk and deterministic maintenance recommendation contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from wind_turbine_pm.constants import RiskLevel
from wind_turbine_pm.contracts.anomaly import AnomalyPrediction
from wind_turbine_pm.contracts.health import HealthAssessment
from wind_turbine_pm.contracts.predictions import BasePrediction, FailurePrediction

MaintenanceAction = Literal[
    "routine_monitoring",
    "plan_inspection",
    "urgent_review",
    "immediate_engineering_review",
]
DecisionConfidence = Literal["low", "medium", "high"]


class MaintenanceRecommendation(BaseModel):
    """Auditable action produced by the maintenance policy."""

    model_config = ConfigDict(extra="forbid")

    action: MaintenanceAction
    recommended_within_hours: int | None = Field(default=None, ge=1)
    target_components: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    message: str
    advisory_only: Literal[True] = True


class UnifiedRiskAssessment(BasePrediction):
    """Combined evidence without hiding the source assessments."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    unified_risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    coverage: float = Field(ge=0.0, le=1.0)
    decision_confidence: DecisionConfidence
    data_quality: float = Field(ge=0.0, le=1.0)
    missing_modules: list[str] = Field(default_factory=list)
    guardrails_triggered: list[str] = Field(default_factory=list)
    component_scores: dict[str, float] = Field(default_factory=dict)
    failure: FailurePrediction | None = None
    health: HealthAssessment | None = None
    anomaly: AnomalyPrediction | None = None
    recommendation: MaintenanceRecommendation


class BatchUnifiedRiskAssessment(BaseModel):
    """Batch maintenance response ordered from highest priority to lowest."""

    model_config = ConfigDict(extra="forbid")

    assessments: list[UnifiedRiskAssessment]
    count: int = Field(ge=0)
    policy_version: str
    advisory_only: Literal[True] = True
