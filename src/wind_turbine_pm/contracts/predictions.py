"""Prediction output contracts.

:class:`RiskFactor` and :class:`BasePrediction` are deliberately generic so the
Health Monitoring and Anomaly Detection modules can subclass them and stay
wire-compatible with the dashboard's rendering components.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from wind_turbine_pm.constants import RiskLevel

#: Direction of a feature's contribution to the predicted risk.
RiskDirection = Literal["increases_risk", "decreases_risk"]


class RiskFactor(BaseModel):
    """A single feature's contribution to a prediction."""

    model_config = ConfigDict(extra="forbid")

    feature: str = Field(description="Model feature name.")
    impact: float = Field(
        description="Signed contribution in the explainer's units (SHAP log-odds)."
    )
    direction: RiskDirection = Field(description="Whether the feature pushed risk up or down.")
    value: float | None = Field(default=None, description="Observed feature value, when available.")
    description: str | None = Field(
        default=None, description="Human-readable phrasing of the factor."
    )


class BasePrediction(BaseModel):
    """Fields every per-turbine model output on the platform must carry."""

    model_config = ConfigDict(extra="forbid")

    turbine_id: str
    timestamp: datetime
    model_version: str
    advisory_only: Literal[True] = Field(
        default=True,
        description="Always true: outputs are decision support, never automated control.",
    )


class FailurePrediction(BasePrediction):
    """Failure Prediction Module output for one turbine at one timestamp."""

    failure_probability: float = Field(ge=0.0, le=1.0, description="P(failure within the horizon).")
    prediction: Literal[0, 1] = Field(description="Binary decision at the optimised threshold.")
    risk_level: RiskLevel = Field(description="Configured probability band.")
    threshold: float = Field(ge=0.0, le=1.0, description="Decision threshold applied.")
    horizon_hours: int = Field(gt=0, description="Prediction horizon in hours.")
    top_risk_factors: list[RiskFactor] = Field(
        default_factory=list, description="Highest-magnitude contributions, most important first."
    )
    explanation: str = Field(default="", description="Human-readable narrative of the drivers.")
    recommendation: str = Field(default="", description="Advisory maintenance message.")


class BatchFailurePrediction(BaseModel):
    """Container for a batch of failure predictions."""

    model_config = ConfigDict(extra="forbid")

    predictions: list[FailurePrediction]
    model_version: str
    threshold: float
    count: int = Field(ge=0)
    advisory_only: Literal[True] = True
