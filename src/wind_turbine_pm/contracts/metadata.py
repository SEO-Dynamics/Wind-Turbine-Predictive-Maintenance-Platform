"""Model metadata contract.

Every model published on the platform - failure prediction today, health
scoring and anomaly detection later - serialises a :class:`ModelMetadata`
document next to its artifact.  The service layer validates feature order
against this document before scoring, which is the main guard against a stale
model being served with a newer feature pipeline.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetSummary(BaseModel):
    """Provenance and shape of the dataset a model was trained on."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(description="'synthetic' or an identifier for a real SCADA export.")
    is_synthetic: bool = Field(description="True when the data was simulated.")
    n_turbines: int = Field(ge=0)
    n_rows: int = Field(ge=0)
    n_features: int = Field(ge=0)
    time_start: datetime | None = None
    time_end: datetime | None = None
    positive_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    rows_train: int = Field(default=0, ge=0)
    rows_valid: int = Field(default=0, ge=0)
    rows_test: int = Field(default=0, ge=0)


class ThresholdInfo(BaseModel):
    """The decision threshold and how it was chosen."""

    model_config = ConfigDict(extra="forbid")

    value: float = Field(ge=0.0, le=1.0)
    method: str = Field(description="'f2' or 'cost'.")
    selected_on: str = Field(
        default="valid", description="Split used for selection - never 'test'."
    )
    costs: dict[str, float] = Field(default_factory=dict, description="Illustrative cost matrix.")
    validation_metrics: dict[str, float] = Field(default_factory=dict)
    alternatives: dict[str, float] = Field(
        default_factory=dict, description="Thresholds produced by the other methods."
    )


class RiskLevelBands(BaseModel):
    """Probability cut-points for the low/medium/high banding."""

    model_config = ConfigDict(extra="forbid")

    low_max: float = Field(gt=0.0, lt=1.0)
    medium_max: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _check_order(self) -> RiskLevelBands:
        if self.low_max >= self.medium_max:
            raise ValueError(
                f"low_max ({self.low_max}) must be strictly below medium_max ({self.medium_max})"
            )
        return self


class ModelMetadata(BaseModel):
    """Full description of a trained, published model."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str
    model_version: str
    algorithm: str = Field(description="Concrete estimator class that was selected.")
    module: str = Field(default="failure_prediction")
    training_date: datetime
    target: str
    horizon_hours: int = Field(gt=0)

    features: list[str] = Field(description="Feature names in the exact order the model expects.")
    feature_groups: dict[str, list[str]] = Field(default_factory=dict)

    threshold: ThresholdInfo
    risk_levels: RiskLevelBands
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
