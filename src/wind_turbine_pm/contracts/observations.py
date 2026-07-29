"""Shared turbine observation contract.

This is a **platform-level** contract, not a failure-prediction one.  The Health
Monitoring and Anomaly Detection modules are expected to accept the same
:class:`TurbineObservation` / :class:`TurbineWindow` payloads so that a single
SCADA ingestion path can feed every module.

Changing a field name here is a breaking change across the platform.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wind_turbine_pm.constants import (
    SENSOR_COLUMNS,
    TIMESTAMP,
    TURBINE_ID,
    OperationalStatus,
)

TurbineId = Annotated[str, Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_\-]+$")]


class TurbineObservation(BaseModel):
    """One SCADA record for one turbine at one timestamp.

    Sensor fields are optional because real SCADA exports have gaps; the
    preprocessing layer is responsible for imputation.  Bounds mirror
    ``configs/data.yaml -> validation.physical_ranges`` and reject obviously
    corrupt input at the API boundary.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    turbine_id: TurbineId = Field(description="Stable turbine identifier, e.g. 'T01'.")
    timestamp: datetime = Field(description="Observation time (UTC recommended, ISO-8601).")

    wind_speed: float | None = Field(default=None, ge=0.0, le=45.0, description="m/s")
    rotor_speed: float | None = Field(default=None, ge=0.0, le=25.0, description="rpm")
    generator_speed: float | None = Field(default=None, ge=0.0, le=2200.0, description="rpm")
    power_output: float | None = Field(default=None, ge=-100.0, le=3600.0, description="kW")
    generator_temperature: float | None = Field(
        default=None, ge=-30.0, le=200.0, description="degC"
    )
    gearbox_temperature: float | None = Field(default=None, ge=-30.0, le=160.0, description="degC")
    bearing_temperature: float | None = Field(default=None, ge=-30.0, le=160.0, description="degC")
    oil_temperature: float | None = Field(default=None, ge=-30.0, le=130.0, description="degC")
    oil_pressure: float | None = Field(default=None, ge=0.0, le=12.0, description="bar")
    vibration: float | None = Field(default=None, ge=0.0, le=25.0, description="mm/s RMS")
    ambient_temperature: float | None = Field(default=None, ge=-45.0, le=55.0, description="degC")
    nacelle_temperature: float | None = Field(default=None, ge=-45.0, le=90.0, description="degC")
    hydraulic_pressure: float | None = Field(default=None, ge=0.0, le=300.0, description="bar")
    brake_temperature: float | None = Field(default=None, ge=-30.0, le=400.0, description="degC")

    operational_status: OperationalStatus = Field(
        default=OperationalStatus.NORMAL,
        description="Controller-reported operating state.",
    )

    @field_validator("turbine_id")
    @classmethod
    def _strip_turbine_id(cls, value: str) -> str:
        return value.strip()


class TurbineWindow(BaseModel):
    """A chronologically ordered history window for a single turbine.

    This is the preferred prediction input: the service computes lag, rolling
    and trend features from the window itself, so callers never have to
    reproduce the feature engineering logic.
    """

    model_config = ConfigDict(extra="forbid")

    turbine_id: TurbineId
    observations: list[TurbineObservation] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_consistency(self) -> TurbineWindow:
        """Ensure all observations belong to this turbine and are ordered."""
        mismatched = {obs.turbine_id for obs in self.observations} - {self.turbine_id}
        if mismatched:
            raise ValueError(
                f"Window declares turbine_id={self.turbine_id!r} but contains "
                f"observations for {sorted(mismatched)}"
            )
        times = [obs.timestamp for obs in self.observations]
        if any(later < earlier for earlier, later in zip(times, times[1:])):
            raise ValueError("Observations must be sorted by ascending timestamp")
        if len(set(times)) != len(times):
            raise ValueError("Window contains duplicate timestamps")
        return self

    @property
    def end_timestamp(self) -> datetime:
        """Timestamp of the most recent observation in the window."""
        return self.observations[-1].timestamp

    def to_frame(self) -> pd.DataFrame:
        """Convert the window to the canonical wide dataframe layout.

        Returns:
            A dataframe with the standard key, sensor and status columns,
            sorted by timestamp.
        """
        records = [obs.model_dump(mode="python") for obs in self.observations]
        frame = pd.DataFrame.from_records(records)
        frame[TIMESTAMP] = pd.to_datetime(frame[TIMESTAMP], errors="coerce")
        frame[TURBINE_ID] = frame[TURBINE_ID].astype(str)
        for column in SENSOR_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.NA
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["operational_status"] = frame["operational_status"].astype(str)
        return frame.sort_values(TIMESTAMP).reset_index(drop=True)


def frame_to_windows(frame: pd.DataFrame) -> list[TurbineWindow]:
    """Split a multi-turbine dataframe into per-turbine windows.

    Useful for tests and for the dashboard, which reads a processed table and
    needs to call the service with the same contract the API uses.

    Args:
        frame: Dataframe with ``turbine_id``/``timestamp`` plus sensor columns.

    Returns:
        One :class:`TurbineWindow` per turbine, each sorted by timestamp.
    """
    windows: list[TurbineWindow] = []
    columns = [TURBINE_ID, TIMESTAMP, *SENSOR_COLUMNS, "operational_status"]
    available = [column for column in columns if column in frame.columns]
    for turbine_id, group in frame[available].groupby(TURBINE_ID, sort=True):
        ordered = group.sort_values(TIMESTAMP)
        observations = [
            TurbineObservation.model_validate(
                {
                    key: (None if pd.isna(value) else value)
                    for key, value in record.items()
                    if key != "operational_status" or not pd.isna(value)
                }
            )
            for record in ordered.to_dict(orient="records")
        ]
        windows.append(TurbineWindow(turbine_id=str(turbine_id), observations=observations))
    return windows
