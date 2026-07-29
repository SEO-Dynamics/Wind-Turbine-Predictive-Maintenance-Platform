"""Stable column names and shared enumerations.

These constants form part of the cross-module data contract.  Downstream
modules (Turbine Health Monitoring, Anomaly Detection) are expected to import
from here rather than repeating string literals.  Renaming anything in this file
is a breaking change for the whole platform.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

# --- Key columns -----------------------------------------------------------
TURBINE_ID: Final[str] = "turbine_id"
TIMESTAMP: Final[str] = "timestamp"

# --- Sensor channels -------------------------------------------------------
WIND_SPEED: Final[str] = "wind_speed"
ROTOR_SPEED: Final[str] = "rotor_speed"
GENERATOR_SPEED: Final[str] = "generator_speed"
POWER_OUTPUT: Final[str] = "power_output"
GENERATOR_TEMPERATURE: Final[str] = "generator_temperature"
GEARBOX_TEMPERATURE: Final[str] = "gearbox_temperature"
BEARING_TEMPERATURE: Final[str] = "bearing_temperature"
OIL_TEMPERATURE: Final[str] = "oil_temperature"
OIL_PRESSURE: Final[str] = "oil_pressure"
VIBRATION: Final[str] = "vibration"
AMBIENT_TEMPERATURE: Final[str] = "ambient_temperature"
NACELLE_TEMPERATURE: Final[str] = "nacelle_temperature"
HYDRAULIC_PRESSURE: Final[str] = "hydraulic_pressure"
BRAKE_TEMPERATURE: Final[str] = "brake_temperature"

SENSOR_COLUMNS: Final[tuple[str, ...]] = (
    WIND_SPEED,
    ROTOR_SPEED,
    GENERATOR_SPEED,
    POWER_OUTPUT,
    GENERATOR_TEMPERATURE,
    GEARBOX_TEMPERATURE,
    BEARING_TEMPERATURE,
    OIL_TEMPERATURE,
    OIL_PRESSURE,
    VIBRATION,
    AMBIENT_TEMPERATURE,
    NACELLE_TEMPERATURE,
    HYDRAULIC_PRESSURE,
    BRAKE_TEMPERATURE,
)

# --- Event / state columns -------------------------------------------------
OPERATIONAL_STATUS: Final[str] = "operational_status"
FAILURE_EVENT: Final[str] = "failure_event"
MAINTENANCE_EVENT: Final[str] = "maintenance_event"

#: Columns every raw SCADA frame must provide.
REQUIRED_RAW_COLUMNS: Final[tuple[str, ...]] = (
    TIMESTAMP,
    TURBINE_ID,
    *SENSOR_COLUMNS,
    OPERATIONAL_STATUS,
    FAILURE_EVENT,
)

#: Optional columns produced by the synthetic generator. They are useful for
#: EDA and for future modules but are never used as model features.
GROUND_TRUTH_COLUMNS: Final[tuple[str, ...]] = (
    MAINTENANCE_EVENT,
    "degradation_level",
    "failure_mode",
    "episode_id",
    "hours_to_failure",
)

# --- Target ----------------------------------------------------------------
TARGET_COLUMN: Final[str] = "failure_within_48h"

# --- Split labels ----------------------------------------------------------
SPLIT_COLUMN: Final[str] = "split"


class SplitName(StrEnum):
    """Chronological split partitions."""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
    EMBARGO = "embargo"


class OperationalStatus(StrEnum):
    """Coarse turbine operating state reported by the controller."""

    NORMAL = "normal"
    IDLE = "idle"
    DERATED = "derated"
    MAINTENANCE = "maintenance"
    FAULT = "fault"


class FailureMode(StrEnum):
    """Degradation mechanisms simulated by the synthetic generator."""

    GEARBOX = "gearbox"
    GENERATOR = "generator"
    BEARING = "bearing"
    LUBRICATION = "lubrication"
    HYDRAULIC = "hydraulic"


class RiskLevel(StrEnum):
    """Risk banding applied to the failure probability."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ValidationSeverity(StrEnum):
    """Severity attached to a data-validation finding."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


#: Units for each numeric channel, part of the published data contract.
COLUMN_UNITS: Final[dict[str, str]] = {
    WIND_SPEED: "m/s",
    ROTOR_SPEED: "rpm",
    GENERATOR_SPEED: "rpm",
    POWER_OUTPUT: "kW",
    GENERATOR_TEMPERATURE: "degC",
    GEARBOX_TEMPERATURE: "degC",
    BEARING_TEMPERATURE: "degC",
    OIL_TEMPERATURE: "degC",
    OIL_PRESSURE: "bar",
    VIBRATION: "mm/s RMS",
    AMBIENT_TEMPERATURE: "degC",
    NACELLE_TEMPERATURE: "degC",
    HYDRAULIC_PRESSURE: "bar",
    BRAKE_TEMPERATURE: "degC",
}

#: Human-readable labels used by the narrative generator and the dashboard.
SENSOR_DISPLAY_NAMES: Final[dict[str, str]] = {
    WIND_SPEED: "wind speed",
    ROTOR_SPEED: "rotor speed",
    GENERATOR_SPEED: "generator speed",
    POWER_OUTPUT: "power output",
    GENERATOR_TEMPERATURE: "generator temperature",
    GEARBOX_TEMPERATURE: "gearbox temperature",
    BEARING_TEMPERATURE: "bearing temperature",
    OIL_TEMPERATURE: "oil temperature",
    OIL_PRESSURE: "oil pressure",
    VIBRATION: "vibration",
    AMBIENT_TEMPERATURE: "ambient temperature",
    NACELLE_TEMPERATURE: "nacelle temperature",
    HYDRAULIC_PRESSURE: "hydraulic pressure",
    BRAKE_TEMPERATURE: "brake temperature",
}

#: Standard disclaimer appended to every advisory recommendation.
ADVISORY_DISCLAIMER: Final[str] = (
    "Advisory output only - requires review by a qualified maintenance engineer "
    "and does not replace inspection or certified safety systems."
)
