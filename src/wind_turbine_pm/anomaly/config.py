"""Stage 3 configuration loading."""

from __future__ import annotations

from functools import lru_cache

from wind_turbine_pm.config import Config, load_config

STAGE3_MODULES = (
    "data.yaml",
    "features.yaml",
    "failure_model.yaml",
    "health_model.yaml",
    "anomaly_model.yaml",
    "maintenance.yaml",
)


def load_anomaly_config() -> Config:
    """Load the complete configuration required by Stage 3."""
    return load_config(STAGE3_MODULES)


@lru_cache(maxsize=1)
def get_anomaly_config() -> Config:
    """Return the process-wide Stage 3 configuration."""
    return load_anomaly_config()
