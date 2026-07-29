"""Configuration loading for the Turbine Health Monitoring Module.

The module needs ``health_model.yaml`` merged on top of the shared files.  It
also keeps ``failure_model.yaml`` in the set so a single merged
:class:`~wind_turbine_pm.config.Config` object can serve both modules - the
dashboard builds one config and hands it to both services.

Every health key lives under the ``health`` namespace, so nothing here can
overwrite a Failure Prediction setting.
"""

from __future__ import annotations

from wind_turbine_pm.config import Config, get_config, load_config

#: Configuration files merged for this module, in application order.
HEALTH_MODULES: tuple[str, ...] = (
    "data.yaml",
    "features.yaml",
    "failure_model.yaml",
    "health_model.yaml",
)


def load_health_config() -> Config:
    """Load a freshly merged configuration including the health module.

    Returns:
        The merged :class:`~wind_turbine_pm.config.Config`.

    Raises:
        wind_turbine_pm.config.ConfigError: If a file is missing or malformed.
    """
    return load_config(HEALTH_MODULES)


def get_health_config() -> Config:
    """Return the process-wide merged configuration including health keys.

    Cached by :func:`wind_turbine_pm.config.get_config`, so repeated calls in a
    request-serving process do not re-read YAML.

    Returns:
        The shared merged :class:`~wind_turbine_pm.config.Config`.
    """
    return get_config(HEALTH_MODULES)


def has_health_config(cfg: Config) -> bool:
    """Report whether a configuration carries the health namespace.

    Used by components that accept an arbitrary config and must fail with a
    readable message rather than a ``KeyError`` deep inside a feature builder.

    Args:
        cfg: Any merged configuration.

    Returns:
        ``True`` when ``health`` keys are present.
    """
    return cfg.get("health.model.name") is not None
