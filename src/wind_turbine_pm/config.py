"""YAML configuration loading with deep merging and attribute-style access.

Configuration is layered: ``base.yaml`` is always loaded first, then each
requested module file is deep-merged on top of it.  The result is wrapped in a
small read-only mapping that supports both ``cfg["a"]["b"]`` and ``cfg.a.b``
access, plus dotted lookup via :meth:`Config.get`.

Nothing in this module triggers training, IO against the data directories, or
any other side effect at import time.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Iterator, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from wind_turbine_pm.utils.paths import project_root, resolve

DEFAULT_CONFIG_DIR = "configs"
BASE_CONFIG = "base.yaml"
DEFAULT_MODULES: tuple[str, ...] = ("data.yaml", "features.yaml", "failure_model.yaml")

_ENV_CONFIG_DIR = "WTPM_CONFIG_DIR"


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded or is internally invalid."""


class Config(Mapping[str, Any]):
    """Immutable, nested, attribute-accessible configuration mapping."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        """Wrap a nested mapping.

        Args:
            data: Parsed configuration mapping.
        """
        object.__setattr__(self, "_data", dict(data))

    # -- Mapping protocol ---------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        return Config(value) if isinstance(value, dict) else value

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - attribute error path
            raise AttributeError(f"No configuration key {name!r}") from exc

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Config({sorted(self._data)})"

    # `collections.abc.Mapping` sets `__hash__ = None`, which would make a
    # Config unusable as an `lru_cache` key. Configs are treated as immutable
    # value objects held for the lifetime of a process, so identity hashing is
    # both safe and what callers expect.
    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Config):
            return self._data == other._data
        return NotImplemented

    # -- Helpers ------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """Look up a possibly dotted key.

        Args:
            key: Key or dotted path such as ``"training.candidates.dummy"``.
            default: Value returned when the path does not exist.

        Returns:
            The configured value, or ``default``.
        """
        node: Any = self._data
        for part in key.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return Config(node) if isinstance(node, dict) else node

    def require(self, key: str) -> Any:
        """Look up a dotted key, raising when it is absent.

        Args:
            key: Dotted configuration path.

        Returns:
            The configured value.

        Raises:
            ConfigError: If the key is missing.
        """
        sentinel = object()
        value = self.get(key, sentinel)
        if value is sentinel:
            raise ConfigError(f"Missing required configuration key: {key!r}")
        return value

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy of the underlying plain dictionary."""
        return copy.deepcopy(self._data)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either.

    Args:
        base: Lower-priority mapping.
        override: Higher-priority mapping.

    Returns:
        A new merged dictionary.
    """
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def config_dir() -> Path:
    """Return the directory configuration files are read from."""
    override = os.environ.get(_ENV_CONFIG_DIR)
    return resolve(override) if override else project_root() / DEFAULT_CONFIG_DIR


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"Configuration file {path} must contain a mapping at the top level")
    return loaded


def load_config(modules: tuple[str, ...] | None = None) -> Config:
    """Load ``base.yaml`` plus the requested module configuration files.

    Args:
        modules: File names inside the configuration directory to merge on top
            of the base file.  Defaults to data, features and failure-model
            configuration.

    Returns:
        The merged :class:`Config`.

    Raises:
        ConfigError: If a file is missing or malformed.
    """
    names = DEFAULT_MODULES if modules is None else modules
    directory = config_dir()
    merged = _read_yaml(directory / BASE_CONFIG)
    for name in names:
        merged = deep_merge(merged, _read_yaml(directory / name))
    validate_config(merged)
    return Config(merged)


@lru_cache(maxsize=8)
def get_config(modules: tuple[str, ...] | None = None) -> Config:
    """Cached variant of :func:`load_config` for use across a process.

    Args:
        modules: See :func:`load_config`.

    Returns:
        The merged :class:`Config` (shared instance).
    """
    return load_config(modules)


def validate_config(data: Mapping[str, Any]) -> None:
    """Check cross-cutting configuration invariants.

    Args:
        data: The merged raw configuration mapping.

    Raises:
        ConfigError: If an invariant is violated.
    """
    risk = data.get("risk_levels")
    if isinstance(risk, Mapping):
        low, medium = float(risk["low_max"]), float(risk["medium_max"])
        if not 0.0 < low < medium < 1.0:
            raise ConfigError(
                f"risk_levels must satisfy 0 < low_max < medium_max < 1, got {low} and {medium}"
            )

    split = data.get("split")
    if isinstance(split, Mapping):
        train_end = float(split["train_end_fraction"])
        valid_end = float(split["valid_end_fraction"])
        if not 0.0 < train_end < valid_end < 1.0:
            raise ConfigError(
                "split fractions must satisfy 0 < train_end_fraction < valid_end_fraction < 1, "
                f"got {train_end} and {valid_end}"
            )
        if float(split.get("embargo_hours", 0)) < 0:
            raise ConfigError("split.embargo_hours must be non-negative")

    target = data.get("target")
    if isinstance(target, Mapping) and float(target.get("horizon_hours", 0)) <= 0:
        raise ConfigError("target.horizon_hours must be positive")

    threshold = data.get("threshold")
    if isinstance(threshold, Mapping):
        method = threshold.get("method")
        if method not in {"f2", "cost"}:
            raise ConfigError(f"threshold.method must be 'f2' or 'cost', got {method!r}")
