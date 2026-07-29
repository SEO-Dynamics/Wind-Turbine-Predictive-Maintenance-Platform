"""Saving and loading the published health-monitoring bundle.

Follows the platform convention set by
:mod:`wind_turbine_pm.models.persistence`: the fitted estimator and its metadata
are separate files, so the metadata stays readable and diffable without
unpickling anything, and loading validates the metadata against its Pydantic
contract so a corrupted artifact fails loudly instead of scoring silently
wrong.

Every artifact name is prefixed ``health_``, as the handoff contract requires,
so nothing here can overwrite a ``failure_*`` artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wind_turbine_pm.config import Config
from wind_turbine_pm.contracts.health import HealthModelMetadata
from wind_turbine_pm.health.drift import MultivariateDriftDetector
from wind_turbine_pm.logging_config import get_logger
from wind_turbine_pm.models.persistence import check_runtime_compatibility
from wind_turbine_pm.utils.io import (
    ArtifactNotFoundError,
    read_json,
    read_pickle,
    write_json,
    write_pickle,
)
from wind_turbine_pm.utils.paths import resolve

logger = get_logger(__name__)

#: Command shown to users when a health artifact is missing.
HEALTH_TRAIN_HINT = "python scripts/run_health_pipeline.py   (or: make health-pipeline)"


@dataclass(frozen=True)
class HealthBundle:
    """A fitted health model together with its validated metadata."""

    estimator: Any
    metadata: HealthModelMetadata
    drift_detector: MultivariateDriftDetector | None = None

    @property
    def features(self) -> list[str]:
        """Feature names in the order the estimator expects."""
        return list(self.metadata.features)

    @property
    def version(self) -> str:
        """The published model version."""
        return self.metadata.model_version


def _artifact(cfg: Config, area: str, key: str) -> Path:
    """Resolve a health artifact path from configuration.

    Args:
        cfg: Merged configuration.
        area: ``paths`` key, e.g. ``artifacts_models``.
        key: ``health.model`` key holding the file name.

    Returns:
        The absolute artifact path.
    """
    return resolve(
        Path(str(cfg.require(f"paths.{area}"))) / str(cfg.require(f"health.model.{key}"))
    )


def health_model_path(cfg: Config) -> Path:
    """Path of the serialised health estimator."""
    return _artifact(cfg, "artifacts_models", "artifact_filename")


def health_drift_model_path(cfg: Config) -> Path:
    """Path of the serialised multivariate drift detector."""
    return _artifact(cfg, "artifacts_models", "drift_model_filename")


def health_background_path(cfg: Config) -> Path:
    """Path of the persisted feature background sample."""
    return _artifact(cfg, "artifacts_models", "background_filename")


def health_metadata_path(cfg: Config) -> Path:
    """Path of the health model metadata document."""
    return _artifact(cfg, "artifacts_metadata", "metadata_filename")


def health_metrics_path(cfg: Config) -> Path:
    """Path of the health metrics document."""
    return _artifact(cfg, "artifacts_metrics", "metrics_filename")


def health_comparison_path(cfg: Config) -> Path:
    """Path of the health candidate-comparison CSV."""
    return _artifact(cfg, "artifacts_metrics", "comparison_filename")


def health_dataset_path(cfg: Config) -> Path:
    """Path of the prepared health dataset."""
    return _artifact(cfg, "data_processed", "dataset_filename")


def health_features_path(cfg: Config) -> Path:
    """Path of the prepared health feature matrix."""
    return _artifact(cfg, "data_processed", "features_filename")


def health_spec_path(cfg: Config) -> Path:
    """Path of the health feature specification."""
    return _artifact(cfg, "data_processed", "spec_filename")


def save_health_bundle(
    estimator: Any,
    metadata: HealthModelMetadata,
    cfg: Config,
    drift_detector: MultivariateDriftDetector | None = None,
) -> tuple[Path, Path]:
    """Persist a health estimator, its metadata and its drift detector.

    Args:
        estimator: The fitted pipeline.
        metadata: Its validated metadata document.
        cfg: Merged configuration.
        drift_detector: Fitted multivariate detector, when one was built.

    Returns:
        ``(estimator_path, metadata_path)``.
    """
    estimator_file = write_pickle(estimator, health_model_path(cfg))
    metadata_file = write_json(metadata.model_dump(mode="json"), health_metadata_path(cfg))
    if drift_detector is not None:
        write_pickle(drift_detector, health_drift_model_path(cfg))
    logger.info(
        "Saved health bundle",
        extra={
            "estimator": str(estimator_file),
            "metadata": str(metadata_file),
            "version": metadata.model_version,
            "drift_detector": drift_detector is not None,
        },
    )
    return estimator_file, metadata_file


def load_health_bundle(cfg: Config) -> HealthBundle:
    """Load the published health bundle.

    Args:
        cfg: Merged configuration.

    Returns:
        The :class:`HealthBundle`.

    Raises:
        ArtifactNotFoundError: If the estimator or metadata is missing, carrying
            the command that regenerates them.
        pydantic.ValidationError: If the metadata does not satisfy its contract.
    """
    estimator = read_pickle(health_model_path(cfg), hint=HEALTH_TRAIN_HINT)
    metadata = HealthModelMetadata.model_validate(
        read_json(health_metadata_path(cfg), hint=HEALTH_TRAIN_HINT)
    )

    # `check_runtime_compatibility` only reads `.library_versions`, which both
    # metadata contracts carry; reusing it keeps one definition of "which
    # libraries can break a pickled estimator" for the whole platform.
    for message in check_runtime_compatibility(metadata):  # type: ignore[arg-type]
        logger.warning(
            "Runtime/artifact version mismatch: %s. Health scores may be unreliable - rebuild "
            "the artifacts in this environment with: python scripts/run_health_pipeline.py --force",
            message,
        )

    detector: MultivariateDriftDetector | None = None
    drift_path = health_drift_model_path(cfg)
    if drift_path.is_file():
        detector = read_pickle(drift_path, hint=HEALTH_TRAIN_HINT)

    logger.info(
        "Loaded health bundle",
        extra={
            "version": metadata.model_version,
            "features": metadata.n_features,
            "drift_detector": detector is not None,
        },
    )
    return HealthBundle(estimator=estimator, metadata=metadata, drift_detector=detector)


def health_bundle_available(cfg: Config) -> bool:
    """Return whether both required health artifacts are present.

    Args:
        cfg: Merged configuration.

    Returns:
        ``True`` when the estimator and metadata files both exist.
    """
    return health_model_path(cfg).is_file() and health_metadata_path(cfg).is_file()


def load_health_metrics(cfg: Config) -> dict[str, Any]:
    """Load the health metrics document.

    Args:
        cfg: Merged configuration.

    Returns:
        The metrics document.

    Raises:
        ArtifactNotFoundError: If the file is missing.
    """
    return read_json(health_metrics_path(cfg), hint=HEALTH_TRAIN_HINT)


def try_load_health_metrics(cfg: Config) -> dict[str, Any] | None:
    """Load health metrics, returning ``None`` instead of raising when absent.

    Args:
        cfg: Merged configuration.

    Returns:
        The metrics document, or ``None``.
    """
    try:
        return load_health_metrics(cfg)
    except ArtifactNotFoundError:
        return None
