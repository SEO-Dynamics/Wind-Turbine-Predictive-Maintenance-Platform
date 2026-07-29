"""Saving and loading the published model bundle.

A "bundle" is the fitted pipeline plus the :class:`ModelMetadata` document that
describes it.  They are written as two files (``.joblib`` + ``.json``) so the
metadata stays readable and diffable without unpickling anything.

Loading validates the metadata against its Pydantic contract, which means a
corrupted or hand-edited artifact fails loudly at load time rather than
producing silently wrong predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wind_turbine_pm.config import Config
from wind_turbine_pm.contracts.metadata import ModelMetadata
from wind_turbine_pm.logging_config import get_logger
from wind_turbine_pm.utils.io import (
    ArtifactNotFoundError,
    read_json,
    read_pickle,
    write_json,
    write_pickle,
)
from wind_turbine_pm.utils.paths import resolve

logger = get_logger(__name__)

#: Command shown to users when an artifact is missing.
TRAIN_HINT = "python scripts/run_failure_pipeline.py   (or: make pipeline)"


@dataclass(frozen=True)
class ModelBundle:
    """A fitted estimator together with its validated metadata."""

    estimator: Any
    metadata: ModelMetadata

    @property
    def features(self) -> list[str]:
        """Feature names in the order the estimator expects."""
        return list(self.metadata.features)

    @property
    def threshold(self) -> float:
        """The published decision threshold."""
        return float(self.metadata.threshold.value)

    @property
    def version(self) -> str:
        """The published model version."""
        return self.metadata.model_version


def model_path(cfg: Config) -> Path:
    """Path of the serialised estimator.

    Args:
        cfg: Merged configuration.

    Returns:
        Absolute artifact path.
    """
    return resolve(
        Path(str(cfg.require("paths.artifacts_models")))
        / str(cfg.require("model.artifact_filename"))
    )


def metadata_path(cfg: Config) -> Path:
    """Path of the model metadata document.

    Args:
        cfg: Merged configuration.

    Returns:
        Absolute artifact path.
    """
    return resolve(
        Path(str(cfg.require("paths.artifacts_metadata")))
        / str(cfg.require("model.metadata_filename"))
    )


def threshold_path(cfg: Config) -> Path:
    """Path of the standalone threshold document.

    Args:
        cfg: Merged configuration.

    Returns:
        Absolute artifact path.
    """
    return resolve(
        Path(str(cfg.require("paths.artifacts_metadata")))
        / str(cfg.require("model.threshold_filename"))
    )


def metrics_path(cfg: Config) -> Path:
    """Path of the metrics document.

    Args:
        cfg: Merged configuration.

    Returns:
        Absolute artifact path.
    """
    return resolve(
        Path(str(cfg.require("paths.artifacts_metrics")))
        / str(cfg.require("model.metrics_filename"))
    )


def comparison_path(cfg: Config) -> Path:
    """Path of the model-comparison CSV.

    Args:
        cfg: Merged configuration.

    Returns:
        Absolute artifact path.
    """
    return resolve(
        Path(str(cfg.require("paths.artifacts_metrics")))
        / str(cfg.require("model.comparison_filename"))
    )


def figures_dir(cfg: Config) -> Path:
    """Directory holding evaluation and explainability figures.

    Args:
        cfg: Merged configuration.

    Returns:
        Absolute directory path.
    """
    return resolve(str(cfg.require("paths.artifacts_figures")))


def save_bundle(estimator: Any, metadata: ModelMetadata, cfg: Config) -> tuple[Path, Path]:
    """Persist an estimator and its metadata.

    Args:
        estimator: The fitted pipeline.
        metadata: Its validated metadata document.
        cfg: Merged configuration.

    Returns:
        ``(estimator_path, metadata_path)``.
    """
    estimator_file = write_pickle(estimator, model_path(cfg))
    metadata_file = write_json(metadata.model_dump(mode="json"), metadata_path(cfg))
    write_json(metadata.threshold.model_dump(mode="json"), threshold_path(cfg))
    logger.info(
        "Saved model bundle",
        extra={
            "estimator": str(estimator_file),
            "metadata": str(metadata_file),
            "version": metadata.model_version,
        },
    )
    return estimator_file, metadata_file


#: Libraries whose version affects whether a pickled estimator loads correctly.
#: scikit-learn only guarantees unpickling under the version that wrote the file;
#: numpy and joblib participate in the serialised object graph.
COMPATIBILITY_CRITICAL: tuple[str, ...] = ("scikit-learn", "numpy", "joblib")


def check_runtime_compatibility(metadata: ModelMetadata) -> list[str]:
    """Compare the runtime's library versions against the artifact's.

    An estimator pickled by one scikit-learn release is not guaranteed to load
    correctly under another. This is easy to hit in practice: the Docker image
    mounts artifacts produced on the host, so any drift between the two
    environments shows up here rather than as a silently wrong prediction.

    Args:
        metadata: The loaded model metadata, carrying ``library_versions``.

    Returns:
        One human-readable message per mismatch; empty when the runtime matches
        (or when the artifact predates version recording).
    """
    recorded = dict(metadata.library_versions or {})
    if not recorded:
        return []

    import importlib.metadata as importlib_metadata

    warnings: list[str] = []
    for package in COMPATIBILITY_CRITICAL:
        expected = recorded.get(package)
        if not expected or expected == "not installed":
            continue
        try:
            actual = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:  # pragma: no cover - broken env
            warnings.append(
                f"{package} is recorded as {expected} in the artifact but is not installed"
            )
            continue
        if actual != expected:
            warnings.append(
                f"{package} {actual} differs from {expected}, which produced the model artifact"
            )
    return warnings


def load_bundle(cfg: Config) -> ModelBundle:
    """Load the published model bundle.

    Args:
        cfg: Merged configuration.

    Returns:
        The :class:`ModelBundle`.

    Raises:
        ArtifactNotFoundError: If either file is missing, carrying the command
            that regenerates them.
        pydantic.ValidationError: If the metadata does not satisfy its contract.
    """
    estimator = read_pickle(model_path(cfg), hint=TRAIN_HINT)
    raw_metadata = read_json(metadata_path(cfg), hint=TRAIN_HINT)
    metadata = ModelMetadata.model_validate(raw_metadata)

    for message in check_runtime_compatibility(metadata):
        logger.warning(
            "Runtime/artifact version mismatch: %s. "
            "Predictions may be unreliable - rebuild the artifacts in this environment "
            "with: python scripts/run_failure_pipeline.py --force",
            message,
        )

    logger.info(
        "Loaded model bundle",
        extra={"version": metadata.model_version, "features": metadata.n_features},
    )
    return ModelBundle(estimator=estimator, metadata=metadata)


def bundle_available(cfg: Config) -> bool:
    """Return whether both bundle files are present.

    Args:
        cfg: Merged configuration.

    Returns:
        ``True`` when the estimator and metadata files both exist.
    """
    return model_path(cfg).is_file() and metadata_path(cfg).is_file()


def load_metrics(cfg: Config) -> dict[str, Any]:
    """Load the evaluation metrics document.

    Args:
        cfg: Merged configuration.

    Returns:
        The metrics document.

    Raises:
        ArtifactNotFoundError: If the file is missing.
    """
    return read_json(metrics_path(cfg), hint=TRAIN_HINT)


def try_load_metrics(cfg: Config) -> dict[str, Any] | None:
    """Load metrics, returning ``None`` instead of raising when absent.

    Used by the dashboard, which must degrade gracefully.

    Args:
        cfg: Merged configuration.

    Returns:
        The metrics document, or ``None``.
    """
    try:
        return load_metrics(cfg)
    except ArtifactNotFoundError:
        return None
