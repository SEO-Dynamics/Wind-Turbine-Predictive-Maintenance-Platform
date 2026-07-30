"""Anomaly artifact paths and validated bundle loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from wind_turbine_pm.config import Config
from wind_turbine_pm.contracts.anomaly import AnomalyCalibration, AnomalyModelMetadata
from wind_turbine_pm.utils.io import (
    ArtifactNotFoundError,
    read_json,
    read_pickle,
    read_table,
    write_json,
    write_pickle,
    write_table,
)
from wind_turbine_pm.utils.paths import resolve

TRAIN_HINT = "python scripts/run_anomaly_pipeline.py"


def _artifact_path(cfg: Config, directory: str, key: str) -> Path:
    return resolve(Path(str(cfg.require(directory))) / str(cfg.require(f"anomaly.artifacts.{key}")))


def model_path(cfg: Config) -> Path:
    return _artifact_path(cfg, "paths.artifacts_models", "model")


def metadata_path(cfg: Config) -> Path:
    return _artifact_path(cfg, "paths.artifacts_metadata", "metadata")


def calibration_path(cfg: Config) -> Path:
    return _artifact_path(cfg, "paths.artifacts_metadata", "calibration")


def metrics_path(cfg: Config) -> Path:
    return _artifact_path(cfg, "paths.artifacts_metrics", "metrics")


def comparison_path(cfg: Config) -> Path:
    return _artifact_path(cfg, "paths.artifacts_metrics", "comparison")


def reference_path(cfg: Config) -> Path:
    return _artifact_path(cfg, "paths.artifacts_models", "reference")


def dataset_path(cfg: Config) -> Path:
    return _artifact_path(cfg, "paths.data_processed", "dataset")


def features_path(cfg: Config) -> Path:
    return _artifact_path(cfg, "paths.data_processed", "features")


def feature_spec_path(cfg: Config) -> Path:
    return _artifact_path(cfg, "paths.data_processed", "feature_spec")


@dataclass(frozen=True)
class AnomalyBundle:
    estimator: Any
    metadata: AnomalyModelMetadata
    calibration: AnomalyCalibration
    reference: pd.DataFrame


def save_bundle(
    estimator: Any,
    metadata: AnomalyModelMetadata,
    calibration: AnomalyCalibration,
    reference: pd.DataFrame,
    cfg: Config,
) -> None:
    write_pickle(estimator, model_path(cfg))
    write_json(metadata.model_dump(mode="json"), metadata_path(cfg))
    write_json(calibration.model_dump(mode="json"), calibration_path(cfg))
    write_table(reference.reset_index(drop=True), reference_path(cfg))


def load_bundle(cfg: Config) -> AnomalyBundle:
    return AnomalyBundle(
        estimator=read_pickle(model_path(cfg), hint=TRAIN_HINT),
        metadata=AnomalyModelMetadata.model_validate(
            read_json(metadata_path(cfg), hint=TRAIN_HINT)
        ),
        calibration=AnomalyCalibration.model_validate(
            read_json(calibration_path(cfg), hint=TRAIN_HINT)
        ),
        reference=read_table(reference_path(cfg), hint=TRAIN_HINT),
    )


def bundle_available(cfg: Config) -> bool:
    return all(
        path.is_file()
        for path in (
            model_path(cfg),
            metadata_path(cfg),
            calibration_path(cfg),
            reference_path(cfg),
        )
    )


def try_load_metrics(cfg: Config) -> dict[str, Any] | None:
    try:
        return read_json(metrics_path(cfg), hint=TRAIN_HINT)
    except ArtifactNotFoundError:
        return None
