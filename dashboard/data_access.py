"""Cached artifact access for the dashboard.

Everything here returns ``None`` (or an empty frame) instead of raising when an
artifact is missing, so pages can render an actionable message.  Prediction
logic is *not* implemented here - the dashboard calls
:class:`~wind_turbine_pm.services.failure_prediction_service.FailurePredictionService`,
the same class the API uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from wind_turbine_pm.anomaly.config import load_anomaly_config
from wind_turbine_pm.anomaly.persistence import (
    dataset_path as anomaly_dataset_path,
)
from wind_turbine_pm.anomaly.persistence import (
    features_path as anomaly_features_path,
)
from wind_turbine_pm.anomaly.persistence import try_load_metrics as try_load_anomaly_metrics
from wind_turbine_pm.config import Config, load_config
from wind_turbine_pm.contracts.observations import frame_to_windows
from wind_turbine_pm.data.preprocessing import preprocess
from wind_turbine_pm.health.config import load_health_config
from wind_turbine_pm.health.persistence import (
    health_dataset_path,
    health_features_path,
    try_load_health_metrics,
)
from wind_turbine_pm.models.persistence import figures_dir, try_load_metrics
from wind_turbine_pm.services.anomaly_detection_service import AnomalyDetectionService
from wind_turbine_pm.services.failure_prediction_service import FailurePredictionService
from wind_turbine_pm.services.health_monitoring_service import HealthMonitoringService
from wind_turbine_pm.services.maintenance_service import MaintenanceDecisionService
from wind_turbine_pm.utils.io import ArtifactNotFoundError, read_table
from wind_turbine_pm.utils.paths import resolve

PIPELINE_COMMAND = "python scripts/run_failure_pipeline.py"
PREPARE_COMMAND = "python scripts/prepare_data.py"
EVALUATE_COMMAND = "python scripts/evaluate_failure_model.py"
HEALTH_PIPELINE_COMMAND = "python scripts/run_health_pipeline.py"
HEALTH_PREPARE_COMMAND = "python scripts/prepare_health_data.py"
ANOMALY_PIPELINE_COMMAND = "python scripts/run_anomaly_pipeline.py"
ANOMALY_PREPARE_COMMAND = "python scripts/prepare_anomaly_data.py"


@st.cache_resource(show_spinner=False)
def get_config_cached() -> Config:
    """Load and cache the merged configuration.

    Returns:
        The configuration.
    """
    return load_config()


@st.cache_resource(show_spinner="Loading model artifacts...")
def get_service_cached() -> FailurePredictionService:
    """Build and cache the prediction service.

    Returns:
        The service (which may not be ready; check ``is_ready``).
    """
    return FailurePredictionService(get_config_cached())


@st.cache_data(show_spinner="Loading prepared dataset...")
def load_dataset() -> pd.DataFrame | None:
    """Load the prepared labelled dataset.

    Returns:
        The dataset, or ``None`` when preparation has not been run.
    """
    cfg = get_config_cached()
    path = resolve(str(cfg.require("paths.data_processed"))) / "failure_dataset.parquet"
    try:
        return read_table(path, hint=PREPARE_COMMAND)
    except ArtifactNotFoundError:
        return None


@st.cache_data(show_spinner="Loading feature matrix...")
def load_features() -> pd.DataFrame | None:
    """Load the prepared feature matrix.

    Returns:
        The feature matrix, or ``None`` when preparation has not been run.
    """
    cfg = get_config_cached()
    path = resolve(str(cfg.require("paths.data_processed"))) / "failure_features.parquet"
    try:
        return read_table(path, hint=PREPARE_COMMAND)
    except ArtifactNotFoundError:
        return None


@st.cache_data(show_spinner="Loading SCADA data...")
def load_raw_sample() -> pd.DataFrame | None:
    """Load the SCADA dataset for sensor-trend plots, with defects treated.

    The raw file deliberately contains physically impossible readings so the
    validation layer has something to detect. Those values are replaced here
    before plotting - charting a 999 degC gearbox would make every real trend
    invisible and misrepresent the data an operator would actually see.

    Returns:
        The cleaned dataset, or ``None`` when generation has not been run.
    """
    cfg = get_config_cached()
    path = resolve(str(cfg.require("paths.data_raw"))) / str(cfg.require("data.raw_filename"))
    try:
        raw = read_table(path, hint=PIPELINE_COMMAND)
    except ArtifactNotFoundError:
        return None
    clean, _ = preprocess(raw, cfg)
    return clean


@st.cache_data(show_spinner=False)
def load_metrics() -> dict[str, Any] | None:
    """Load the evaluation metrics document.

    Returns:
        The metrics document, or ``None`` when it has not been written.
    """
    return try_load_metrics(get_config_cached())


@st.cache_data(show_spinner=False)
def load_global_importance() -> pd.DataFrame | None:
    """Load the global feature-importance table.

    Returns:
        The importance table, or ``None`` when evaluation has not been run.
    """
    cfg = get_config_cached()
    path = resolve(str(cfg.require("paths.artifacts_metrics"))) / "global_feature_importance.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path)


def figure_path(name: str) -> Path | None:
    """Resolve a figure by file name.

    Args:
        name: File name inside the figures directory.

    Returns:
        The path when it exists, otherwise ``None``.
    """
    path = figures_dir(get_config_cached()) / name
    return path if path.is_file() else None


# ---------------------------------------------------------------------------
# Turbine Health Monitoring
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_health_config_cached() -> Config:
    """Load and cache the configuration including the ``health`` namespace.

    Kept separate from :func:`get_config_cached` because the health module needs
    ``health_model.yaml`` merged in, and the Failure Prediction page must keep
    seeing exactly the configuration it was trained with.

    Returns:
        The merged configuration.
    """
    return load_health_config()


@st.cache_resource(show_spinner="Loading health model artifacts...")
def get_health_service_cached() -> HealthMonitoringService:
    """Build and cache the health monitoring service.

    Returns:
        The service (which may not be ready; check ``is_ready``).
    """
    return HealthMonitoringService(get_health_config_cached())


@st.cache_data(show_spinner="Loading health dataset...")
def load_health_dataset() -> pd.DataFrame | None:
    """Load the prepared health dataset.

    Returns:
        The dataset, or ``None`` when health preparation has not been run.
    """
    try:
        return read_table(
            health_dataset_path(get_health_config_cached()), hint=HEALTH_PREPARE_COMMAND
        )
    except ArtifactNotFoundError:
        return None


@st.cache_data(show_spinner="Loading health features...")
def load_health_features() -> pd.DataFrame | None:
    """Load the prepared health feature matrix.

    Returns:
        The feature matrix, or ``None`` when health preparation has not been run.
    """
    try:
        return read_table(
            health_features_path(get_health_config_cached()), hint=HEALTH_PREPARE_COMMAND
        )
    except ArtifactNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def load_health_metrics() -> dict[str, Any] | None:
    """Load the health metrics document.

    Returns:
        The metrics document, or ``None`` when it has not been written.
    """
    return try_load_health_metrics(get_health_config_cached())


@st.cache_data(show_spinner="Scoring fleet health...")
def score_health(split: str = "test") -> pd.DataFrame | None:
    """Score one prepared split with the published health model.

    Args:
        split: Split name to score.

    Returns:
        A frame of health scores joined with the ground-truth columns, or
        ``None`` when the required artifacts are missing.
    """
    dataset, features = load_health_dataset(), load_health_features()
    service = get_health_service_cached()
    if dataset is None or features is None or not service.is_ready:
        return None

    subset = dataset.loc[dataset["split"] == split]
    if subset.empty:
        return None

    scored = service.score_frame(subset, features.loc[subset.index])
    carried = [
        column
        for column in ("health_score_target", "health_class", "degradation_level", "failure_mode")
        if column in subset.columns
    ]
    # The scored frame carries its own health_class (the prediction); the
    # ground-truth column of the same name is renamed so both survive the join
    # and a chart can compare them.
    truth = subset[carried].rename(columns={"health_class": "true_health_class"})
    return scored.join(truth).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_health_data_quality() -> dict[str, Any] | None:
    """Load the health data-quality summary written by preparation.

    Returns:
        The document, or ``None`` when preparation has not been run.
    """
    cfg = get_health_config_cached()
    path = resolve(str(cfg.require("paths.artifacts_metrics"))) / "health_data_quality.json"
    if not path.is_file():
        return None
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner="Scoring the fleet...")
def score_split(split: str = "test") -> pd.DataFrame | None:
    """Score one prepared split with the published model.

    Args:
        split: Split name to score.

    Returns:
        A frame of predictions joined with ground truth, or ``None`` when the
        required artifacts are missing.
    """
    dataset, features = load_dataset(), load_features()
    service = get_service_cached()
    if dataset is None or features is None or not service.is_ready:
        return None

    mask = dataset["split"] == split
    subset = dataset.loc[mask]
    if subset.empty:
        return None

    scored = service.predict_frame(subset, features.loc[subset.index])
    joined = subset.join(scored[["failure_probability", "prediction", "risk_level"]])
    return joined.reset_index(drop=True)


def latest_per_turbine(scored: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Reduce a scored frame to each turbine's latest observation at a point in time.

    The ``as_of`` cut-off makes the fleet view answer "what did the fleet look
    like at time T?", which is how an operator actually reads it. Without it the
    view is pinned to the final timestamp of the record, where most turbines are
    healthy by construction.

    Args:
        scored: Output of :func:`score_split`.
        as_of: Cut-off timestamp; the newest observation is used when omitted.

    Returns:
        One row per turbine, sorted by descending failure probability. The
        original dataframe index is preserved so feature rows can be looked up.
    """
    frame = scored.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    if as_of is not None:
        frame = frame.loc[frame["timestamp"] <= pd.Timestamp(as_of)]
    if frame.empty:
        return frame
    latest = frame.sort_values("timestamp").groupby("turbine_id", as_index=False).tail(1)
    return latest.sort_values("failure_probability", ascending=False)


# ---------------------------------------------------------------------------
# Anomaly Detection and Maintenance Decision Support
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_anomaly_config_cached() -> Config:
    """Load all three model namespaces and the maintenance policy."""
    return load_anomaly_config()


@st.cache_resource(show_spinner="Loading anomaly model artifacts...")
def get_anomaly_service_cached() -> AnomalyDetectionService:
    """Build the anomaly service used by both dashboard and API."""
    return AnomalyDetectionService(get_anomaly_config_cached())


@st.cache_resource(show_spinner="Loading unified maintenance services...")
def get_maintenance_service_cached() -> MaintenanceDecisionService:
    """Compose the three source services using the Stage 3 configuration."""
    cfg = get_anomaly_config_cached()
    return MaintenanceDecisionService(
        cfg,
        failure=FailurePredictionService(cfg),
        health=HealthMonitoringService(cfg),
        anomaly=get_anomaly_service_cached(),
    )


@st.cache_data(show_spinner="Loading anomaly dataset...")
def load_anomaly_dataset() -> pd.DataFrame | None:
    """Load row-aligned anomaly preparation output."""
    try:
        return read_table(
            anomaly_dataset_path(get_anomaly_config_cached()),
            hint=ANOMALY_PREPARE_COMMAND,
        )
    except ArtifactNotFoundError:
        return None


@st.cache_data(show_spinner="Loading anomaly features...")
def load_anomaly_features() -> pd.DataFrame | None:
    """Load the anomaly feature matrix in published model order."""
    try:
        return read_table(
            anomaly_features_path(get_anomaly_config_cached()),
            hint=ANOMALY_PREPARE_COMMAND,
        )
    except ArtifactNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def load_anomaly_metrics() -> dict[str, Any] | None:
    """Load calibration, comparison and held-out metrics."""
    return try_load_anomaly_metrics(get_anomaly_config_cached())


@st.cache_data(show_spinner="Scoring fleet anomalies...")
def score_anomaly(split: str = "test") -> pd.DataFrame | None:
    """Score one prepared split through the production anomaly service."""
    dataset, features = load_anomaly_dataset(), load_anomaly_features()
    service = get_anomaly_service_cached()
    if dataset is None or features is None or not service.is_ready:
        return None
    subset = dataset.loc[dataset["split"] == split]
    if subset.empty:
        return None
    scored = service.score_frame(subset, features.loc[subset.index])
    carried = [
        column
        for column in ("anomaly_truth", "degradation_level", "operational_status")
        if column in subset
    ]
    return scored.join(subset[carried]).reset_index(drop=True)


@st.cache_data(show_spinner="Building fleet maintenance priority queue...")
def assess_maintenance_fleet() -> list[dict[str, Any]] | None:
    """Assess the latest 97-hour window for every turbine with available models."""
    dataset = load_anomaly_dataset()
    service = get_maintenance_service_cached()
    if dataset is None or not bool(service.status()["model_loaded"]):
        return None
    windows = []
    for _, group in dataset.groupby("turbine_id", sort=True):
        windows.extend(frame_to_windows(group.sort_values("timestamp").tail(97)))
    if not windows:
        return None
    return [
        assessment.model_dump(mode="json")
        for assessment in service.assess_batch(windows).assessments
    ]
