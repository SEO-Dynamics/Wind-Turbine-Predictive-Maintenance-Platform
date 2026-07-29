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

from wind_turbine_pm.config import Config, load_config
from wind_turbine_pm.data.preprocessing import preprocess
from wind_turbine_pm.models.persistence import figures_dir, try_load_metrics
from wind_turbine_pm.services.failure_prediction_service import FailurePredictionService
from wind_turbine_pm.utils.io import ArtifactNotFoundError, read_table
from wind_turbine_pm.utils.paths import resolve

PIPELINE_COMMAND = "python scripts/run_failure_pipeline.py"
PREPARE_COMMAND = "python scripts/prepare_data.py"
EVALUATE_COMMAND = "python scripts/evaluate_failure_model.py"


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
