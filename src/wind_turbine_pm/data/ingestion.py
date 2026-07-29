"""Dataset ingestion.

Two sources are supported today:

``synthetic``
    Generate the dataset in-process with
    :func:`wind_turbine_pm.data.synthetic.generate_scada_dataset`.
``file``
    Read a CSV or Parquet SCADA export from ``configs/data.yaml -> data.path``.

**Dataset decision (documented in README):** no wind-turbine SCADA dataset with
a public, stable, no-authentication download endpoint was available that could
be wired in without scraping or manual cookie handling, so the project ships a
synthetic generator as its default source.  The ``file`` branch below is the
integration point for a real export - nothing downstream of this module needs to
change, because everything consumes the canonical column contract in
:mod:`wind_turbine_pm.constants`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import REQUIRED_RAW_COLUMNS, TIMESTAMP, TURBINE_ID
from wind_turbine_pm.data.synthetic import generate_scada_dataset
from wind_turbine_pm.logging_config import get_logger
from wind_turbine_pm.utils.io import read_table, write_table
from wind_turbine_pm.utils.paths import resolve

logger = get_logger(__name__)


class IngestionError(RuntimeError):
    """Raised when a dataset cannot be ingested."""


def raw_data_path(cfg: Config) -> Path:
    """Return the canonical path of the raw dataset.

    Args:
        cfg: Merged configuration.

    Returns:
        Absolute path to the raw dataset file.
    """
    return resolve(Path(str(cfg.require("paths.data_raw"))) / str(cfg.require("data.raw_filename")))


def sample_data_path(cfg: Config) -> Path:
    """Return the path of the small committed sample extract.

    Args:
        cfg: Merged configuration.

    Returns:
        Absolute path to the sample CSV.
    """
    return resolve(
        Path(str(cfg.require("paths.data_samples"))) / str(cfg.require("data.sample_filename"))
    )


def is_synthetic(cfg: Config) -> bool:
    """Return whether the configured data source is simulated.

    Args:
        cfg: Merged configuration.

    Returns:
        ``True`` when ``data.source`` is ``"synthetic"``.
    """
    return str(cfg.require("data.source")).lower() == "synthetic"


def load_raw_dataset(cfg: Config, regenerate: bool = False) -> pd.DataFrame:
    """Load the raw SCADA dataset according to configuration.

    Args:
        cfg: Merged configuration.
        regenerate: For the synthetic source, rebuild even if a cached file
            exists.

    Returns:
        The raw dataframe.

    Raises:
        IngestionError: If the configured source is unknown, the configured
            file is missing, or required columns are absent.
    """
    source = str(cfg.require("data.source")).lower()

    if source == "synthetic":
        target = raw_data_path(cfg)
        if target.is_file() and not regenerate:
            logger.info("Loading cached synthetic dataset", extra={"path": str(target)})
            frame = read_table(target)
        else:
            frame = generate_scada_dataset(cfg)
    elif source == "file":
        configured = cfg.get("data.path")
        if not configured:
            raise IngestionError("data.source='file' requires data.path in configuration")
        path = resolve(str(configured))
        if not path.is_file():
            raise IngestionError(f"Configured SCADA export not found: {path}")
        logger.info("Loading SCADA export", extra={"path": str(path)})
        frame = read_table(path)
    else:
        raise IngestionError(f"Unknown data.source {source!r}; expected 'synthetic' or 'file'")

    missing = [column for column in REQUIRED_RAW_COLUMNS if column not in frame.columns]
    if missing:
        raise IngestionError(
            f"Ingested dataset is missing required columns: {missing}. "
            "See wind_turbine_pm.constants.REQUIRED_RAW_COLUMNS for the contract."
        )
    return frame


def persist_raw_dataset(frame: pd.DataFrame, cfg: Config) -> tuple[Path, Path]:
    """Write the raw dataset plus a small committed sample extract.

    The full dataset is deliberately git-ignored; the sample is small enough to
    commit so that reviewers can inspect the schema without regenerating data.

    Args:
        frame: Raw dataframe.
        cfg: Merged configuration.

    Returns:
        ``(raw_path, sample_path)``.
    """
    raw_path = write_table(frame, raw_data_path(cfg))
    n_sample = int(cfg.get("data.sample_rows", 2000))
    sample = (
        frame.sort_values([TURBINE_ID, TIMESTAMP], na_position="last")
        .groupby(TURBINE_ID, dropna=True, group_keys=False)
        .head(max(n_sample // max(frame[TURBINE_ID].nunique(), 1), 1))
    )
    sample_path = write_table(sample, sample_data_path(cfg))
    logger.info(
        "Persisted raw dataset",
        extra={"raw": str(raw_path), "sample": str(sample_path), "rows": len(frame)},
    )
    return raw_path, sample_path
