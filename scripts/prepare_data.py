#!/usr/bin/env python
"""Validate, preprocess, label and feature-engineer the raw SCADA dataset.

Writes to ``data/processed/``:

* ``failure_dataset.parquet``  - labelled, cleaned observations with metadata
* ``failure_features.parquet`` - the model-facing feature matrix
* ``feature_spec.json``        - the feature contract (names, order, groups)
* ``split_boundaries.json``    - the chronological split boundaries

Order matters: features are built on the *full* cleaned history before rows are
filtered for modelling eligibility.  Filtering first would silently truncate the
rolling windows of every row that follows a removed observation.

Usage:
    python scripts/prepare_data.py [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from wind_turbine_pm.config import Config, load_config
from wind_turbine_pm.constants import SPLIT_COLUMN, TARGET_COLUMN, TIMESTAMP, TURBINE_ID
from wind_turbine_pm.data.ingestion import is_synthetic, load_raw_dataset
from wind_turbine_pm.data.preprocessing import (
    apply_modelling_filter,
    create_failure_target,
    preprocess,
)
from wind_turbine_pm.data.splitting import assign_splits, compute_boundaries
from wind_turbine_pm.data.validation import validate_scada_frame, validate_target
from wind_turbine_pm.features.failure_features import build_failure_features
from wind_turbine_pm.logging_config import configure_from_config, get_logger
from wind_turbine_pm.utils.io import write_json, write_table
from wind_turbine_pm.utils.paths import resolve
from wind_turbine_pm.utils.reproducibility import set_global_seed

logger = get_logger("scripts.prepare_data")

DATASET_FILENAME = "failure_dataset.parquet"
FEATURES_FILENAME = "failure_features.parquet"
SPEC_FILENAME = "feature_spec.json"
BOUNDARIES_FILENAME = "split_boundaries.json"

#: Non-feature columns carried alongside the feature matrix for the dashboard,
#: EDA and evaluation. They are never passed to the model.
METADATA_COLUMNS = (
    TURBINE_ID,
    TIMESTAMP,
    TARGET_COLUMN,
    SPLIT_COLUMN,
    "operational_status",
    "failure_event",
    "maintenance_event",
    "degradation_level",
    "failure_mode",
    "hours_to_failure",
)


def processed_dir(cfg: Config) -> Path:
    """Return the processed-data directory.

    Args:
        cfg: Merged configuration.

    Returns:
        Absolute directory path.
    """
    return resolve(str(cfg.require("paths.data_processed")))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; ``sys.argv[1:]`` when omitted.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild even if outputs exist.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort when data validation reports any error-severity finding.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the preparation step.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    cfg = load_config()
    configure_from_config(cfg)
    set_global_seed(int(cfg.get("random_seed", 42)))

    output_dir = processed_dir(cfg)
    features_path = output_dir / FEATURES_FILENAME
    if features_path.is_file() and not args.force:
        logger.info(
            "Processed data already exists; use --force to rebuild",
            extra={"path": str(features_path)},
        )
        return 0

    # --- Ingest & validate ------------------------------------------------
    raw = load_raw_dataset(cfg)
    report = validate_scada_frame(raw, cfg)
    if args.strict:
        report.raise_for_errors()
    elif not report.is_valid:
        logger.error(
            "Validation reported errors; continuing because --strict was not set",
            extra={"errors": len(report.errors)},
        )

    # --- Clean, label, engineer ------------------------------------------
    clean, preprocessing_stats = preprocess(raw, cfg)
    labelled = create_failure_target(clean, cfg)
    features, spec = build_failure_features(labelled, cfg)

    eligible = apply_modelling_filter(labelled)
    features = features.loc[eligible.index]

    report = validate_target(eligible, TARGET_COLUMN, report)
    if args.strict:
        report.raise_for_errors()

    # --- Split ------------------------------------------------------------
    boundaries = compute_boundaries(eligible, cfg)
    eligible = eligible.copy()
    eligible[SPLIT_COLUMN] = assign_splits(eligible, boundaries)

    metadata_columns = [column for column in METADATA_COLUMNS if column in eligible.columns]
    dataset = eligible[metadata_columns].reset_index(drop=True)
    features = features.reset_index(drop=True)

    # --- Persist ----------------------------------------------------------
    dataset_path = write_table(dataset, output_dir / DATASET_FILENAME)
    features_path = write_table(features, output_dir / FEATURES_FILENAME)
    spec_payload = spec.to_dict()
    spec_payload["target"] = TARGET_COLUMN
    spec_payload["horizon_hours"] = int(cfg.require("target.horizon_hours"))
    spec_path = write_json(spec_payload, output_dir / SPEC_FILENAME)
    boundaries_path = write_json(boundaries.to_dict(), output_dir / BOUNDARIES_FILENAME)

    report.summary["preprocessing"] = preprocessing_stats
    report.summary["split"] = boundaries.to_dict()
    report.summary["split_counts"] = dataset[SPLIT_COLUMN].value_counts().to_dict()
    report.summary["is_synthetic"] = is_synthetic(cfg)
    validation_path = write_json(
        report.to_dict(),
        resolve(str(cfg.require("paths.artifacts_metrics"))) / "validation_report.json",
    )

    positives_by_split = dataset.groupby(SPLIT_COLUMN)[TARGET_COLUMN].agg(["size", "sum", "mean"])
    logger.info(
        "Preparation complete",
        extra={
            "dataset": str(dataset_path),
            "features": str(features_path),
            "spec": str(spec_path),
            "boundaries": str(boundaries_path),
            "validation_report": str(validation_path),
            "rows": len(dataset),
            "n_features": len(spec.names),
            "splits": {
                str(name): {
                    "rows": int(row["size"]),
                    "positives": int(row["sum"]),
                    "rate": round(float(row["mean"]), 5),
                }
                for name, row in positives_by_split.iterrows()
            },
        },
    )
    print(positives_by_split.to_string())  # noqa: T201 - operator-facing summary
    return 0


def load_prepared(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load the prepared dataset, features and feature spec.

    Args:
        cfg: Merged configuration.

    Returns:
        ``(dataset, features, feature_spec)``.

    Raises:
        wind_turbine_pm.utils.io.ArtifactNotFoundError: If preparation has not
            been run.
    """
    from wind_turbine_pm.utils.io import read_json, read_table

    hint = "python scripts/prepare_data.py"
    directory = processed_dir(cfg)
    dataset = read_table(directory / DATASET_FILENAME, hint=hint)
    features = read_table(directory / FEATURES_FILENAME, hint=hint)
    spec = read_json(directory / SPEC_FILENAME, hint=hint)
    return dataset, features, spec


if __name__ == "__main__":
    sys.exit(main())
