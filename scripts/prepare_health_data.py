#!/usr/bin/env python
"""Prepare the Turbine Health Monitoring dataset and feature matrix.

Writes to ``data/processed/``:

* ``health_dataset.parquet``      - cleaned observations with the health target,
  the operating regime and the split label
* ``health_features.parquet``     - the model-facing health feature matrix
* ``health_feature_spec.json``    - the feature contract (names, order, groups)
* ``health_split_boundaries.json`` - the chronological split boundaries

Why this is a separate script rather than a flag on ``prepare_data.py``
---------------------------------------------------------------------
The two modules answer different questions and therefore need different
pipelines.  Failure prediction builds a forward-looking binary label and is
filtered by ``modelling_eligible`` (which drops the unlabelable tail of each
turbine's record).  Health scoring regresses onto the machine's *current*
condition, has no horizon, and must keep rows the failure filter removes.
Sharing one script would mean one of the two silently getting the other's
eligibility rules.

Both scripts read the same raw file and the same ``configs/data.yaml``, so the
cleaning and the physical ranges are identical - which is what keeps the two
modules describing the same fleet.

Order matters: features are built on the *full* cleaned history before rows are
filtered for label eligibility.  Filtering first would truncate the trailing
windows of every row that follows a removed observation.

Usage:
    python scripts/prepare_health_data.py [--force] [--strict]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import (
    HEALTH_CLASS,
    OPERATING_REGIME,
    SENSOR_COLUMNS,
    SPLIT_COLUMN,
    TIMESTAMP,
    TURBINE_ID,
    SplitName,
)
from wind_turbine_pm.data.ingestion import is_synthetic, load_raw_dataset
from wind_turbine_pm.data.preprocessing import preprocess
from wind_turbine_pm.data.splitting import assign_splits, compute_boundaries
from wind_turbine_pm.data.validation import validate_scada_frame
from wind_turbine_pm.health.config import load_health_config
from wind_turbine_pm.health.health_class import class_distribution, classify_health_series
from wind_turbine_pm.health.health_features import build_health_features
from wind_turbine_pm.health.health_score import apply_label_filter, build_health_target
from wind_turbine_pm.health.persistence import (
    health_dataset_path,
    health_features_path,
    health_spec_path,
)
from wind_turbine_pm.health.regimes import attach_regimes, regime_summary
from wind_turbine_pm.health.sensor_rules import evaluate_rules, verify_against_physical_ranges
from wind_turbine_pm.logging_config import configure_from_config, get_logger
from wind_turbine_pm.utils.io import write_json, write_table
from wind_turbine_pm.utils.paths import resolve
from wind_turbine_pm.utils.reproducibility import set_global_seed

logger = get_logger("scripts.prepare_health_data")

BOUNDARIES_FILENAME = "health_split_boundaries.json"

#: Non-feature columns carried alongside the feature matrix for the dashboard,
#: evaluation and EDA. They are never passed to the model.
#:
#: The raw sensor channels are included deliberately. A health assessment reports
#: rule violations and component scores that are computed from *raw* readings, not
#: from the feature vector, so without them the prepared dataset could not be
#: assessed and the dashboard would have to recompute features from a short raw
#: window - producing a different score for the same turbine at the same
#: timestamp than the fleet table shows. See
#: :meth:`~wind_turbine_pm.services.health_monitoring_service.HealthMonitoringService.assess_from_prepared`.
METADATA_COLUMNS = (
    TURBINE_ID,
    TIMESTAMP,
    OPERATING_REGIME,
    SPLIT_COLUMN,
    *SENSOR_COLUMNS,
    "operational_status",
    "failure_event",
    "maintenance_event",
    "degradation_level",
    "failure_mode",
    "hours_to_failure",
)


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
        help="Abort when data validation or rule verification reports a problem.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the health preparation step.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    cfg = load_health_config()
    configure_from_config(cfg)
    set_global_seed(int(cfg.get("random_seed", 42)))

    features_path = health_features_path(cfg)
    if features_path.is_file() and not args.force:
        logger.info(
            "Prepared health data already exists; use --force to rebuild",
            extra={"path": str(features_path)},
        )
        return 0

    # --- Rule provenance check -------------------------------------------
    # The health rules restate the platform's physical ranges so they can be
    # read on their own. Restating a value is a chance to disagree with it, so
    # the agreement is verified before anything is built on the rules.
    problems = verify_against_physical_ranges(cfg)
    if problems:
        message = (
            "Health sensor rules disagree with configs/data.yaml -> "
            "validation.physical_ranges:\n  - " + "\n  - ".join(problems)
        )
        if args.strict:
            raise ValueError(message)
        logger.error(message)

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
    regimed = attach_regimes(clean, cfg)
    labelled = build_health_target(regimed, cfg)

    features, spec = build_health_features(labelled, cfg)

    eligible = apply_label_filter(labelled)
    features = features.loc[eligible.index]

    # --- Split ------------------------------------------------------------
    boundaries = compute_boundaries(eligible, cfg)
    eligible = eligible.copy()
    eligible[SPLIT_COLUMN] = assign_splits(eligible, boundaries)

    target_name = str(cfg.require("health.target.name"))
    metadata_columns = [
        column for column in (*METADATA_COLUMNS, target_name) if column in eligible.columns
    ]
    dataset = eligible[metadata_columns].reset_index(drop=True)
    features = features.reset_index(drop=True)

    # The ground-truth class distribution is the operator-facing summary of what
    # the model has to learn, and the fastest way to notice a band that no row
    # ever falls into.
    dataset[HEALTH_CLASS] = classify_health_series(dataset[target_name], cfg).to_numpy()

    # --- Data-quality summary from the rules ------------------------------
    rule_evaluation = evaluate_rules(clean, cfg)

    # --- Persist ----------------------------------------------------------
    dataset_file = write_table(dataset, health_dataset_path(cfg))
    features_file = write_table(features, health_features_path(cfg))

    spec_payload = spec.to_dict()
    spec_payload["target"] = target_name
    spec_payload["target_source"] = str(cfg.require("health.target.source_column"))
    spec_file = write_json(spec_payload, health_spec_path(cfg))
    boundaries_file = write_json(
        boundaries.to_dict(),
        resolve(str(cfg.require("paths.data_processed"))) / BOUNDARIES_FILENAME,
    )

    regimes = regime_summary(dataset)
    classes = class_distribution(dataset[HEALTH_CLASS])
    quality_file = write_json(
        {
            "preprocessing": preprocessing_stats,
            "sensor_rules": rule_evaluation.to_dict(),
            "regimes": regimes.to_dict(orient="records"),
            "health_class_counts": classes,
            "split_counts": dataset[SPLIT_COLUMN].value_counts().to_dict(),
            "split": boundaries.to_dict(),
            "is_synthetic": is_synthetic(cfg),
            "rule_range_problems": problems,
        },
        resolve(str(cfg.require("paths.artifacts_metrics"))) / "health_data_quality.json",
    )

    # A split with no degraded observations makes validation and test
    # meaningless: `mae_degraded` is the primary selection metric, and it is
    # only defined where degraded rows exist. Catch it here, where the cause and
    # the fix are obvious, rather than as a confusing selection failure later.
    boundary = float(cfg.get("health.classes.monitor_min", 60.0))
    degraded_by_split = (
        dataset.assign(_degraded=dataset[target_name] < boundary)
        .groupby(SPLIT_COLUMN)["_degraded"]
        .agg(["size", "sum", "mean"])
    )
    empty = [
        str(name)
        for name in (SplitName.TRAIN, SplitName.VALID, SplitName.TEST)
        if str(name) in degraded_by_split.index and degraded_by_split.loc[str(name), "sum"] == 0
    ]
    if empty:
        message = (
            f"Split(s) {empty} contain no observations below the Monitor boundary "
            f"({boundary:.0f}), so the health model cannot be selected or evaluated on them. "
            f"The dataset is most likely too small or too short for degradation episodes to "
            f"reach every period. Generate more data, e.g.:\n"
            f"  python scripts/generate_synthetic_data.py --turbines 20 --months 12 --force"
        )
        if args.strict:
            raise ValueError(message)
        logger.error(message)

    logger.info(
        "Health preparation complete",
        extra={
            "dataset": str(dataset_file),
            "features": str(features_file),
            "spec": str(spec_file),
            "boundaries": str(boundaries_file),
            "data_quality": str(quality_file),
            "rows": len(dataset),
            "n_features": len(spec.names),
            "health_class_counts": classes,
            "data_quality_fraction": round(rule_evaluation.data_quality(), 4),
        },
    )
    print(degraded_by_split.to_string())  # noqa: T201 - operator-facing summary
    print(regimes.to_string(index=False))  # noqa: T201
    return 0


def load_prepared_health(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load the prepared health dataset, features and feature spec.

    Args:
        cfg: Merged configuration carrying the ``health`` namespace.

    Returns:
        ``(dataset, features, feature_spec)``.

    Raises:
        wind_turbine_pm.utils.io.ArtifactNotFoundError: If preparation has not
            been run.
    """
    from wind_turbine_pm.utils.io import read_json, read_table

    hint = "python scripts/prepare_health_data.py"
    dataset = read_table(health_dataset_path(cfg), hint=hint)
    features = read_table(health_features_path(cfg), hint=hint)
    spec = read_json(health_spec_path(cfg), hint=hint)
    return dataset, features, spec


def boundaries_path(cfg: Config) -> Path:
    """Path of the health split-boundary document.

    Args:
        cfg: Merged configuration.

    Returns:
        Absolute path.
    """
    return resolve(str(cfg.require("paths.data_processed"))) / BOUNDARIES_FILENAME


if __name__ == "__main__":
    sys.exit(main())
