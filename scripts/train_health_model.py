#!/usr/bin/env python
"""Train the health-score model, fit the drift detector and publish the bundle.

Information flow is strictly one-directional, exactly as in the Failure
Prediction Module: train fits, valid ranks and selects, test is scored once and
never fed back.  The multivariate drift detector is fitted on the **training
split only**, so "normal" means "what this fleet looked like during the training
period" rather than "whatever the scored data happens to contain".

Writes:

* ``artifacts/models/health_model.joblib``             - the selected estimator
* ``artifacts/models/health_drift_detector.joblib``    - the Isolation Forest
* ``artifacts/models/health_background.parquet``       - feature background sample
* ``artifacts/metadata/health_model_metadata.json``    - the published contract
* ``artifacts/metrics/health_metrics.json``            - all metrics, all splits
* ``artifacts/metrics/health_model_comparison.csv``    - candidate comparison
* ``artifacts/metrics/health_error_by_band.csv``       - error per health band
* ``artifacts/figures/health_*.png``                   - evaluation figures

Usage:
    python scripts/train_health_model.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from wind_turbine_pm.constants import SPLIT_COLUMN, TIMESTAMP, TURBINE_ID, SplitName
from wind_turbine_pm.contracts.health import HealthModelMetadata
from wind_turbine_pm.contracts.metadata import DatasetSummary
from wind_turbine_pm.data.ingestion import is_synthetic
from wind_turbine_pm.data.splitting import SplitBoundaries, verify_split_integrity
from wind_turbine_pm.health.config import load_health_config
from wind_turbine_pm.health.drift import (
    DriftCalibration,
    DriftSettings,
    MultivariateDriftDetector,
)
from wind_turbine_pm.health.evaluation import (
    band_error_table,
    plot_class_confusion,
    plot_error_by_band,
    plot_health_model_comparison,
    plot_predicted_vs_actual,
)
from wind_turbine_pm.health.health_class import ClassBands
from wind_turbine_pm.health.health_score import (
    SplitData,
    comparison_table,
    dataset_shape,
    evaluate_on_test,
    library_versions,
    predict_health_scores,
    select_best_health_candidate,
    to_split_data,
    train_health_candidates,
    training_timestamp,
)
from wind_turbine_pm.health.persistence import (
    health_background_path,
    health_comparison_path,
    health_metrics_path,
    save_health_bundle,
)
from wind_turbine_pm.logging_config import configure_from_config, get_logger
from wind_turbine_pm.models.persistence import figures_dir
from wind_turbine_pm.utils.io import read_json, write_json, write_table
from wind_turbine_pm.utils.paths import ensure_dir, resolve
from wind_turbine_pm.utils.reproducibility import set_global_seed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_health_data import boundaries_path, load_prepared_health  # noqa: E402

logger = get_logger("scripts.train_health_model")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; ``sys.argv[1:]`` when omitted.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip the evaluation figures (useful in CI smoke tests).",
    )
    return parser.parse_args(argv)


def _split_data(dataset: pd.DataFrame, features: pd.DataFrame, target: str) -> dict[str, SplitData]:
    """Slice the prepared data into train/valid/test split objects."""
    splits: dict[str, SplitData] = {}
    for name in (SplitName.TRAIN, SplitName.VALID, SplitName.TEST):
        subset = dataset.loc[dataset[SPLIT_COLUMN] == str(name)]
        if subset.empty:
            raise ValueError(f"Prepared health dataset has no rows in the {name} split")
        splits[str(name)] = to_split_data(subset, features, target, str(name))
    return splits


def main(argv: list[str] | None = None) -> int:
    """Run the health training step.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    cfg = load_health_config()
    configure_from_config(cfg)
    set_global_seed(int(cfg.get("random_seed", 42)))

    target = str(cfg.require("health.target.name"))
    dataset, features, spec = load_prepared_health(cfg)
    raw_boundaries = read_json(boundaries_path(cfg), hint="python scripts/prepare_health_data.py")
    boundaries = SplitBoundaries(
        train_end=pd.Timestamp(raw_boundaries["train_end"]),
        valid_start=pd.Timestamp(raw_boundaries["valid_start"]),
        valid_end=pd.Timestamp(raw_boundaries["valid_end"]),
        test_start=pd.Timestamp(raw_boundaries["test_start"]),
        embargo_hours=float(raw_boundaries["embargo_hours"]),
    )

    splits = _split_data(dataset, features, target)
    verify_split_integrity(
        dataset.loc[dataset[SPLIT_COLUMN] == str(SplitName.TRAIN)],
        dataset.loc[dataset[SPLIT_COLUMN] == str(SplitName.VALID)],
        dataset.loc[dataset[SPLIT_COLUMN] == str(SplitName.TEST)],
        boundaries,
    )
    train, valid, test = splits["train"], splits["valid"], splits["test"]
    logger.info(
        "Loaded health splits",
        extra={
            "train": f"{len(train.x)} rows / {train.n_degraded} degraded",
            "valid": f"{len(valid.x)} rows / {valid.n_degraded} degraded",
            "test": f"{len(test.x)} rows / {test.n_degraded} degraded",
            "features": features.shape[1],
        },
    )

    # --- Train and compare ------------------------------------------------
    candidates = train_health_candidates(train, valid, cfg)
    table = comparison_table(candidates)
    write_table(table, health_comparison_path(cfg))

    winner, rationale = select_best_health_candidate(candidates, cfg)
    logger.info("Health selection rationale: %s", rationale)

    # --- Test, reported once ---------------------------------------------
    test_metrics = evaluate_on_test(winner, test, cfg)
    all_metrics = {
        "train": winner.metrics["train"],
        "valid": winner.metrics["valid"],
        "test": test_metrics,
    }

    # --- Drift detectors (training split only) ----------------------------
    settings = DriftSettings.from_config(cfg)
    residual_columns = [
        column
        for column in features.columns
        if column.endswith("_drift_z") and column.removesuffix("_drift_z") in set(settings.sensors)
    ]
    detector: MultivariateDriftDetector | None = None
    if residual_columns:
        detector = MultivariateDriftDetector.fit(
            features.loc[train.x.index, residual_columns].astype(float), cfg
        )
    if detector is None:
        logger.warning(
            "No multivariate drift detector was fitted; per-channel CUSUM/EWMA detection is "
            "still active",
            extra={"residual_columns": len(residual_columns)},
        )

    # Persistence thresholds are calibrated against the *ground-truth* healthy
    # rows of the training split, so "drift" means "more crossings than a healthy
    # machine on this fleet" instead of a constant that assumes the residuals are
    # independent. See DriftCalibration for why that assumption fails here.
    healthy_min = float(cfg.get("health.drift.calibration.healthy_min_score", 95.0))
    train_rows = dataset.loc[dataset[SPLIT_COLUMN] == str(SplitName.TRAIN)]
    train_statistics = features.loc[train_rows.index].astype(float)
    # The multivariate threshold is calibrated from the detector's own scores on
    # the healthy training rows, so it too has a defined false-alarm rate rather
    # than being a constant on a squashed score.
    anomaly_scores = (
        detector.score(train_statistics)
        if detector is not None and all(c in train_statistics.columns for c in detector.columns)
        else None
    )
    calibration = DriftCalibration.fit(
        train_statistics,
        pd.to_numeric(train_rows[target], errors="coerce") >= healthy_min,
        cfg,
        anomaly_scores=anomaly_scores,
    )

    # --- Persist ----------------------------------------------------------
    ensure_dir(str(cfg.require("paths.artifacts_models")))
    background = train.x.sample(
        min(int(cfg.get("explainability.background_samples", 200)) * 2, len(train.x)),
        random_state=int(cfg.get("random_seed", 42)),
    )
    background.to_parquet(health_background_path(cfg), index=False)

    times = pd.to_datetime(dataset[TIMESTAMP])
    metadata = HealthModelMetadata(
        model_name=str(cfg.require("health.model.name")),
        model_version=str(cfg.require("health.model.version")),
        algorithm=winner.algorithm,
        module="turbine_health_monitoring",
        training_date=training_timestamp(),
        target=target,
        target_source=str(cfg.require("health.target.source_column")),
        features=list(features.columns),
        feature_groups=dict(spec.get("groups", {})),
        health_classes=ClassBands.from_config(cfg).to_contract(),
        drift={
            **settings.to_dict(),
            "detector": detector.to_dict() if detector is not None else None,
            "calibration": calibration.to_dict() if calibration is not None else None,
        },
        metrics=all_metrics,
        dataset=DatasetSummary(
            source=str(cfg.require("data.source")),
            is_synthetic=is_synthetic(cfg),
            n_turbines=int(dataset[TURBINE_ID].nunique()),
            n_rows=int(len(dataset)),
            n_features=int(features.shape[1]),
            time_start=times.min(),
            time_end=times.max(),
            # `positive_rate` is documented for a binary target; for a
            # regression score the share of degraded observations is the
            # comparable quantity and is what the field carries here.
            positive_rate=round(
                float(
                    (
                        pd.to_numeric(dataset[target], errors="coerce")
                        < ClassBands.from_config(cfg).monitor_min
                    ).mean()
                ),
                6,
            ),
            rows_train=len(train.x),
            rows_valid=len(valid.x),
            rows_test=len(test.x),
        ),
        selection_rationale=rationale,
        config_snapshot=cfg.to_dict(),
        library_versions=library_versions(),
    )
    save_health_bundle(winner.estimator, metadata, cfg, detector)

    # --- Figures and per-band error ---------------------------------------
    test_predictions = predict_health_scores(winner.estimator, test.x)
    band_table = band_error_table(test.y, test_predictions, cfg)
    write_table(
        band_table,
        resolve(str(cfg.require("paths.artifacts_metrics"))) / "health_error_by_band.csv",
    )

    if not args.skip_figures:
        figures = figures_dir(cfg)
        plot_health_model_comparison(table, figures / "health_model_comparison.png", split="valid")
        plot_predicted_vs_actual(
            test.y,
            test_predictions,
            cfg,
            figures / "health_predicted_vs_actual.png",
            "Health score: predicted vs actual (test split)",
        )
        plot_error_by_band(test.y, test_predictions, cfg, figures / "health_error_by_band.png")
        plot_class_confusion(test.y, test_predictions, cfg, figures / "health_class_confusion.png")

    write_json(
        {
            "model": winner.name,
            "algorithm": winner.algorithm,
            "model_version": metadata.model_version,
            "selection_rationale": rationale,
            "target": target,
            "target_source": metadata.target_source,
            "health_classes": ClassBands.from_config(cfg).to_dict(),
            "split_boundaries": boundaries.to_dict(),
            "metrics": all_metrics,
            "error_by_band": band_table.to_dict(orient="records"),
            "drift": metadata.drift,
            "train_seconds": round(winner.train_seconds, 3),
            "candidates": {
                candidate.name: {
                    "algorithm": candidate.algorithm,
                    "valid": candidate.metrics["valid"],
                    "train_seconds": round(candidate.train_seconds, 3),
                    "rejected_reason": candidate.rejected_reason or "",
                }
                for candidate in candidates
            },
            "dataset": dataset_shape(dataset),
            "is_synthetic": is_synthetic(cfg),
        },
        health_metrics_path(cfg),
    )

    print("\n=== Health candidate comparison (validation) ===")  # noqa: T201
    print(  # noqa: T201
        table.loc[table["split"] == "valid", ["model", "mae", "mae_degraded", "rmse", "spearman"]]
        .round(4)
        .to_string(index=False)
    )
    print(f"\nSelected: {winner.name} ({winner.algorithm})")  # noqa: T201
    print(  # noqa: T201
        "TEST  "
        + " ".join(
            f"{key}={test_metrics[key]:.4f}"
            for key in ("mae", "mae_degraded", "rmse", "spearman", "class_agreement")
        )
    )
    print("\n=== Test error by true health band ===")  # noqa: T201
    print(band_table.to_string(index=False))  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
