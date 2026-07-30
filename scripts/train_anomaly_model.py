#!/usr/bin/env python
"""Train, calibrate, compare and publish Stage 3 anomaly models."""

from __future__ import annotations

import sys
from datetime import UTC, datetime

import pandas as pd

from wind_turbine_pm.anomaly.config import load_anomaly_config
from wind_turbine_pm.anomaly.modeling import (
    anomaly_metrics,
    deterministic_reference_sample,
    percentile_scores,
    raw_novelty_score,
    select_best,
    train_candidates,
)
from wind_turbine_pm.anomaly.persistence import (
    comparison_path,
    dataset_path,
    feature_spec_path,
    features_path,
    metrics_path,
    save_bundle,
)
from wind_turbine_pm.constants import SPLIT_COLUMN, TIMESTAMP, TURBINE_ID, SplitName
from wind_turbine_pm.contracts.anomaly import AnomalyModelMetadata
from wind_turbine_pm.data.ingestion import is_synthetic
from wind_turbine_pm.health.health_score import library_versions
from wind_turbine_pm.logging_config import configure_from_config, get_logger
from wind_turbine_pm.utils.io import read_json, read_table, write_json, write_table
from wind_turbine_pm.utils.reproducibility import set_global_seed

logger = get_logger("scripts.train_anomaly_model")


def main(argv: list[str] | None = None) -> int:
    del argv
    cfg = load_anomaly_config()
    configure_from_config(cfg)
    seed = int(cfg.get("random_seed", 42))
    set_global_seed(seed)

    dataset = read_table(dataset_path(cfg), hint="python scripts/prepare_anomaly_data.py")
    features = read_table(features_path(cfg), hint="python scripts/prepare_anomaly_data.py")
    spec = read_json(feature_spec_path(cfg), hint="python scripts/prepare_anomaly_data.py")
    if len(dataset) != len(features):
        raise ValueError("Anomaly dataset and features are not row-aligned")

    train_mask = dataset[SPLIT_COLUMN] == str(SplitName.TRAIN)
    valid_mask = dataset[SPLIT_COLUMN] == str(SplitName.VALID)
    test_mask = dataset[SPLIT_COLUMN] == str(SplitName.TEST)
    healthy_train = train_mask & dataset["anomaly_healthy_reference"].astype(bool)
    if not healthy_train.any() or not valid_mask.any() or not test_mask.any():
        raise ValueError("Anomaly preparation did not produce usable train/valid/test rows")

    train_reference = deterministic_reference_sample(
        features.loc[healthy_train],
        dataset.loc[healthy_train],
        maximum=int(cfg.require("anomaly.training.max_reference_rows")),
        seed=seed,
    )
    valid_x = features.loc[valid_mask]
    valid_y = dataset.loc[valid_mask, "anomaly_truth"].to_numpy(dtype=int)
    valid_healthy = dataset.loc[valid_mask, "anomaly_healthy_reference"].to_numpy(dtype=bool)
    results = train_candidates(train_reference, valid_x, valid_y, valid_healthy, cfg)
    winner, rationale = select_best(results)

    test_x = features.loc[test_mask]
    test_y = dataset.loc[test_mask, "anomaly_truth"].to_numpy(dtype=int)
    test_scores = percentile_scores(raw_novelty_score(winner.estimator, test_x), winner.calibration)
    test_metrics = anomaly_metrics(test_y, test_scores, winner.calibration)
    all_metrics = {"valid": winner.metrics, "test": test_metrics}

    comparison = pd.DataFrame(
        [
            {
                "model": item.name,
                "algorithm": item.algorithm,
                **item.metrics,
                "latency_ms": item.latency_ms,
                "selected": item.name == winner.name,
            }
            for item in results
        ]
    )
    write_table(comparison, comparison_path(cfg))

    reference = pd.DataFrame(
        [
            {"_statistic": "median", **train_reference.median(numeric_only=True).to_dict()},
            {"_statistic": "q1", **train_reference.quantile(0.25).to_dict()},
            {"_statistic": "q3", **train_reference.quantile(0.75).to_dict()},
        ]
    )
    times = pd.to_datetime(dataset[TIMESTAMP])
    metadata = AnomalyModelMetadata(
        model_version=str(cfg.require("anomaly.version")),
        algorithm=winner.algorithm,
        training_date=datetime.now(UTC),
        features=list(features.columns),
        feature_groups=dict(spec.get("groups", {})),
        metrics=all_metrics,
        selection_rationale=rationale,
        dataset={
            "source": str(cfg.require("data.source")),
            "is_synthetic": is_synthetic(cfg),
            "n_turbines": int(dataset[TURBINE_ID].nunique()),
            "n_rows": int(len(dataset)),
            "n_features": int(features.shape[1]),
            "time_start": times.min(),
            "time_end": times.max(),
            "positive_rate": float(dataset["anomaly_truth"].mean()),
            "rows_train_reference": int(len(train_reference)),
            "rows_valid": int(valid_mask.sum()),
            "rows_test": int(test_mask.sum()),
        },
        config_snapshot={
            "anomaly": cfg.require("anomaly").to_dict(),
            "split": cfg.require("split").to_dict(),
        },
        library_versions=library_versions(),
    )
    save_bundle(winner.estimator, metadata, winner.calibration, reference, cfg)
    write_json(
        {
            "model": winner.name,
            "algorithm": winner.algorithm,
            "model_version": metadata.model_version,
            "selection_rationale": rationale,
            "calibration": winner.calibration.model_dump(mode="json"),
            "metrics": all_metrics,
            "candidates": comparison.to_dict(orient="records"),
            "dataset": metadata.dataset,
            "is_synthetic": is_synthetic(cfg),
        },
        metrics_path(cfg),
    )
    logger.info("Published anomaly model", extra={"model": winner.name, **test_metrics})
    print(comparison.round(4).to_string(index=False))  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
