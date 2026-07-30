#!/usr/bin/env python
"""Prepare the shared SCADA history for Stage 3 anomaly detection."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from wind_turbine_pm.anomaly.config import load_anomaly_config
from wind_turbine_pm.anomaly.features import build_anomaly_features
from wind_turbine_pm.anomaly.persistence import dataset_path, feature_spec_path, features_path
from wind_turbine_pm.constants import (
    OPERATING_REGIME,
    SENSOR_COLUMNS,
    SPLIT_COLUMN,
    TIMESTAMP,
    TURBINE_ID,
)
from wind_turbine_pm.data.ingestion import load_raw_dataset
from wind_turbine_pm.data.preprocessing import preprocess
from wind_turbine_pm.data.splitting import assign_splits, compute_boundaries
from wind_turbine_pm.data.validation import validate_scada_frame
from wind_turbine_pm.health.regimes import attach_regimes
from wind_turbine_pm.logging_config import configure_from_config, get_logger
from wind_turbine_pm.utils.io import write_json, write_table
from wind_turbine_pm.utils.reproducibility import set_global_seed

logger = get_logger("scripts.prepare_anomaly_data")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_anomaly_config()
    configure_from_config(cfg)
    set_global_seed(int(cfg.get("random_seed", 42)))
    if features_path(cfg).is_file() and not args.force:
        logger.info("Prepared anomaly data already exists; use --force to rebuild")
        return 0

    raw = load_raw_dataset(cfg)
    report = validate_scada_frame(raw, cfg)
    if args.strict:
        report.raise_for_errors()
    clean, _ = preprocess(raw, cfg)
    regimed = attach_regimes(clean, cfg)
    features, spec = build_anomaly_features(regimed, cfg)

    excluded_states = set(cfg.require("anomaly.target.exclude_states"))
    eligible = ~regimed["operational_status"].astype(str).isin(excluded_states)
    regimed = regimed.loc[eligible].copy()
    features = features.loc[regimed.index]
    boundaries = compute_boundaries(regimed, cfg)
    regimed[SPLIT_COLUMN] = assign_splits(regimed, boundaries)

    truth_column = str(cfg.require("anomaly.target.evaluation_column"))
    positive_from = float(cfg.require("anomaly.target.positive_from"))
    healthy_max = float(cfg.require("anomaly.target.healthy_max"))
    severity = pd.to_numeric(regimed[truth_column], errors="coerce").fillna(0.0)
    regimed["anomaly_truth"] = (severity >= positive_from).astype("int8")
    regimed["anomaly_healthy_reference"] = (severity <= healthy_max).astype("int8")

    metadata = [
        TURBINE_ID,
        TIMESTAMP,
        OPERATING_REGIME,
        SPLIT_COLUMN,
        "operational_status",
        "anomaly_truth",
        "anomaly_healthy_reference",
        truth_column,
        "failure_event",
        "maintenance_event",
        "failure_mode",
        "episode_id",
        *SENSOR_COLUMNS,
    ]
    dataset = regimed[[name for name in metadata if name in regimed]].reset_index(drop=True)
    features = features.reset_index(drop=True)
    write_table(dataset, dataset_path(cfg))
    write_table(features, features_path(cfg))
    write_json(
        {
            **spec.to_dict(),
            "evaluation_truth": f"{truth_column} >= {positive_from}",
            "healthy_reference": f"{truth_column} <= {healthy_max}",
            "split": boundaries.to_dict(),
        },
        feature_spec_path(cfg),
    )
    logger.info(
        "Anomaly preparation complete",
        extra={
            "rows": len(dataset),
            "features": features.shape[1],
            "positive_rate": float(dataset["anomaly_truth"].mean()),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
