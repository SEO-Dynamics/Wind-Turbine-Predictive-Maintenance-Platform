#!/usr/bin/env python
"""Train candidate models, optimise the threshold and publish the final bundle.

Information flow is strictly one-directional:
train -> fit, valid -> rank + threshold, test -> reported once, never fed back.

Usage:
    python scripts/train_failure_model.py [--threshold-method f2|cost]
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from wind_turbine_pm.config import Config, load_config
from wind_turbine_pm.constants import SPLIT_COLUMN, TARGET_COLUMN, TIMESTAMP, SplitName
from wind_turbine_pm.contracts.metadata import (
    DatasetSummary,
    ModelMetadata,
    RiskLevelBands,
    ThresholdInfo,
)
from wind_turbine_pm.data.ingestion import is_synthetic
from wind_turbine_pm.data.splitting import SplitBoundaries, verify_split_integrity
from wind_turbine_pm.logging_config import configure_from_config, get_logger
from wind_turbine_pm.models.evaluation import plot_model_comparison
from wind_turbine_pm.models.persistence import (
    comparison_path,
    figures_dir,
    metrics_path,
    save_bundle,
)
from wind_turbine_pm.models.threshold import optimise_threshold, plot_threshold_curve
from wind_turbine_pm.models.training import (
    SplitData,
    comparison_table,
    evaluate_on_test,
    library_versions,
    select_best_candidate,
    to_split_data,
    train_candidates,
    training_timestamp,
)
from wind_turbine_pm.services.failure_prediction_service import background_path
from wind_turbine_pm.utils.io import read_json, write_json, write_table
from wind_turbine_pm.utils.paths import ensure_dir
from wind_turbine_pm.utils.reproducibility import set_global_seed

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from prepare_data import BOUNDARIES_FILENAME, load_prepared, processed_dir  # noqa: E402

logger = get_logger("scripts.train_failure_model")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; ``sys.argv[1:]`` when omitted.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold-method",
        choices=["f2", "cost"],
        default=None,
        help="Override threshold.method from configuration.",
    )
    return parser.parse_args(argv)


def _split_frames(dataset: pd.DataFrame, features: pd.DataFrame) -> dict[str, SplitData]:
    """Slice the prepared data into train/valid/test split objects."""
    splits: dict[str, SplitData] = {}
    for name in (SplitName.TRAIN, SplitName.VALID, SplitName.TEST):
        subset = dataset.loc[dataset[SPLIT_COLUMN] == str(name)]
        if subset.empty:
            raise ValueError(f"Prepared dataset has no rows in the {name} split")
        splits[str(name)] = to_split_data(subset, features, TARGET_COLUMN)
    return splits


def main(argv: list[str] | None = None) -> int:
    """Run the training step.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    cfg = load_config()
    configure_from_config(cfg)
    set_global_seed(int(cfg.get("random_seed", 42)))

    dataset, features, spec = load_prepared(cfg)
    boundaries_raw = read_json(
        processed_dir(cfg) / BOUNDARIES_FILENAME, hint="python scripts/prepare_data.py"
    )
    boundaries = SplitBoundaries(
        train_end=pd.Timestamp(boundaries_raw["train_end"]),
        valid_start=pd.Timestamp(boundaries_raw["valid_start"]),
        valid_end=pd.Timestamp(boundaries_raw["valid_end"]),
        test_start=pd.Timestamp(boundaries_raw["test_start"]),
        embargo_hours=float(boundaries_raw["embargo_hours"]),
    )

    splits = _split_frames(dataset, features)
    verify_split_integrity(
        dataset.loc[dataset[SPLIT_COLUMN] == str(SplitName.TRAIN)],
        dataset.loc[dataset[SPLIT_COLUMN] == str(SplitName.VALID)],
        dataset.loc[dataset[SPLIT_COLUMN] == str(SplitName.TEST)],
        boundaries,
    )
    train, valid, test = splits["train"], splits["valid"], splits["test"]
    logger.info(
        "Loaded splits",
        extra={
            "train": f"{len(train.x)} rows / {train.n_positive} positive",
            "valid": f"{len(valid.x)} rows / {valid.n_positive} positive",
            "test": f"{len(test.x)} rows / {test.n_positive} positive",
            "features": features.shape[1],
        },
    )

    # --- Train and compare ------------------------------------------------
    candidates = train_candidates(train, valid, cfg)
    table = comparison_table(candidates)
    write_table(table, comparison_path(cfg))
    plot_model_comparison(table, figures_dir(cfg) / "model_comparison.png", split="valid")

    winner, rationale = select_best_candidate(candidates, cfg)
    logger.info("Selection rationale: %s", rationale)

    # --- Threshold (validation only) --------------------------------------
    threshold_result = optimise_threshold(
        valid.y,
        winner.valid_probabilities,
        cfg,
        method=args.threshold_method,
        split_name="valid",
    )
    plot_threshold_curve(threshold_result, figures_dir(cfg) / "threshold_curve.png")
    write_table(threshold_result.curve, figures_dir(cfg).parent / "metrics" / "threshold_curve.csv")
    winner.threshold = threshold_result

    # --- Test, reported once ---------------------------------------------
    test_metrics = evaluate_on_test(winner, test)
    all_metrics = {
        "train": winner.metrics["train"],
        "valid": threshold_result.metrics,
        "test": test_metrics,
    }

    # --- Persist ----------------------------------------------------------
    background = train.x.sample(
        min(int(cfg.get("explainability.background_samples", 200)) * 2, len(train.x)),
        random_state=int(cfg.get("random_seed", 42)),
    )
    ensure_dir(str(cfg.require("paths.artifacts_models")))
    background.to_parquet(background_path(cfg), index=False)

    times = pd.to_datetime(dataset[TIMESTAMP])
    metadata = ModelMetadata(
        model_name=str(cfg.require("model.name")),
        model_version=str(cfg.require("model.version")),
        algorithm=winner.algorithm,
        module="failure_prediction",
        training_date=training_timestamp(),
        target=TARGET_COLUMN,
        horizon_hours=int(cfg.require("target.horizon_hours")),
        features=list(features.columns),
        feature_groups=dict(spec.get("groups", {})),
        threshold=ThresholdInfo(
            value=threshold_result.threshold,
            method=threshold_result.method,
            selected_on=threshold_result.selected_on,
            costs=threshold_result.costs,
            validation_metrics={
                key: float(value)
                for key, value in threshold_result.metrics.items()
                if isinstance(value, int | float)
            },
            alternatives=threshold_result.alternatives,
        ),
        risk_levels=RiskLevelBands(
            low_max=float(cfg.require("risk_levels.low_max")),
            medium_max=float(cfg.require("risk_levels.medium_max")),
        ),
        metrics=all_metrics,
        dataset=DatasetSummary(
            source=str(cfg.require("data.source")),
            is_synthetic=is_synthetic(cfg),
            n_turbines=int(dataset["turbine_id"].nunique()),
            n_rows=int(len(dataset)),
            n_features=int(features.shape[1]),
            time_start=times.min(),
            time_end=times.max(),
            positive_rate=float(dataset[TARGET_COLUMN].mean()),
            rows_train=len(train.x),
            rows_valid=len(valid.x),
            rows_test=len(test.x),
        ),
        selection_rationale=rationale,
        config_snapshot=cfg.to_dict(),
        library_versions=library_versions(),
    )
    save_bundle(winner.estimator, metadata, cfg)

    write_json(
        {
            "model": winner.name,
            "algorithm": winner.algorithm,
            "model_version": metadata.model_version,
            "selection_rationale": rationale,
            "threshold": threshold_result.to_dict(),
            "split_boundaries": boundaries.to_dict(),
            "metrics": all_metrics,
            "latency_ms": winner.latency,
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
            "is_synthetic": is_synthetic(cfg),
        },
        metrics_path(cfg),
    )

    print("\n=== Validation comparison ===")  # noqa: T201
    print(  # noqa: T201
        table.loc[
            table["split"] == "valid", ["model", "pr_auc", "roc_auc", "recall", "precision", "f2"]
        ]
        .round(4)
        .to_string(index=False)
    )
    print(f"\nSelected: {winner.name} | threshold {threshold_result.threshold:.4f}")  # noqa: T201
    print(  # noqa: T201
        "TEST  "
        + " ".join(
            f"{key}={test_metrics[key]:.4f}"
            for key in (
                "pr_auc",
                "roc_auc",
                "recall",
                "precision",
                "f1",
                "f2",
                "false_negative_rate",
            )
        )
    )
    return 0


def load_config_for_scripts() -> Config:
    """Load configuration for use by other scripts.

    Returns:
        The merged configuration.
    """
    return load_config()


if __name__ == "__main__":
    sys.exit(main())
