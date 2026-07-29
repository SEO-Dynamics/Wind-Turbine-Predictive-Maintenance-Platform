"""Tests for the health target, metrics and model selection.

The selection guard rails are the part worth testing hardest: they are what stop
a model that cannot order turbines correctly, or one no better than predicting
the mean, from being published.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import OPERATIONAL_STATUS, TIMESTAMP, TURBINE_ID
from wind_turbine_pm.health.health_class import ClassBands
from wind_turbine_pm.health.health_score import (
    HealthCandidate,
    HealthTargetError,
    SplitData,
    apply_label_filter,
    build_health_target,
    class_agreement,
    comparison_table,
    dataset_shape,
    evaluate_on_test,
    library_versions,
    predict_health_scores,
    regression_metrics,
    select_best_health_candidate,
    to_split_data,
    train_health_candidates,
    training_timestamp,
)


def _candidate(name: str, valid: dict[str, float]) -> HealthCandidate:
    return HealthCandidate(
        name=name,
        algorithm=name,
        estimator=None,
        metrics={"train": dict(valid), "valid": dict(valid)},
    )


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
def test_target_inverts_the_degradation_state(
    small_health_config: Config, health_toy_frame
) -> None:
    out = build_health_target(health_toy_frame, small_health_config)
    target = str(small_health_config.require("health.target.name"))

    assert target in out.columns
    # 100 is as-new, 0 is the worst state in the record.
    zero_degradation = out.loc[health_toy_frame["degradation_level"] == 0.0, target]
    assert (zero_degradation == 100.0).all()
    full = out.loc[health_toy_frame["degradation_level"] >= 1.0, target]
    if not full.empty:
        assert (full == 0.0).all()
    assert out[target].between(0.0, 100.0).all()


def test_target_is_monotonically_decreasing_in_degradation(
    small_health_config: Config, health_toy_frame
) -> None:
    out = build_health_target(health_toy_frame, small_health_config)
    target = str(small_health_config.require("health.target.name"))
    ordered = out.sort_values("degradation_level")
    assert (ordered[target].diff().dropna() <= 1e-9).all()


def test_absent_ground_truth_raises_with_the_instruction_to_supply_one(
    small_health_config: Config, health_toy_frame
) -> None:
    frame = health_toy_frame.drop(columns=["degradation_level"])
    with pytest.raises(HealthTargetError, match="independent of the SCADA feature channels"):
        build_health_target(frame, small_health_config)


def test_out_of_range_ground_truth_is_rejected(
    small_health_config: Config, health_toy_frame
) -> None:
    frame = health_toy_frame.copy()
    frame["degradation_level"] = frame["degradation_level"] * 5.0
    with pytest.raises(HealthTargetError, match=r"must lie in \[0, 1\]"):
        build_health_target(frame, small_health_config)


def test_excluded_operating_states_are_marked_ineligible_not_dropped(
    small_health_config: Config, health_toy_frame
) -> None:
    frame = health_toy_frame.copy()
    frame.loc[frame.index[:10], OPERATIONAL_STATUS] = "maintenance"
    out = build_health_target(frame, small_health_config)

    # Rows are kept so the caller retains the full history for feature building.
    assert len(out) == len(frame)
    assert not out.loc[out.index[:10], "health_label_eligible"].any()

    filtered = apply_label_filter(out)
    assert len(filtered) == int(out["health_label_eligible"].sum())
    assert "health_label_eligible" not in filtered.columns


def test_label_filter_requires_the_target_step_first() -> None:
    with pytest.raises(KeyError, match="build_health_target"):
        apply_label_filter(pd.DataFrame({"a": [1]}))


# ---------------------------------------------------------------------------
# Split slicing
# ---------------------------------------------------------------------------
def test_to_split_data_aligns_features_and_target(health_prepared) -> None:
    labelled, features, _ = health_prepared
    subset = labelled.iloc[:200]
    split = to_split_data(subset, features, "health_score_target", "train")

    assert split.name == "train"
    assert len(split.x) == len(subset)
    assert split.x.index.equals(subset.index)
    assert split.y.shape == (len(subset),)
    assert split.n_degraded >= 0


def test_to_split_data_rejects_a_missing_target_or_empty_split(health_prepared) -> None:
    labelled, features, _ = health_prepared
    with pytest.raises(ValueError, match="no target column"):
        to_split_data(labelled.iloc[:10], features, "not_a_column", "train")
    with pytest.raises(ValueError, match="empty"):
        to_split_data(labelled.iloc[:0], features, "health_score_target", "train")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_perfect_prediction_scores_perfectly() -> None:
    y = np.array([100.0, 80.0, 50.0, 20.0])
    metrics = regression_metrics(y, y.copy(), degraded_boundary=60.0)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["mae_degraded"] == 0.0
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["within_5_points"] == 1.0


def test_mae_degraded_isolates_the_part_that_has_to_be_right() -> None:
    # Overall MAE is dominated by the healthy majority; a model can win it while
    # being useless exactly where the score has to be trusted. This is why
    # mae_degraded is the primary selection metric.
    y_true = np.array([100.0] * 99 + [30.0])
    y_pred = np.array([100.0] * 99 + [90.0])
    metrics = regression_metrics(y_true, y_pred, degraded_boundary=60.0)

    assert metrics["mae"] == pytest.approx(0.6)
    assert metrics["mae_degraded"] == pytest.approx(60.0)
    assert metrics["n_degraded"] == 1


def test_bias_reports_direction_of_error() -> None:
    y_true = np.array([50.0, 50.0])
    metrics = regression_metrics(y_true, np.array([60.0, 60.0]), degraded_boundary=60.0)
    assert metrics["bias"] == pytest.approx(10.0)


def test_metrics_reject_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        regression_metrics(np.zeros(3), np.zeros(4), degraded_boundary=60.0)


def test_no_degraded_observations_yields_zero_rather_than_nan() -> None:
    y = np.array([100.0, 95.0])
    metrics = regression_metrics(y, y.copy(), degraded_boundary=60.0)
    assert metrics["mae_degraded"] == 0.0
    assert metrics["n_degraded"] == 0


def test_class_agreement_measures_the_decision_not_the_error(
    health_config: Config,
) -> None:
    bands = ClassBands.from_config(health_config)
    # A three-point error that straddles a boundary is a wrong decision; the same
    # error inside a band is not.
    straddling = class_agreement(
        np.array([bands.healthy_min + 1.0]), np.array([bands.healthy_min - 2.0]), health_config
    )
    inside = class_agreement(
        np.array([bands.healthy_min + 5.0]), np.array([bands.healthy_min + 8.0]), health_config
    )
    assert straddling["class_agreement"] == 0.0
    assert inside["class_agreement"] == 1.0


def test_optimistic_rate_counts_only_the_dangerous_direction(health_config: Config) -> None:
    # Predicting healthier than the truth is the costly error.
    optimistic = class_agreement(np.array([30.0]), np.array([95.0]), health_config)
    pessimistic = class_agreement(np.array([95.0]), np.array([30.0]), health_config)
    assert optimistic["class_optimistic_rate"] == 1.0
    assert pessimistic["class_optimistic_rate"] == 0.0


def test_predictions_are_clamped_to_the_published_range() -> None:
    class _Wild:
        def predict(self, features):
            return np.array([-500.0, 50.0, 900.0][: len(features)])

    scores = predict_health_scores(_Wild(), pd.DataFrame(index=range(3)))
    assert scores.min() >= 0.0
    assert scores.max() <= 100.0


# ---------------------------------------------------------------------------
# Selection guard rails
# ---------------------------------------------------------------------------
def test_baseline_is_never_published(small_health_config: Config) -> None:
    candidates = [
        _candidate(
            "mean_baseline", {"mae": 5.0, "mae_degraded": 5.0, "rmse": 5.0, "spearman": 0.9}
        ),
        _candidate("ridge", {"mae": 3.0, "mae_degraded": 3.0, "rmse": 3.0, "spearman": 0.9}),
    ]
    winner, rationale = select_best_health_candidate(candidates, small_health_config)
    assert winner.name == "ridge"
    assert "mean_baseline" in rationale


def test_a_model_that_cannot_order_turbines_is_rejected(small_health_config: Config) -> None:
    # A score that does not rank turbines correctly cannot prioritise maintenance,
    # whatever its MAE.
    floor = float(small_health_config.get("health.training.selection.min_spearman", 0.5))
    candidates = [
        _candidate(
            "mean_baseline", {"mae": 20.0, "mae_degraded": 20.0, "rmse": 20.0, "spearman": 0.9}
        ),
        _candidate(
            "ridge", {"mae": 1.0, "mae_degraded": 1.0, "rmse": 1.0, "spearman": floor - 0.2}
        ),
    ]
    with pytest.raises(ValueError, match="Every health candidate was rejected"):
        select_best_health_candidate(candidates, small_health_config)


def test_a_model_no_better_than_the_baseline_is_rejected(small_health_config: Config) -> None:
    ratio = float(
        small_health_config.get("health.training.selection.max_mae_ratio_vs_baseline", 0.85)
    )
    candidates = [
        _candidate(
            "mean_baseline", {"mae": 10.0, "mae_degraded": 10.0, "rmse": 10.0, "spearman": 0.9}
        ),
        _candidate(
            "ridge",
            {"mae": 10.0 * ratio + 0.5, "mae_degraded": 1.0, "rmse": 1.0, "spearman": 0.95},
        ),
    ]
    with pytest.raises(ValueError, match="not better than"):
        select_best_health_candidate(candidates, small_health_config)


def test_nan_rank_correlation_is_rejected(small_health_config: Config) -> None:
    candidates = [
        _candidate(
            "ridge",
            {"mae": 1.0, "mae_degraded": 1.0, "rmse": 1.0, "spearman": float("nan")},
        )
    ]
    with pytest.raises(ValueError, match="Every health candidate was rejected"):
        select_best_health_candidate(candidates, small_health_config)


def test_ties_within_tolerance_are_broken_by_the_configured_metrics(
    small_health_config: Config,
) -> None:
    data = small_health_config.to_dict()
    data["health"]["training"]["selection"]["primary_tolerance"] = 0.20
    data["health"]["training"]["selection"]["tie_breakers"] = ["mae"]
    cfg = Config(data)

    candidates = [
        _candidate(
            "mean_baseline", {"mae": 20.0, "mae_degraded": 20.0, "rmse": 20.0, "spearman": 0.9}
        ),
        # Slightly worse on the primary metric but clearly better on the
        # tie-breaker, and inside the tolerance band.
        _candidate("ridge", {"mae": 2.0, "mae_degraded": 5.2, "rmse": 3.0, "spearman": 0.9}),
        _candidate("forest", {"mae": 8.0, "mae_degraded": 5.0, "rmse": 9.0, "spearman": 0.9}),
    ]
    winner, _ = select_best_health_candidate(candidates, cfg)
    assert winner.name == "ridge"


def test_rationale_names_the_winner_and_every_rejection(small_health_config: Config) -> None:
    candidates = [
        _candidate(
            "mean_baseline", {"mae": 20.0, "mae_degraded": 20.0, "rmse": 20.0, "spearman": 0.9}
        ),
        _candidate("ridge", {"mae": 3.0, "mae_degraded": 3.0, "rmse": 3.0, "spearman": 0.9}),
        _candidate("bad", {"mae": 3.0, "mae_degraded": 3.0, "rmse": 3.0, "spearman": 0.1}),
    ]
    _, rationale = select_best_health_candidate(candidates, small_health_config)
    assert "ridge" in rationale
    assert "bad" in rationale
    assert "Spearman" in rationale


def test_comparison_table_is_tidy_and_ranked() -> None:
    candidates = [
        _candidate("worse", {"mae": 9.0, "mae_degraded": 9.0, "rmse": 9.0, "spearman": 0.5}),
        _candidate("better", {"mae": 1.0, "mae_degraded": 1.0, "rmse": 1.0, "spearman": 0.9}),
    ]
    table = comparison_table(candidates)
    assert {"model", "algorithm", "split", "mae", "mae_degraded"} <= set(table.columns)
    # Best validation mae_degraded first.
    assert table["model"].iloc[0] == "better"


def test_comparison_table_of_nothing_is_empty() -> None:
    assert comparison_table([]).empty


# ---------------------------------------------------------------------------
# End-to-end training on the fixture
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_training_selects_a_model_that_beats_the_baseline(
    small_health_config: Config, health_prepared
) -> None:
    labelled, features, eligible = health_prepared
    features = features.loc[eligible.index]

    midpoint = len(eligible) // 2
    train = to_split_data(eligible.iloc[:midpoint], features, "health_score_target", "train")
    valid = to_split_data(eligible.iloc[midpoint:], features, "health_score_target", "valid")

    data = small_health_config.to_dict()
    # Two fast candidates are enough to exercise the training and selection path.
    data["health"]["training"]["candidates"] = {
        "mean_baseline": {"enabled": True, "params": {"strategy": "mean"}},
        "ridge": {"enabled": True, "params": {"alpha": 1.0}},
    }
    cfg = Config(data)

    candidates = train_health_candidates(train, valid, cfg)
    assert {c.name for c in candidates} == {"mean_baseline", "ridge"}

    winner, _ = select_best_health_candidate(candidates, cfg)
    baseline = next(c for c in candidates if c.name == "mean_baseline")
    assert winner.name == "ridge"
    assert winner.metrics["valid"]["mae"] < baseline.metrics["valid"]["mae"]

    test_metrics = evaluate_on_test(winner, valid, cfg)
    assert 0.0 <= test_metrics["class_agreement"] <= 1.0
    assert test_metrics["n_samples"] == len(valid.y)


def test_no_enabled_candidate_is_an_error(small_health_config: Config, health_prepared) -> None:
    labelled, features, eligible = health_prepared
    features = features.loc[eligible.index]
    split = to_split_data(eligible.iloc[:50], features, "health_score_target", "train")

    data = small_health_config.to_dict()
    data["health"]["training"]["candidates"] = {"ridge": {"enabled": False, "params": {}}}
    with pytest.raises(ValueError, match="No health candidates are enabled"):
        train_health_candidates(split, split, Config(data))


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------
def test_library_versions_record_the_artifact_critical_packages() -> None:
    versions = library_versions()
    assert {"scikit-learn", "numpy", "joblib"} <= set(versions)
    assert all(isinstance(value, str) and value for value in versions.values())


def test_training_timestamp_is_timezone_aware() -> None:
    assert training_timestamp().tzinfo is not None


def test_dataset_shape_summarises_rows_and_turbines(health_prepared) -> None:
    labelled, _, _ = health_prepared
    shape = dataset_shape(labelled)
    assert shape["rows"] == len(labelled)
    assert shape["turbines"] == labelled[TURBINE_ID].nunique()
    assert dataset_shape(pd.DataFrame({TIMESTAMP: []}))["turbines"] == 0


def test_split_data_counts_degraded_observations() -> None:
    split = SplitData(x=pd.DataFrame(index=range(3)), y=np.array([100.0, 50.0, 10.0]), name="t")
    assert split.n_degraded == 2
