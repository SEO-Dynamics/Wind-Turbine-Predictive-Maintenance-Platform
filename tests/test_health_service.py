"""Tests for the health monitoring service.

Assessment tests need a published health bundle and are skipped when artifacts
are absent; the artifact-missing behaviour itself is tested explicitly with an
isolated service pointed at an empty directory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import (
    HEALTH_CLASS,
    HEALTH_SCORE,
    OPERATING_REGIME,
    TIMESTAMP,
    TURBINE_ID,
    HealthClass,
)
from wind_turbine_pm.contracts.health import HealthAssessment
from wind_turbine_pm.contracts.observations import frame_to_windows
from wind_turbine_pm.health.config import load_health_config
from wind_turbine_pm.health.persistence import health_bundle_available
from wind_turbine_pm.services.failure_prediction_service import (
    InsufficientHistoryError,
    ServiceNotReadyError,
)
from wind_turbine_pm.services.health_monitoring_service import (
    HealthMonitoringService,
    latest_per_turbine,
)

needs_artifacts = pytest.mark.skipif(
    not health_bundle_available(load_health_config()),
    reason="Health artifacts not built; run python scripts/run_health_pipeline.py",
)


@pytest.fixture(scope="module")
def service() -> HealthMonitoringService:
    """A service bound to the shipped configuration and published artifacts."""
    return HealthMonitoringService()


@pytest.fixture(scope="module")
def published(service: HealthMonitoringService):
    """The prepared dataset and features that the published artifact was built on.

    The ``health_prepared`` fixture deliberately uses a narrowed configuration so
    the suite stays fast, which means its feature matrix does not match the
    published model's feature contract.  Tests that exercise the real artifact
    have to be fed the real prepared data.

    Returns:
        ``(dataset, features)``.
    """
    from wind_turbine_pm.health.persistence import health_dataset_path, health_features_path
    from wind_turbine_pm.utils.io import ArtifactNotFoundError, read_table

    try:
        dataset = read_table(health_dataset_path(service.config))
        features = read_table(health_features_path(service.config))
    except ArtifactNotFoundError:
        pytest.skip("Prepared health data not built; run python scripts/run_health_pipeline.py")
    return dataset, features


@pytest.fixture
def unbuilt_service(tmp_path) -> HealthMonitoringService:
    """A service whose artifact directories are empty."""
    data = load_health_config().to_dict()
    data["paths"]["artifacts_models"] = str(tmp_path / "models")
    data["paths"]["artifacts_metadata"] = str(tmp_path / "metadata")
    return HealthMonitoringService(Config(data))


# ---------------------------------------------------------------------------
# Degraded mode
# ---------------------------------------------------------------------------
def test_missing_artifacts_report_not_ready_rather_than_raising(
    unbuilt_service: HealthMonitoringService,
) -> None:
    assert unbuilt_service.is_ready is False
    status = unbuilt_service.status()
    assert status.model_loaded is False
    assert status.model_version is None
    assert "run_health_pipeline" in status.detail
    # The status document must stay serialisable so /health can return it.
    assert unbuilt_service.status().to_dict()["model_loaded"] is False


def test_assessing_without_artifacts_raises_with_the_fixing_command(
    unbuilt_service: HealthMonitoringService, health_toy_frame
) -> None:
    window = frame_to_windows(health_toy_frame.loc[health_toy_frame[TURBINE_ID] == "A"])[0]
    with pytest.raises(ServiceNotReadyError, match="run_health_pipeline"):
        unbuilt_service.assess_from_window(window)


def test_configuration_dependent_accessors_work_without_artifacts(
    unbuilt_service: HealthMonitoringService,
) -> None:
    # Rules, class bands and the history requirement are configuration, so they
    # must be available for the API to document itself before training.
    assert unbuilt_service.rules
    assert unbuilt_service.sensor_rule_records()
    assert unbuilt_service.class_bands.healthy_min > 0
    assert unbuilt_service.minimum_history_hours() > 0
    assert unbuilt_service.drift_calibration is None


def test_minimum_history_is_never_below_the_longest_feature_window() -> None:
    from wind_turbine_pm.health.health_features import minimum_history_hours

    data = load_health_config().to_dict()
    data["health"]["serving"]["min_history_hours"] = 1  # deliberately too small
    service = HealthMonitoringService(Config(data))
    # A window that leaves a configured feature undefined must never be accepted.
    assert service.minimum_history_hours() >= minimum_history_hours(service.config)


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------
@needs_artifacts
def test_short_window_is_rejected(service: HealthMonitoringService, health_toy_frame) -> None:
    subset = health_toy_frame.loc[health_toy_frame[TURBINE_ID] == "A"].head(4)
    with pytest.raises(InsufficientHistoryError, match="history is required"):
        service.assess_from_window(frame_to_windows(subset)[0])


@needs_artifacts
def test_assessment_satisfies_the_published_contract(
    service: HealthMonitoringService, published
) -> None:
    labelled, features = published
    turbine = str(labelled[TURBINE_ID].iloc[0])
    assessment = service.assess_from_prepared(labelled, features, turbine_id=turbine)

    assert isinstance(assessment, HealthAssessment)
    assert 0.0 <= assessment.health_score <= 100.0
    assert 0.0 <= assessment.raw_health_score <= 100.0
    assert assessment.drift_penalty >= 0.0
    assert 0.0 <= assessment.data_quality <= 1.0
    assert assessment.health_class in set(HealthClass)
    assert assessment.advisory_only is True
    assert assessment.explanation.strip()
    assert assessment.recommendation.strip()
    # Round-trips through JSON, which is what the API returns.
    assert HealthAssessment.model_validate_json(assessment.model_dump_json())


@needs_artifacts
def test_published_score_is_the_raw_score_minus_the_penalty(
    service: HealthMonitoringService, published
) -> None:
    # The contract enforces this invariant; this test proves the service produces
    # values that satisfy it rather than relying on the validator alone.
    labelled, features = published
    for turbine in labelled[TURBINE_ID].unique()[:3]:
        assessment = service.assess_from_prepared(labelled, features, turbine_id=str(turbine))
        expected = max(assessment.raw_health_score - assessment.drift_penalty, 0.0)
        assert assessment.health_score == pytest.approx(expected, abs=1e-6)


@needs_artifacts
def test_class_follows_the_published_score(service: HealthMonitoringService, published) -> None:
    from wind_turbine_pm.health.health_class import classify_health

    labelled, features = published
    for turbine in labelled[TURBINE_ID].unique()[:3]:
        assessment = service.assess_from_prepared(labelled, features, turbine_id=str(turbine))
        assert assessment.health_class is classify_health(
            assessment.health_score, service.config, service.class_bands
        )


@needs_artifacts
def test_component_roll_up_covers_the_configured_components(
    service: HealthMonitoringService, published
) -> None:
    labelled, features = published
    assessment = service.assess_from_prepared(
        labelled, features, turbine_id=str(labelled[TURBINE_ID].iloc[0])
    )
    configured = set(service.config.get("health.components", {}) or {})
    reported = {item.component for item in assessment.component_health}

    assert reported <= configured
    assert reported, "at least one component must be scored"
    # Worst first, so an operator reads the actionable one immediately.
    scores = [item.score for item in assessment.component_health]
    assert scores == sorted(scores)
    for item in assessment.component_health:
        assert 0.0 <= item.score <= 100.0
        assert item.sensors


@needs_artifacts
def test_reported_factors_are_grounded_and_ranked(
    service: HealthMonitoringService, published
) -> None:
    labelled, features = published
    top_k = int(service.config.get("health.serving.top_k_factors", 5))
    # Search for a turbine that actually has deviations to report.
    for turbine in labelled[TURBINE_ID].unique():
        assessment = service.assess_from_prepared(labelled, features, turbine_id=str(turbine))
        if not assessment.top_factors:
            continue
        assert len(assessment.top_factors) <= top_k
        impacts = [factor.impact for factor in assessment.top_factors]
        assert impacts == sorted(impacts, reverse=True)
        for factor in assessment.top_factors:
            assert factor.impact > 0
            assert factor.description
            # A raw feature name must never be the whole explanation.
            assert factor.description != factor.feature
        return
    pytest.skip("no turbine in the fixture reports a condition deviation")


@needs_artifacts
def test_assessment_of_an_unknown_turbine_is_an_error(
    service: HealthMonitoringService, published
) -> None:
    labelled, features = published
    with pytest.raises(ValueError, match="No prepared observations"):
        service.assess_from_prepared(labelled, features, turbine_id="does_not_exist")


@needs_artifacts
def test_prepared_assessment_requires_a_single_turbine(
    service: HealthMonitoringService, published
) -> None:
    labelled, features = published
    if labelled[TURBINE_ID].nunique() < 2:
        pytest.skip("fixture has only one turbine")
    with pytest.raises(ValueError, match="single turbine"):
        service.assess_from_prepared(labelled, features)


@needs_artifacts
def test_as_of_cut_off_moves_the_assessed_timestamp(
    service: HealthMonitoringService, published
) -> None:
    labelled, features = published
    turbine = str(labelled[TURBINE_ID].iloc[0])
    history = labelled.loc[labelled[TURBINE_ID] == turbine].sort_values(TIMESTAMP)
    cut_off = pd.Timestamp(history[TIMESTAMP].iloc[len(history) // 2])

    assessment = service.assess_from_prepared(labelled, features, turbine_id=turbine, as_of=cut_off)
    assert pd.Timestamp(assessment.timestamp) <= cut_off


# ---------------------------------------------------------------------------
# Batch and bulk paths
# ---------------------------------------------------------------------------
@needs_artifacts
def test_batch_respects_the_configured_limit(
    service: HealthMonitoringService, health_toy_frame
) -> None:
    window = frame_to_windows(health_toy_frame.loc[health_toy_frame[TURBINE_ID] == "A"])[0]
    limit = int(service.config.get("health.serving.max_batch_turbines", 200))
    with pytest.raises(ValueError, match="exceeds the configured limit"):
        service.assess_batch_from_windows([window] * (limit + 1))


@needs_artifacts
def test_score_frame_reports_one_row_per_observation(
    service: HealthMonitoringService, published
) -> None:
    labelled, features = published
    subset = labelled.iloc[:500]
    scored = service.score_frame(subset, features.loc[subset.index])

    assert len(scored) == len(subset)
    assert {TURBINE_ID, TIMESTAMP, HEALTH_SCORE, HEALTH_CLASS, "raw_health_score"} <= set(
        scored.columns
    )
    assert scored[HEALTH_SCORE].between(0.0, 100.0).all()
    # The published score can only be at or below the model's own estimate.
    assert (scored[HEALTH_SCORE] <= scored["raw_health_score"] + 1e-9).all()


@needs_artifacts
def test_bulk_and_single_paths_agree_on_the_same_row(
    service: HealthMonitoringService, published
) -> None:
    # If these disagreed, the dashboard's fleet table and its detail panel would
    # report different scores for the same turbine at the same timestamp.
    labelled, features = published
    turbine = str(labelled[TURBINE_ID].iloc[0])
    history = labelled.loc[labelled[TURBINE_ID] == turbine].sort_values(TIMESTAMP)

    scored = service.score_frame(history, features.loc[history.index])
    single = service.assess_from_prepared(labelled, features, turbine_id=turbine)

    assert single.health_score == pytest.approx(float(scored[HEALTH_SCORE].iloc[-1]), abs=0.01)
    assert single.raw_health_score == pytest.approx(
        float(scored["raw_health_score"].iloc[-1]), abs=0.01
    )


@needs_artifacts
def test_drift_penalty_is_not_a_constant_offset(
    service: HealthMonitoringService, published
) -> None:
    """Regression test for the defect that made drift a fixed deduction.

    Before the CUSUM restart and the empirical threshold calibration, 92% of all
    observations took the maximum penalty, which is a constant offset on every
    health score rather than a signal. A detector that fires almost everywhere
    carries no information, so the spread of the penalty is the property worth
    pinning.
    """
    labelled, features = published
    scored = service.score_frame(labelled, features)
    penalty = scored["raw_health_score"] - scored[HEALTH_SCORE]

    maximum = float(service.config.get("health.drift.penalty.max_points", 15.0))
    at_maximum = float((penalty >= maximum - 1e-6).mean())
    assert at_maximum < 0.5, f"{at_maximum:.1%} of rows take the maximum drift penalty"


@needs_artifacts
def test_fleet_summary_rolls_up_the_latest_row_per_turbine(
    service: HealthMonitoringService, published
) -> None:
    labelled, features = published
    subset = labelled.iloc[:2000]
    scored = service.score_frame(subset, features.loc[subset.index])
    summary = service.fleet_summary(scored)

    assert summary.n_turbines == scored[TURBINE_ID].nunique()
    assert sum(summary.class_counts.values()) == summary.n_turbines
    assert summary.min_health_score <= summary.mean_health_score
    assert len(summary.worst_turbines) <= 5
    assert summary.advisory_only is True
    # Worst first.
    worst_scores = [item["health_score"] for item in summary.worst_turbines]
    assert worst_scores == sorted(worst_scores)


@needs_artifacts
def test_fleet_summary_of_an_empty_frame_is_well_formed(
    service: HealthMonitoringService,
) -> None:
    empty = pd.DataFrame(
        columns=[
            TURBINE_ID,
            TIMESTAMP,
            OPERATING_REGIME,
            "raw_health_score",
            HEALTH_SCORE,
            HEALTH_CLASS,
        ]
    )
    summary = service.fleet_summary(empty)
    assert summary.n_turbines == 0
    assert set(summary.class_counts) == {str(name) for name in HealthClass}
    assert summary.worst_turbines == []


@needs_artifacts
def test_window_and_prepared_paths_both_produce_a_valid_assessment(
    service: HealthMonitoringService, published
) -> None:
    # The two paths can legitimately differ - the window path has less history
    # for its expanding baselines - but both must produce a contract-valid result.
    labelled, features = published
    turbine = str(labelled[TURBINE_ID].iloc[0])
    history = labelled.loc[labelled[TURBINE_ID] == turbine].sort_values(TIMESTAMP)

    from_prepared = service.assess_from_prepared(labelled, features, turbine_id=turbine)
    from_window = service.assess_from_window(frame_to_windows(history.tail(300))[0])

    for assessment in (from_prepared, from_window):
        assert 0.0 <= assessment.health_score <= 100.0
        assert assessment.turbine_id == turbine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def test_latest_per_turbine_picks_the_newest_row_at_or_before_the_cut_off() -> None:
    frame = pd.DataFrame(
        {
            TURBINE_ID: ["A", "A", "B", "B"],
            TIMESTAMP: pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-02", "2024-01-04"]),
            HEALTH_SCORE: [90.0, 70.0, 80.0, 60.0],
        }
    )
    latest = latest_per_turbine(frame, pd.Timestamp("2024-01-03"))
    assert len(latest) == 2
    assert dict(zip(latest[TURBINE_ID], latest[HEALTH_SCORE])) == {"A": 70.0, "B": 80.0}
    # Worst first.
    assert list(latest[HEALTH_SCORE]) == [70.0, 80.0]


def test_latest_per_turbine_before_any_data_is_empty() -> None:
    frame = pd.DataFrame(
        {
            TURBINE_ID: ["A"],
            TIMESTAMP: pd.to_datetime(["2024-06-01"]),
            HEALTH_SCORE: [90.0],
        }
    )
    assert latest_per_turbine(frame, pd.Timestamp("2024-01-01")).empty


@needs_artifacts
def test_row_drift_penalties_are_bounded_by_the_configured_cap(
    service: HealthMonitoringService, published
) -> None:
    labelled, features = published
    penalties = service._row_drift_penalties(labelled, features)
    maximum = float(service.config.get("health.drift.penalty.max_points", 15.0))

    assert len(penalties) == len(labelled)
    assert np.all(penalties >= 0.0)
    assert np.all(penalties <= maximum + 1e-9)
