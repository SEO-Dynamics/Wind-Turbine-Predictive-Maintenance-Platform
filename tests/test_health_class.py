"""Tests for health classification and the drift penalty that feeds it.

The class is what an operator acts on, so the boundary behaviour has to be exact
and the two code paths - scalar and vectorised - must never disagree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import HEALTH_CLASS_ORDER, HealthClass
from wind_turbine_pm.contracts.health import HealthClassBands
from wind_turbine_pm.health.health_class import (
    ClassBands,
    adaptive_bands,
    apply_drift_penalty,
    class_distribution,
    classify_health,
    classify_health_series,
    recommended_action,
    severity_rank,
)


# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------
def test_shipped_bands_are_strictly_ordered(health_config: Config) -> None:
    bands = ClassBands.from_config(health_config)
    assert bands.degraded_min < bands.monitor_min < bands.healthy_min
    bands.to_contract()  # must validate


def test_contract_rejects_inverted_bands() -> None:
    with pytest.raises(ValidationError, match="degraded_min < monitor_min < healthy_min"):
        HealthClassBands(healthy_min=40.0, monitor_min=60.0, degraded_min=80.0)


def test_shifted_bands_stay_ordered_and_clamped() -> None:
    bands = ClassBands(healthy_min=80.0, monitor_min=60.0, degraded_min=40.0)
    for offset in (-500.0, -25.0, 0.0, 25.0, 500.0):
        shifted = bands.shifted(offset)
        assert shifted.degraded_min < shifted.monitor_min < shifted.healthy_min
        assert shifted.degraded_min >= 0.0
        assert shifted.healthy_min <= 100.0


def test_bands_round_trip_through_a_dictionary(health_config: Config) -> None:
    bands = ClassBands.from_config(health_config)
    assert ClassBands(**bands.to_dict()) == bands


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (100.0, HealthClass.HEALTHY),
        (80.0, HealthClass.HEALTHY),  # inclusive lower bound
        (79.9, HealthClass.MONITOR),
        (60.0, HealthClass.MONITOR),
        (59.9, HealthClass.DEGRADED),
        (40.0, HealthClass.DEGRADED),
        (39.9, HealthClass.CRITICAL),
        (0.0, HealthClass.CRITICAL),
    ],
)
def test_boundaries_are_inclusive_from_below(
    health_config: Config, score: float, expected: HealthClass
) -> None:
    assert classify_health(score, health_config) is expected


def test_scalar_and_vectorised_paths_agree(health_config: Config) -> None:
    # One reading must not be Monitor by one code path and Degraded by the other.
    scores = pd.Series(np.linspace(0.0, 100.0, 501))
    vectorised = classify_health_series(scores, health_config)
    scalar = [str(classify_health(float(value), health_config)) for value in scores]
    assert list(vectorised) == scalar


def test_series_classification_preserves_the_index(health_config: Config) -> None:
    scores = pd.Series([95.0, 50.0], index=[7, 11])
    result = classify_health_series(scores, health_config)
    assert list(result.index) == [7, 11]


def test_classification_with_explicit_bands_overrides_configuration(
    health_config: Config,
) -> None:
    strict = ClassBands(healthy_min=95.0, monitor_min=90.0, degraded_min=85.0)
    assert classify_health(92.0, health_config) is HealthClass.HEALTHY
    assert classify_health(92.0, health_config, strict) is HealthClass.MONITOR


def test_classification_rejects_inverted_explicit_bands(health_config: Config) -> None:
    broken = ClassBands(healthy_min=10.0, monitor_min=50.0, degraded_min=90.0)
    with pytest.raises(ValidationError):
        classify_health(50.0, health_config, broken)


# ---------------------------------------------------------------------------
# Adaptive bands
# ---------------------------------------------------------------------------
def test_adaptation_is_off_by_default(health_config: Config) -> None:
    scores = pd.Series([10.0] * 100)
    assert adaptive_bands(scores, health_config) == ClassBands.from_config(health_config)


def test_adaptation_shifts_with_the_fleet_and_respects_the_cap(health_config: Config) -> None:
    data = health_config.to_dict()
    data["health"]["classes"]["adaptive"]["enabled"] = True
    data["health"]["classes"]["adaptive"]["max_shift"] = 10.0
    cfg = Config(data)

    baseline = ClassBands.from_config(cfg)
    # A fleet sitting far below the Monitor midpoint would shift the bands down,
    # but never by more than max_shift.
    shifted = adaptive_bands(pd.Series([5.0] * 200), cfg)
    assert shifted.healthy_min < baseline.healthy_min
    assert baseline.healthy_min - shifted.healthy_min <= 10.0 + 1e-9
    assert shifted.degraded_min < shifted.monitor_min < shifted.healthy_min


def test_adaptation_of_an_empty_series_returns_the_configured_bands(
    health_config: Config,
) -> None:
    data = health_config.to_dict()
    data["health"]["classes"]["adaptive"]["enabled"] = True
    cfg = Config(data)
    assert adaptive_bands(pd.Series([], dtype=float), cfg) == ClassBands.from_config(cfg)


# ---------------------------------------------------------------------------
# Drift penalty
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "penalty", "expected"),
    [
        (100.0, 0.0, 100.0),
        (100.0, 15.0, 85.0),
        (10.0, 15.0, 0.0),  # floored, never negative
        (50.0, -5.0, 50.0),  # a negative penalty cannot raise the score
    ],
)
def test_penalty_is_subtracted_and_clamped(raw: float, penalty: float, expected: float) -> None:
    assert apply_drift_penalty(raw, penalty) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Ordering and reporting
# ---------------------------------------------------------------------------
def test_severity_rank_orders_best_to_worst() -> None:
    ranks = [severity_rank(name) for name in HEALTH_CLASS_ORDER]
    assert ranks == sorted(ranks)
    assert severity_rank(HealthClass.HEALTHY) == 0
    assert severity_rank("critical") == len(HEALTH_CLASS_ORDER) - 1


def test_severity_rank_rejects_an_unknown_class() -> None:
    with pytest.raises(ValueError):
        severity_rank("perfectly_fine")


def test_every_class_has_an_advisory_action_that_escalates() -> None:
    actions = {name: recommended_action(name) for name in HEALTH_CLASS_ORDER}
    assert all(text.strip() for text in actions.values())
    assert len(set(actions.values())) == len(actions), "each class needs distinct advice"
    # The advisory must never instruct anyone to operate the machine.
    for text in actions.values():
        assert "shut down" not in text.lower()
        assert "stop the turbine" not in text.lower()


def test_class_distribution_always_reports_all_four_classes() -> None:
    counts = class_distribution(pd.Series(["healthy", "healthy", "critical"]))
    assert set(counts) == {str(name) for name in HEALTH_CLASS_ORDER}
    assert counts["healthy"] == 2
    assert counts["critical"] == 1
    assert counts["monitor"] == 0


def test_class_distribution_of_an_empty_series_is_all_zero() -> None:
    counts = class_distribution(pd.Series([], dtype=object))
    assert set(counts) == {str(name) for name in HEALTH_CLASS_ORDER}
    assert sum(counts.values()) == 0
