"""Tests for sensor validation and operating-envelope rules.

The rules are configuration rather than code, so the tests concentrate on the
two things configuration can get wrong - internally inconsistent limits, and
limits that disagree with the platform's shared physical ranges - plus the two
failure modes a plain range check cannot see (frozen signals and impossible
slew).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wind_turbine_pm.config import Config, ConfigError
from wind_turbine_pm.constants import TIMESTAMP, TURBINE_ID, DriftSeverity
from wind_turbine_pm.health.sensor_rules import (
    KNOWN_SOURCES,
    SensorRule,
    evaluate_rules,
    load_sensor_rules,
    rules_to_records,
    verify_against_physical_ranges,
)


def _high_rule(**overrides) -> dict:
    payload = {
        "sensor": "gearbox_temperature",
        "component": "gearbox",
        "unit": "degC",
        "hard_min": -30.0,
        "hard_max": 160.0,
        "direction": "high_is_bad",
        "source": "expert_judgement",
        "rationale": "test",
        "warn_above": 75.0,
        "alarm_above": 90.0,
    }
    payload.update(overrides)
    return payload


def _low_rule(**overrides) -> dict:
    payload = {
        "sensor": "oil_pressure",
        "component": "lubrication",
        "unit": "bar",
        "hard_min": 0.0,
        "hard_max": 12.0,
        "direction": "low_is_bad",
        "source": "data_analysis",
        "rationale": "test",
        "warn_below": 3.6,
        "alarm_below": 2.8,
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Rule validation
# ---------------------------------------------------------------------------
def test_shipped_rules_load_and_declare_known_provenance(health_config: Config) -> None:
    rules = load_sensor_rules(health_config)
    assert rules, "the shipped configuration must define sensor rules"
    for rule in rules.values():
        assert rule.source in KNOWN_SOURCES
        # Every limit must be justified in prose: an unexplained threshold is not
        # reviewable by the engineer who has to trust it.
        assert rule.rationale.strip(), f"{rule.sensor} has no rationale"


def test_shipped_hard_limits_agree_with_platform_physical_ranges(health_config: Config) -> None:
    # The health rules restate configs/data.yaml -> validation.physical_ranges so
    # they can be read on their own; restating a value is a chance to disagree.
    assert verify_against_physical_ranges(health_config) == []


def test_missing_rules_block_raises_with_actionable_message() -> None:
    with pytest.raises(ConfigError, match="load_health_config"):
        load_sensor_rules(Config({}))


def test_inverted_hard_limits_rejected() -> None:
    with pytest.raises(ConfigError, match="hard_min"):
        SensorRule(**_high_rule(hard_min=200.0))


def test_high_is_bad_without_envelope_limits_rejected() -> None:
    with pytest.raises(ConfigError, match="warn_above"):
        SensorRule(**_high_rule(warn_above=None, alarm_above=None))


def test_low_is_bad_without_envelope_limits_rejected() -> None:
    with pytest.raises(ConfigError, match="warn_below"):
        SensorRule(**_low_rule(warn_below=None, alarm_below=None))


def test_misordered_envelope_limits_rejected() -> None:
    with pytest.raises(ConfigError, match="warn_above"):
        SensorRule(**_high_rule(warn_above=95.0, alarm_above=90.0))
    with pytest.raises(ConfigError, match="alarm_below"):
        SensorRule(**_low_rule(warn_below=2.0, alarm_below=3.0))


def test_unknown_provenance_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown source"):
        SensorRule(**_high_rule(source="somebody_guessed"))


# ---------------------------------------------------------------------------
# Exceedance and severity
# ---------------------------------------------------------------------------
def test_exceedance_is_zero_below_warning_one_at_alarm() -> None:
    rule = SensorRule(**_high_rule())
    values = pd.Series([20.0, 75.0, 82.5, 90.0, 105.0])
    exceedance = rule.exceedance(values)
    assert exceedance.iloc[0] == 0.0
    assert exceedance.iloc[1] == 0.0  # at the warning limit, not yet past it
    assert exceedance.iloc[2] == pytest.approx(0.5)
    assert exceedance.iloc[3] == pytest.approx(1.0)
    assert exceedance.iloc[4] > 1.0


def test_exceedance_direction_is_inverted_for_low_is_bad() -> None:
    rule = SensorRule(**_low_rule())
    exceedance = rule.exceedance(pd.Series([5.0, 3.6, 3.2, 2.8, 1.0]))
    assert exceedance.iloc[0] == 0.0
    assert exceedance.iloc[2] == pytest.approx(0.5)
    assert exceedance.iloc[3] == pytest.approx(1.0)
    assert exceedance.iloc[4] > 1.0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (20.0, DriftSeverity.NONE),
        (75.0, DriftSeverity.NONE),
        (80.0, DriftSeverity.WARNING),
        (90.0, DriftSeverity.ALARM),
        (float("nan"), DriftSeverity.NONE),
    ],
)
def test_severity_of_bands_a_single_reading(value: float, expected: DriftSeverity) -> None:
    assert SensorRule(**_high_rule()).severity_of(value) is expected


def test_severity_and_exceedance_agree_on_every_reading() -> None:
    # The same reading must not be a warning by one method and healthy by the
    # other: the component score uses exceedance, the API reports severity.
    for payload in (_high_rule(), _low_rule()):
        rule = SensorRule(**payload)
        values = np.linspace(rule.hard_min, rule.hard_max, 200)
        exceedance = rule.exceedance(pd.Series(values)).to_numpy()
        for value, margin in zip(values, exceedance):
            severity = rule.severity_of(float(value))
            if severity is DriftSeverity.ALARM:
                assert margin >= 1.0
            elif severity is DriftSeverity.WARNING:
                assert 0.0 < margin < 1.0
            else:
                assert margin == 0.0


def test_rules_to_records_is_json_serialisable(health_config: Config) -> None:
    import json

    records = rules_to_records(load_sensor_rules(health_config))
    assert len(records) == len(load_sensor_rules(health_config))
    json.dumps(records)  # must not raise


# ---------------------------------------------------------------------------
# Evaluation against a frame
# ---------------------------------------------------------------------------
def _frame(values: dict[str, list[float]], hours: int) -> pd.DataFrame:
    start = pd.Timestamp("2024-01-01")
    frame = pd.DataFrame(
        {
            TIMESTAMP: [start + pd.Timedelta(hours=h) for h in range(hours)],
            TURBINE_ID: ["A"] * hours,
        }
    )
    for column, series in values.items():
        frame[column] = series
    return frame


def test_evaluate_rules_requires_key_columns(health_config: Config) -> None:
    with pytest.raises(ValueError, match="turbine_id"):
        evaluate_rules(pd.DataFrame({"vibration": [1.0]}), health_config)


def test_frozen_signal_is_flagged_even_though_it_is_in_range(
    small_health_config: Config,
) -> None:
    # A sensor stuck at a plausible value passes every range check forever. This
    # is the case range checks cannot see and the stuck detector exists for.
    hours = 48
    frame = _frame({"vibration": [3.0] * hours}, hours)
    evaluation = evaluate_rules(frame, small_health_config)

    assert "vibration__stuck" in evaluation.flags.columns
    assert evaluation.flags["vibration__stuck"].any()
    assert not evaluation.flags["vibration__out_of_range"].any()
    # Stuck is a validity problem, so it must depress data quality.
    assert evaluation.data_quality() < 1.0


def test_moving_signal_is_not_flagged_as_frozen(small_health_config: Config) -> None:
    hours = 48
    rng = np.random.default_rng(0)
    frame = _frame({"vibration": list(3.0 + rng.normal(0, 0.3, hours))}, hours)
    evaluation = evaluate_rules(frame, small_health_config)
    assert not evaluation.flags["vibration__stuck"].any()


def test_impossible_slew_is_flagged(small_health_config: Config) -> None:
    hours = 24
    values = [3.0] * hours
    values[12] = 20.0  # a jump far beyond vibration's max_rate_per_hour of 8.0
    frame = _frame({"vibration": values}, hours)
    evaluation = evaluate_rules(frame, small_health_config)
    assert evaluation.flags["vibration__rate"].sum() >= 1


def test_out_of_range_reading_is_flagged(small_health_config: Config) -> None:
    hours = 12
    values = [3.0] * hours
    values[5] = 999.0
    frame = _frame({"vibration": values}, hours)
    evaluation = evaluate_rules(frame, small_health_config)
    assert evaluation.flags["vibration__out_of_range"].sum() == 1


def test_envelope_exceedance_does_not_count_as_a_data_quality_problem(
    small_health_config: Config,
) -> None:
    # A hot gearbox is a real measurement of an unhealthy machine, not a broken
    # sensor. Conflating the two is how condition monitoring goes wrong.
    hours = 24
    rng = np.random.default_rng(1)
    frame = _frame({"gearbox_temperature": list(95.0 + rng.normal(0, 0.4, hours))}, hours)
    evaluation = evaluate_rules(frame, small_health_config)

    assert evaluation.flags["gearbox_temperature__alarm"].all()
    assert evaluation.data_quality() == pytest.approx(1.0)


def test_evaluation_outputs_are_aligned_to_the_input_index(
    small_health_config: Config, health_prepared
) -> None:
    labelled, _, _ = health_prepared
    subset = labelled.iloc[100:400]
    evaluation = evaluate_rules(subset, small_health_config)
    assert evaluation.flags.index.equals(subset.index)
    assert evaluation.exceedance.index.equals(subset.index)
    assert 0.0 <= evaluation.data_quality() <= 1.0


def test_counts_and_summary_are_reported(small_health_config: Config, health_prepared) -> None:
    labelled, _, _ = health_prepared
    evaluation = evaluate_rules(labelled, small_health_config)
    summary = evaluation.to_dict()
    assert summary["n_rows"] == len(labelled)
    assert set(evaluation.counts) <= set(load_sensor_rules(small_health_config))
    for per_sensor in evaluation.counts.values():
        assert all(count >= 0 for count in per_sensor.values())
