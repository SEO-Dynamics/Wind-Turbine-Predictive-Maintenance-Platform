"""Tests for health explanation and advisory text.

The property under test is that the generator never fabricates: every sentence
has to trace back to a value it was given, and when there is no evidence it must
say so rather than produce plausible filler.
"""

from __future__ import annotations

import pytest

from wind_turbine_pm.constants import (
    ADVISORY_DISCLAIMER,
    DriftDirection,
    DriftSeverity,
    HealthClass,
    OperatingRegime,
)
from wind_turbine_pm.contracts.health import ComponentHealth, SensorDriftSignal
from wind_turbine_pm.health.narratives import (
    build_health_advisory,
    build_health_explanation,
    build_health_recommendation,
    describe_drift_signal,
    humanise_health_feature,
)


def _component(name: str, score: float, drivers: list[str] | None = None) -> ComponentHealth:
    return ComponentHealth(
        component=name,
        score=score,
        health_class=HealthClass.HEALTHY if score >= 80 else HealthClass.DEGRADED,
        sensors=["vibration"],
        drivers=drivers or [],
    )


def _drift(sensor: str = "oil_pressure", severity: DriftSeverity = DriftSeverity.ALARM):
    return SensorDriftSignal(
        sensor=sensor,
        method="cusum",
        detected=True,
        severity=severity,
        direction=DriftDirection.DOWNWARD,
        statistic=7.5,
        control_limit=5.0,
        description="Cumulative downward shift in oil pressure crossed its limit 4 time(s).",
    )


# ---------------------------------------------------------------------------
# Feature humanisation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "fragment"),
    [
        ("vibration_rms_24h", "24-hour RMS level of vibration"),
        ("vibration_crest_24h", "crest factor"),
        ("vibration_kurtosis_24h", "impulsiveness"),
        ("vibration_hf_ratio_24h", "high-frequency content"),
        ("vibration_zcr_24h", "zero-crossing rate"),
        ("vibration_p2p_24h", "peak-to-peak"),
        ("gearbox_temperature_rule_exceedance", "alarm band"),
        ("gearbox_temperature_dev_from_regime_baseline", "own baseline"),
        ("vibration_regime_robust_z", "how unusual"),
        ("vibration_cusum_pos", "accumulated upward drift"),
        ("vibration_cusum_neg", "accumulated downward drift"),
        ("vibration_ewma", "smoothed drift residual"),
        ("rule_alarm_count", "alarm limit"),
        ("regime_high_load", "high_load"),
    ],
)
def test_health_features_are_described_readably(name: str, fragment: str) -> None:
    described = humanise_health_feature(name)
    assert fragment in described
    # A raw feature name must never be handed to an operator verbatim.
    assert described != name


def test_shared_platform_patterns_still_apply() -> None:
    # Rolling and trend features must read the same in both modules.
    assert "average" in humanise_health_feature("vibration_roll_mean_24h")
    assert "trend" in humanise_health_feature("vibration_slope_24h")


def test_unknown_feature_degrades_to_a_readable_form() -> None:
    assert humanise_health_feature("some_new_feature") == "some new feature"


# ---------------------------------------------------------------------------
# Drift phrasing
# ---------------------------------------------------------------------------
def test_drift_description_is_reused_without_a_trailing_stop() -> None:
    # The caller punctuates the sentence the clause is embedded in.
    described = describe_drift_signal(_drift())
    assert not described.endswith(".")
    assert "oil pressure" in described


def test_drift_description_is_synthesised_when_absent() -> None:
    signal = _drift().model_copy(update={"description": ""})
    described = describe_drift_signal(signal)
    assert "oil pressure" in described
    assert "alarm" in described


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------
def test_explanation_states_the_score_class_and_regime() -> None:
    text = build_health_explanation(
        score=72.5,
        health_class=HealthClass.MONITOR,
        regime=OperatingRegime.HIGH_LOAD,
        components=[],
        drift_signals=[],
    )
    assert "72.5" in text
    assert "monitor" in text
    assert "high load" in text


def test_explanation_says_so_when_there_is_no_evidence() -> None:
    # No components below the attention threshold and no drift: the narrative must
    # state that plainly instead of inventing a driver.
    text = build_health_explanation(
        score=99.0,
        health_class=HealthClass.HEALTHY,
        regime=OperatingRegime.MEDIUM_LOAD,
        components=[_component("gearbox", 100.0)],
        drift_signals=[],
    )
    assert "No component is showing" in text
    assert "No sensor drift was detected" in text


def test_explanation_names_the_worst_component_and_its_drivers() -> None:
    text = build_health_explanation(
        score=55.0,
        health_class=HealthClass.DEGRADED,
        regime=OperatingRegime.MEDIUM_LOAD,
        components=[
            _component("lubrication", 42.0, ["oil pressure is 3.1 sigma from baseline"]),
            _component("gearbox", 61.0),
        ],
        drift_signals=[],
    )
    assert "lubrication system" in text
    assert "42.0/100" in text
    assert "3.1 sigma" in text
    assert "gearbox" in text  # listed as also below nominal


def test_explanation_reports_drift_and_explains_the_deduction() -> None:
    text = build_health_explanation(
        score=85.0,
        health_class=HealthClass.HEALTHY,
        regime=OperatingRegime.LOW_LOAD,
        components=[],
        drift_signals=[_drift()],
        drift_penalty=5.0,
    )
    assert "1 sensor-drift signal" in text
    assert "5.0 points were deducted" in text
    # No doubled full stop where the clause is embedded.
    assert ".." not in text


def test_explanation_flags_poor_data_quality() -> None:
    text = build_health_explanation(
        score=90.0,
        health_class=HealthClass.HEALTHY,
        regime=OperatingRegime.MEDIUM_LOAD,
        components=[],
        drift_signals=[],
        data_quality=0.6,
    )
    assert "40%" in text
    assert "caution" in text


def test_explanation_omits_the_data_quality_clause_when_clean() -> None:
    text = build_health_explanation(
        score=90.0,
        health_class=HealthClass.HEALTHY,
        regime=OperatingRegime.MEDIUM_LOAD,
        components=[],
        drift_signals=[],
        data_quality=1.0,
    )
    assert "caution" not in text


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------
def test_recommendation_always_carries_the_platform_disclaimer() -> None:
    for health_class in HealthClass:
        text = build_health_recommendation(health_class, [], [])
        assert ADVISORY_DISCLAIMER in text


def test_recommendation_escalates_with_the_class() -> None:
    healthy = build_health_recommendation(HealthClass.HEALTHY, [], [])
    critical = build_health_recommendation(HealthClass.CRITICAL, [], [])
    assert "No action required" in healthy
    assert "Prioritise inspection" in critical


def test_recommendation_names_a_degraded_component_even_when_overall_healthy() -> None:
    # The fleet-trained score and this turbine's own rule evidence can legitimately
    # disagree. Dropping the component finding would leave an operator reading
    # "no action required" while one subsystem sits in its alarm band.
    text = build_health_recommendation(HealthClass.HEALTHY, [_component("lubrication", 55.0)], [])
    assert "lubrication system" in text
    assert "55/100" in text
    assert "before dismissing it" in text


def test_recommendation_points_at_the_worst_component_when_not_healthy() -> None:
    text = build_health_recommendation(
        HealthClass.DEGRADED, [_component("gearbox", 45.0), _component("brake", 70.0)], []
    )
    assert "Start with the gearbox" in text


def test_recommendation_asks_for_calibration_checks_on_drifting_channels() -> None:
    text = build_health_recommendation(
        HealthClass.MONITOR, [], [_drift("oil_pressure"), _drift("vibration")]
    )
    assert "calibration" in text
    assert "oil pressure" in text
    assert "vibration" in text
    assert "instrument fault" in text


def test_multivariate_signal_is_not_named_as_a_channel_to_calibrate() -> None:
    signal = _drift("multivariate").model_copy(update={"method": "isolation_forest"})
    text = build_health_recommendation(HealthClass.MONITOR, [], [signal])
    assert "multivariate" not in text


def test_advisory_pairs_both_texts() -> None:
    advisory = build_health_advisory(
        score=61.0,
        health_class=HealthClass.MONITOR,
        regime=OperatingRegime.MEDIUM_LOAD,
        components=[_component("gearbox", 70.0)],
        drift_signals=[_drift()],
        drift_penalty=2.5,
        data_quality=0.95,
    )
    assert "61.0" in advisory.explanation
    assert ADVISORY_DISCLAIMER in advisory.recommendation
