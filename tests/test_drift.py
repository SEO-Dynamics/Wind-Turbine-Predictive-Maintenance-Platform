"""Tests for sensor-drift detection.

Two properties matter most, and both are regression tests for defects that made
the detector useless in practice:

* the CUSUM statistic must stay **bounded**, because an unbounded arm pins every
  row past the first crossing at ``alarm`` forever;
* the thresholds must be **calibrated against the healthy population**, because
  the textbook constants assume independent residuals and hourly SCADA channels
  are strongly autocorrelated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import (
    OPERATING_REGIME,
    TIMESTAMP,
    TURBINE_ID,
    DriftSeverity,
    OperatingRegime,
)
from wind_turbine_pm.contracts.health import SensorDriftSignal
from wind_turbine_pm.health.drift import (
    DriftCalibration,
    DriftSettings,
    MultivariateDriftDetector,
    compute_drift_statistics,
    cusum_statistics,
    drift_penalty,
    drift_report,
    ewma_statistic,
    producing_mask,
    regime_conditioned_z,
    summarise_drift,
    trailing_signal_count,
)


def _signal(sensor: str, method: str, severity: DriftSeverity) -> SensorDriftSignal:
    return SensorDriftSignal(
        sensor=sensor,
        method=method,
        detected=True,
        severity=severity,
        statistic=7.0,
        control_limit=5.0,
    )


# ---------------------------------------------------------------------------
# CUSUM
# ---------------------------------------------------------------------------
def test_cusum_ignores_noise_below_the_slack() -> None:
    # Slack k is what stops the statistic accumulating on noise alone.
    rng = np.random.default_rng(0)
    residuals = pd.Series(rng.normal(0, 0.2, 500))
    groups = pd.Series(["A"] * 500)
    upward, downward = cusum_statistics(residuals, groups, k=0.5)
    assert upward.max() < 2.0
    assert downward.max() < 2.0


def test_cusum_accumulates_a_sustained_shift() -> None:
    residuals = pd.Series([1.5] * 100)
    groups = pd.Series(["A"] * 100)
    upward, downward = cusum_statistics(residuals, groups, k=0.5)
    assert upward.iloc[-1] > 5.0
    assert downward.iloc[-1] == 0.0


def test_cusum_resets_at_a_turbine_boundary() -> None:
    residuals = pd.Series([2.0] * 20 + [0.0] * 5)
    groups = pd.Series(["A"] * 20 + ["B"] * 5)
    upward, _ = cusum_statistics(residuals, groups, k=0.5)
    assert upward.iloc[19] > 5.0
    assert upward.iloc[20] == 0.0, "the statistic must not carry across turbines"


def test_cusum_treats_a_missing_residual_as_no_evidence() -> None:
    residuals = pd.Series([1.5, 1.5, np.nan, 1.5])
    groups = pd.Series(["A"] * 4)
    upward, _ = cusum_statistics(residuals, groups, k=0.5)
    # The NaN row holds its value rather than resetting or jumping.
    assert upward.iloc[2] == pytest.approx(upward.iloc[1])
    assert upward.iloc[3] > upward.iloc[2]


def test_restart_bounds_the_statistic_while_no_restart_does_not() -> None:
    """Regression test for the defect that made the drift penalty a constant.

    Without a restart the arm grows without limit under any sustained residual.
    Measured on the real dataset it reached ~1900 sigma, every row past the first
    crossing reported an alarm, and 92% of all observations took the maximum
    drift penalty - a constant offset on every health score rather than a signal.
    """
    residuals = pd.Series([1.5] * 1000)
    groups = pd.Series(["A"] * 1000)

    unbounded, _ = cusum_statistics(residuals, groups, k=0.5, h=None)
    bounded, _ = cusum_statistics(residuals, groups, k=0.5, h=5.0)

    assert unbounded.iloc[-1] > 500.0
    # After a restart the arm can only reach h plus one step's contribution.
    assert bounded.max() <= 5.0 + 1.5


def test_restart_produces_repeated_crossings_for_a_persistent_drift() -> None:
    residuals = pd.Series([1.5] * 200)
    groups = pd.Series(["A"] * 200)
    upward, _ = cusum_statistics(residuals, groups, k=0.5, h=5.0)
    crossings = int((upward >= 5.0).sum())
    # Persistence becomes countable, which is what severity is graded on.
    assert crossings > 10


def test_trailing_signal_count_is_windowed_and_per_turbine() -> None:
    crossings = pd.Series([True] * 5 + [False] * 5 + [True] * 2)
    groups = pd.Series(["A"] * 10 + ["B"] * 2)
    counts = trailing_signal_count(crossings, groups, window=100)
    assert counts.iloc[9] == 5.0
    assert counts.iloc[10] == 1.0, "the count must restart at a turbine boundary"
    assert counts.iloc[11] == 2.0


def test_trailing_signal_count_forgets_outside_the_window() -> None:
    crossings = pd.Series([True] * 3 + [False] * 50)
    groups = pd.Series(["A"] * 53)
    counts = trailing_signal_count(crossings, groups, window=10)
    assert counts.iloc[2] == 3.0
    assert counts.iloc[-1] == 0.0


# ---------------------------------------------------------------------------
# EWMA and residuals
# ---------------------------------------------------------------------------
def test_ewma_reacts_to_a_step_and_stays_bounded_by_the_step() -> None:
    residuals = pd.Series([0.0] * 50 + [2.0] * 200)
    groups = pd.Series(["A"] * 250)
    ewma = ewma_statistic(residuals, groups, lam=0.1)
    assert abs(ewma.iloc[49]) < 0.05
    assert ewma.iloc[-1] == pytest.approx(2.0, abs=0.05)


def test_ewma_control_limit_follows_the_configured_smoothing() -> None:
    settings = DriftSettings.from_config(
        Config({"health": {"drift": {"ewma": {"lambda": 0.1, "l_sigma": 3.0}}}})
    )
    assert settings.ewma_control_limit == pytest.approx(3.0 * np.sqrt(0.1 / 1.9))


def test_regime_conditioned_z_is_past_only() -> None:
    values = pd.Series([5.0] * 60 + [50.0])
    groups = pd.Series(["A"] * 61)
    mask = pd.Series([True] * 61)
    z = regime_conditioned_z(values, groups, mask, min_periods=24)
    # The final spike must not contribute to the baseline it is judged against,
    # so it has to register as extreme rather than be absorbed.
    assert abs(float(z.iloc[-1])) > 3.0


def test_producing_mask_selects_only_producing_regimes() -> None:
    frame = pd.DataFrame(
        {
            OPERATING_REGIME: [
                str(OperatingRegime.IDLE),
                str(OperatingRegime.LOW_LOAD),
                str(OperatingRegime.HIGH_LOAD),
                str(OperatingRegime.OFFLINE),
                str(OperatingRegime.CURTAILED),
            ]
        }
    )
    assert list(producing_mask(frame)) == [False, True, True, False, False]


def test_producing_mask_defaults_to_everything_without_the_column() -> None:
    assert producing_mask(pd.DataFrame({"a": [1, 2]})).all()


# ---------------------------------------------------------------------------
# Statistics over a frame
# ---------------------------------------------------------------------------
def test_compute_drift_statistics_emits_every_column_per_sensor(
    small_health_config: Config, health_prepared
) -> None:
    labelled, _, _ = health_prepared
    statistics = compute_drift_statistics(labelled, small_health_config)
    settings = DriftSettings.from_config(small_health_config)

    for sensor in settings.sensors:
        for suffix in ("_drift_z", "_cusum_pos", "_cusum_neg", "_cusum_signals", "_ewma"):
            assert f"{sensor}{suffix}" in statistics.columns
    assert statistics.index.equals(labelled.index)


def test_computed_cusum_arms_stay_bounded_on_real_data(
    small_health_config: Config, health_prepared
) -> None:
    labelled, _, _ = health_prepared
    statistics = compute_drift_statistics(labelled, small_health_config)
    settings = DriftSettings.from_config(small_health_config)
    assert settings.cusum_reset, "the restart scheme must be enabled by default"

    for column in [c for c in statistics.columns if c.endswith(("_cusum_pos", "_cusum_neg"))]:
        sensor = column.removesuffix("_cusum_pos").removesuffix("_cusum_neg")
        peak = float(statistics[column].max())
        # With the restart the arm can never exceed the decision limit plus the
        # single largest residual that can be added in one step. That bound is
        # what matters: without the restart the arm grows without limit instead.
        largest_step = float(statistics[f"{sensor}_drift_z"].abs().max())
        assert peak <= settings.cusum_h + largest_step + 1e-6, f"{column} reached {peak}"


def test_compute_drift_statistics_requires_turbine_id(small_health_config: Config) -> None:
    with pytest.raises(ValueError, match="turbine_id"):
        compute_drift_statistics(pd.DataFrame({"vibration": [1.0]}), small_health_config)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def test_calibration_thresholds_hit_their_target_rate_in_sample(
    small_health_config: Config, health_prepared
) -> None:
    labelled, _, _ = health_prepared
    statistics = compute_drift_statistics(labelled, small_health_config)
    target = pd.to_numeric(labelled["health_score_target"], errors="coerce")
    healthy = target >= 95.0

    calibration = DriftCalibration.fit(statistics, healthy, small_health_config)
    assert calibration is not None

    settings = DriftSettings.from_config(small_health_config)
    subset = statistics.loc[healthy]
    for sensor, threshold in calibration.warning_at.items():
        counts = subset[f"{sensor}_cusum_signals"]
        rate = float((counts >= threshold).mean())
        # A quantile threshold cannot be exact on discrete counts, so allow
        # headroom; the point is that it is near the target rather than at 100%.
        assert rate <= calibration.target_warning_rate + 0.10, f"{sensor} fires at {rate:.2%}"
        assert threshold >= 1.0
        assert calibration.alarm_at[sensor] > threshold
    assert settings.sensors


def test_calibrated_ewma_limits_exceed_the_theoretical_limit(
    small_health_config: Config, health_prepared
) -> None:
    # The asymptotic EWMA limit assumes independent residuals. On autocorrelated
    # channels the empirical quantile lands well above it - which is precisely
    # why the uncalibrated limit fired almost everywhere.
    labelled, _, _ = health_prepared
    statistics = compute_drift_statistics(labelled, small_health_config)
    healthy = pd.to_numeric(labelled["health_score_target"], errors="coerce") >= 95.0
    calibration = DriftCalibration.fit(statistics, healthy, small_health_config)
    settings = DriftSettings.from_config(small_health_config)

    assert calibration is not None
    assert calibration.ewma_warning_at
    for limit in calibration.ewma_warning_at.values():
        assert limit >= settings.ewma_control_limit


def test_calibration_survives_a_json_round_trip(
    small_health_config: Config, health_prepared
) -> None:
    import json

    labelled, _, _ = health_prepared
    statistics = compute_drift_statistics(labelled, small_health_config)
    healthy = pd.to_numeric(labelled["health_score_target"], errors="coerce") >= 95.0
    original = DriftCalibration.fit(statistics, healthy, small_health_config)
    assert original is not None

    restored = DriftCalibration.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored is not None
    assert restored.warning_at == pytest.approx(original.warning_at)
    assert restored.target_alarm_rate == original.target_alarm_rate


def test_calibration_is_skipped_when_there_are_too_few_healthy_rows(
    small_health_config: Config, health_prepared
) -> None:
    labelled, _, _ = health_prepared
    statistics = compute_drift_statistics(labelled, small_health_config)
    almost_none = pd.Series(False, index=labelled.index)
    assert DriftCalibration.fit(statistics, almost_none, small_health_config) is None


def test_calibration_can_be_disabled(small_health_config: Config, health_prepared) -> None:
    data = small_health_config.to_dict()
    data["health"]["drift"]["calibration"]["enabled"] = False
    labelled, _, _ = health_prepared
    statistics = compute_drift_statistics(labelled, small_health_config)
    healthy = pd.Series(True, index=labelled.index)
    assert DriftCalibration.fit(statistics, healthy, Config(data)) is None


def test_from_dict_returns_none_for_an_absent_calibration() -> None:
    assert DriftCalibration.from_dict(None) is None
    assert DriftCalibration.from_dict({}) is None


def test_multivariate_threshold_is_calibrated_when_scores_are_supplied(
    small_health_config: Config, health_prepared
) -> None:
    # The Isolation Forest score is logistic-squashed around the training median,
    # so the configured constant has no defined false-alarm rate. Measured before
    # it was calibrated it fired on 43% of healthy observations.
    labelled, _, _ = health_prepared
    statistics = compute_drift_statistics(labelled, small_health_config)
    detector = MultivariateDriftDetector.fit(statistics, small_health_config)
    assert detector is not None

    healthy = pd.to_numeric(labelled["health_score_target"], errors="coerce") >= 95.0
    scores = detector.score(statistics)
    calibration = DriftCalibration.fit(
        statistics, healthy, small_health_config, anomaly_scores=scores
    )
    assert calibration is not None
    assert calibration.multivariate_warning_at is not None

    healthy_scores = scores.loc[healthy].dropna()
    rate = float((healthy_scores >= calibration.multivariate_warning_at).mean())
    assert rate <= calibration.target_warning_rate + 0.02
    assert calibration.multivariate_alarm_at >= calibration.multivariate_warning_at


def test_multivariate_threshold_falls_back_to_the_configured_constant(
    small_health_config: Config,
) -> None:
    settings = DriftSettings.from_config(small_health_config)
    uncalibrated = DriftCalibration(
        warning_at={"vibration": 5.0},
        alarm_at={"vibration": 9.0},
        ewma_warning_at={},
        ewma_alarm_at={},
        target_warning_rate=0.05,
        target_alarm_rate=0.01,
        fitted_rows=100,
    )
    warning, alarm = uncalibrated.multivariate_thresholds_for(settings)
    assert warning == pytest.approx(settings.isolation_threshold)
    assert alarm >= warning


def test_multivariate_calibration_survives_a_round_trip(
    small_health_config: Config, health_prepared
) -> None:
    import json

    labelled, _, _ = health_prepared
    statistics = compute_drift_statistics(labelled, small_health_config)
    detector = MultivariateDriftDetector.fit(statistics, small_health_config)
    assert detector is not None
    healthy = pd.to_numeric(labelled["health_score_target"], errors="coerce") >= 95.0
    original = DriftCalibration.fit(
        statistics, healthy, small_health_config, anomaly_scores=detector.score(statistics)
    )
    assert original is not None

    restored = DriftCalibration.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored is not None
    assert restored.multivariate_warning_at == pytest.approx(
        original.multivariate_warning_at, abs=1e-5
    )


def test_summarise_uses_the_calibrated_multivariate_threshold(
    small_health_config: Config,
) -> None:
    statistics = pd.DataFrame(
        {
            "vibration_cusum_pos": [0.0] * 20,
            "vibration_cusum_neg": [0.0] * 20,
            "vibration_cusum_signals": [0.0] * 20,
            "vibration_ewma": [0.0] * 20,
        }
    )
    calibration = DriftCalibration(
        warning_at={"vibration": 5.0},
        alarm_at={"vibration": 9.0},
        ewma_warning_at={"vibration": 9.9},
        ewma_alarm_at={"vibration": 10.0},
        target_warning_rate=0.05,
        target_alarm_rate=0.01,
        fitted_rows=100,
        multivariate_warning_at=0.90,
        multivariate_alarm_at=0.98,
    )
    # A score the configured constant (0.62) would flag but the calibration does not.
    quiet = summarise_drift(
        statistics, small_health_config, anomaly_score=0.70, calibration=calibration
    )
    assert quiet == []

    loud = summarise_drift(
        statistics, small_health_config, anomaly_score=0.99, calibration=calibration
    )
    assert [s.method for s in loud] == ["isolation_forest"]
    assert loud[0].severity is DriftSeverity.ALARM


def test_thresholds_for_falls_back_for_an_uncalibrated_sensor(
    small_health_config: Config,
) -> None:
    settings = DriftSettings.from_config(small_health_config)
    calibration = DriftCalibration(
        warning_at={"vibration": 12.0},
        alarm_at={"vibration": 20.0},
        ewma_warning_at={},
        ewma_alarm_at={},
        target_warning_rate=0.05,
        target_alarm_rate=0.01,
        fitted_rows=100,
    )
    assert calibration.thresholds_for("vibration", settings) == (12.0, 20.0)
    assert calibration.thresholds_for("oil_pressure", settings) == (
        1.0,
        float(settings.alarm_signals),
    )
    warning, alarm = calibration.ewma_thresholds_for("oil_pressure", settings)
    assert warning == pytest.approx(settings.ewma_control_limit * settings.warning_ratio)
    assert alarm > warning


# ---------------------------------------------------------------------------
# Summarising and penalties
# ---------------------------------------------------------------------------
def test_summarise_returns_nothing_for_an_empty_frame(small_health_config: Config) -> None:
    assert summarise_drift(pd.DataFrame(), small_health_config) == []


def test_summarise_reports_a_persistent_drift_and_orders_by_severity(
    small_health_config: Config,
) -> None:
    hours = 300
    index = pd.RangeIndex(hours)
    statistics = pd.DataFrame(index=index)
    for sensor, signals in (("vibration", 40), ("gearbox_temperature", 1)):
        statistics[f"{sensor}_cusum_pos"] = 6.0
        statistics[f"{sensor}_cusum_neg"] = 0.0
        statistics[f"{sensor}_cusum_signals"] = float(signals)
        statistics[f"{sensor}_ewma"] = 0.0

    # A warning needs one crossing, an alarm twenty. Vibration has forty and so
    # alarms; the gearbox has one and so stays a warning.
    calibration = DriftCalibration(
        warning_at={"vibration": 1.0, "gearbox_temperature": 1.0},
        alarm_at={"vibration": 20.0, "gearbox_temperature": 20.0},
        # EWMA limits set far above the fixture's zero residual so only the CUSUM
        # arm is under test here.
        ewma_warning_at={"vibration": 9.9, "gearbox_temperature": 9.9},
        ewma_alarm_at={"vibration": 10.0, "gearbox_temperature": 10.0},
        target_warning_rate=0.05,
        target_alarm_rate=0.01,
        fitted_rows=1000,
    )
    signals = summarise_drift(statistics, small_health_config, calibration=calibration)
    kinds = {(s.sensor, s.severity) for s in signals}

    assert ("vibration", DriftSeverity.ALARM) in kinds
    assert ("gearbox_temperature", DriftSeverity.WARNING) in kinds
    assert all(signal.method == "cusum" for signal in signals)
    # Most severe first.
    assert signals[0].severity is DriftSeverity.ALARM


def test_summarise_reports_nothing_when_no_channel_crosses(
    small_health_config: Config,
) -> None:
    statistics = pd.DataFrame(
        {
            "vibration_cusum_pos": [0.0] * 50,
            "vibration_cusum_neg": [0.0] * 50,
            "vibration_cusum_signals": [0.0] * 50,
            "vibration_ewma": [0.0] * 50,
        }
    )
    assert summarise_drift(statistics, small_health_config) == []


def test_penalty_counts_each_sensor_once_at_its_worst_severity(
    small_health_config: Config,
) -> None:
    # CUSUM and EWMA firing on the same channel is one drifting sensor, not two.
    warning_points = float(small_health_config.get("health.drift.penalty.per_warning_points", 2.5))
    alarm_points = float(small_health_config.get("health.drift.penalty.per_alarm_points", 5.0))

    both = [
        _signal("vibration", "cusum", DriftSeverity.ALARM),
        _signal("vibration", "ewma", DriftSeverity.WARNING),
    ]
    assert drift_penalty(both, small_health_config) == pytest.approx(alarm_points)
    assert drift_penalty(
        [_signal("vibration", "ewma", DriftSeverity.WARNING)], small_health_config
    ) == pytest.approx(warning_points)


def test_penalty_is_capped_so_drift_alone_cannot_manufacture_a_critical(
    small_health_config: Config,
) -> None:
    maximum = float(small_health_config.get("health.drift.penalty.max_points", 15.0))
    degraded_min = float(small_health_config.get("health.classes.degraded_min", 40.0))

    many = [_signal(f"sensor_{index}", "cusum", DriftSeverity.ALARM) for index in range(20)]
    penalty = drift_penalty(many, small_health_config)
    assert penalty == pytest.approx(maximum)
    # A perfect machine cannot be pushed into Critical by drift alone.
    assert 100.0 - penalty > degraded_min


def test_penalty_of_no_signals_is_zero(small_health_config: Config) -> None:
    assert drift_penalty([], small_health_config) == 0.0


def test_multivariate_signal_adds_its_own_points(small_health_config: Config) -> None:
    points = float(small_health_config.get("health.drift.penalty.multivariate_points", 5.0))
    signal = _signal("multivariate", "isolation_forest", DriftSeverity.WARNING)
    assert drift_penalty([signal], small_health_config) == pytest.approx(points)


# ---------------------------------------------------------------------------
# Multivariate detector and reporting
# ---------------------------------------------------------------------------
def test_multivariate_detector_scores_into_the_unit_interval(
    small_health_config: Config, health_prepared
) -> None:
    labelled, _, _ = health_prepared
    statistics = compute_drift_statistics(labelled, small_health_config)
    detector = MultivariateDriftDetector.fit(statistics, small_health_config)
    assert detector is not None

    scores = detector.score(statistics).dropna()
    assert not scores.empty
    assert scores.between(0.0, 1.0).all()
    assert detector.to_dict()["n_columns"] == len(detector.columns)


def test_multivariate_detector_rejects_missing_columns(
    small_health_config: Config, health_prepared
) -> None:
    labelled, _, _ = health_prepared
    statistics = compute_drift_statistics(labelled, small_health_config)
    detector = MultivariateDriftDetector.fit(statistics, small_health_config)
    assert detector is not None
    with pytest.raises(ValueError, match="missing"):
        detector.score(pd.DataFrame({"unrelated": [0.0]}))


def test_multivariate_detector_is_none_when_disabled(
    small_health_config: Config, health_prepared
) -> None:
    data = small_health_config.to_dict()
    data["health"]["drift"]["isolation_forest"]["enabled"] = False
    labelled, _, _ = health_prepared
    statistics = compute_drift_statistics(labelled, small_health_config)
    assert MultivariateDriftDetector.fit(statistics, Config(data)) is None


def test_drift_report_is_tabular_and_empty_when_nothing_fires(
    small_health_config: Config, health_prepared
) -> None:
    labelled, _, _ = health_prepared
    statistics = compute_drift_statistics(labelled, small_health_config)
    report = drift_report(statistics, labelled, small_health_config)

    assert isinstance(report, pd.DataFrame)
    if not report.empty:
        assert {TURBINE_ID, "sensor", "method", "severity"} <= set(report.columns)
        assert set(report["severity"]) <= {"warning", "alarm"}

    assert drift_report(pd.DataFrame(), pd.DataFrame(), small_health_config).empty


def test_drift_statistics_are_leakage_safe(small_health_config: Config, health_toy_frame) -> None:
    from wind_turbine_pm.data.preprocessing import preprocess
    from wind_turbine_pm.features.transformers import assert_no_future_leakage
    from wind_turbine_pm.health.regimes import attach_regimes

    prepared, _ = preprocess(health_toy_frame, small_health_config)
    prepared = attach_regimes(prepared, small_health_config)
    assert_no_future_leakage(
        prepared,
        lambda frame: compute_drift_statistics(frame, small_health_config),
        TURBINE_ID,
        TIMESTAMP,
    )
