"""Tests for operating-regime detection.

The regime label is load-bearing: every regime-relative feature, every drift
baseline and every component score is conditioned on it, so a mislabelled regime
propagates silently into the score.  The tests pin the precedence order of the
threshold rules, the agreement of the power curve with the Failure Prediction
Module, and the invariants both methods must satisfy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import (
    OPERATING_REGIME,
    OPERATIONAL_STATUS,
    PRODUCING_REGIMES,
    TIMESTAMP,
    TURBINE_ID,
    OperatingRegime,
)
from wind_turbine_pm.health.regimes import (
    RegimeThresholds,
    assign_regimes,
    assign_regimes_kmeans,
    assign_regimes_threshold,
    attach_regimes,
    reference_power,
    regime_summary,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    start = pd.Timestamp("2024-01-01")
    frame = pd.DataFrame(rows)
    frame[TIMESTAMP] = [start + pd.Timedelta(hours=i) for i in range(len(frame))]
    frame[TURBINE_ID] = "A"
    return frame


# ---------------------------------------------------------------------------
# Power curve
# ---------------------------------------------------------------------------
def test_reference_power_reads_the_shared_failure_module_configuration(
    health_config: Config,
) -> None:
    # Both modules must describe the same turbine with the same rated curve;
    # reading the same keys is what guarantees it.
    rated_power = float(health_config.require("features.interactions.rated_power_kw"))
    rated_wind = float(health_config.require("features.interactions.rated_wind_speed"))
    cut_in = float(health_config.require("features.interactions.cut_in_wind_speed"))
    cut_out = float(health_config.require("features.interactions.cut_out_wind_speed"))

    wind = pd.Series([0.0, cut_in - 0.1, rated_wind, rated_wind + 2.0, cut_out + 1.0])
    expected = reference_power(wind, health_config)

    assert expected.iloc[0] == 0.0
    assert expected.iloc[1] == 0.0  # below cut-in
    assert expected.iloc[2] == pytest.approx(rated_power)
    assert expected.iloc[3] == pytest.approx(rated_power)
    assert expected.iloc[4] == 0.0  # above cut-out


def test_reference_power_is_monotonic_through_the_ramp(health_config: Config) -> None:
    wind = pd.Series(np.linspace(3.0, 12.5, 40))
    expected = reference_power(wind, health_config)
    assert (expected.diff().dropna() >= -1e-9).all()


# ---------------------------------------------------------------------------
# Threshold method
# ---------------------------------------------------------------------------
def test_thresholds_load_from_configuration(health_config: Config) -> None:
    thresholds = RegimeThresholds.from_config(health_config)
    assert thresholds.cut_in_wind_speed < thresholds.low_medium_edge
    assert thresholds.low_medium_edge < thresholds.medium_high_edge
    assert thresholds.medium_high_edge < thresholds.cut_out_wind_speed
    assert 0.0 < thresholds.curtailment_power_ratio < 1.0


def test_controller_fault_outranks_every_other_signal(health_config: Config) -> None:
    # The controller reporting a fault is a fact, not something to be inferred
    # from the sensors, so it must win regardless of how the machine looks.
    frame = _frame(
        [
            {"wind_speed": 12.0, "power_output": 1800.0, OPERATIONAL_STATUS: "fault"},
            {"wind_speed": 12.0, "power_output": 1800.0, OPERATIONAL_STATUS: "maintenance"},
            {"wind_speed": 12.0, "power_output": 1800.0, OPERATIONAL_STATUS: "normal"},
        ]
    )
    regimes = assign_regimes_threshold(frame, health_config)
    assert regimes.iloc[0] == str(OperatingRegime.OFFLINE)
    assert regimes.iloc[1] == str(OperatingRegime.OFFLINE)
    assert regimes.iloc[2] != str(OperatingRegime.OFFLINE)


def test_below_cut_in_and_above_cut_out_are_idle(health_config: Config) -> None:
    frame = _frame(
        [
            {"wind_speed": 1.0, "power_output": 0.0, OPERATIONAL_STATUS: "normal"},
            {"wind_speed": 30.0, "power_output": 0.0, OPERATIONAL_STATUS: "normal"},
        ]
    )
    regimes = assign_regimes_threshold(frame, health_config)
    assert (regimes == str(OperatingRegime.IDLE)).all()


def test_no_meaningful_output_is_idle_even_in_strong_wind(health_config: Config) -> None:
    frame = _frame([{"wind_speed": 14.0, "power_output": 5.0, OPERATIONAL_STATUS: "normal"}])
    assert assign_regimes_threshold(frame, health_config).iloc[0] == str(OperatingRegime.IDLE)


def test_load_bands_follow_wind_speed(health_config: Config) -> None:
    thresholds = RegimeThresholds.from_config(health_config)
    # Power is set on the reference curve so nothing is mistaken for curtailment,
    # and each wind speed is chosen to produce above the idle power floor (see
    # test_low_wind_on_the_curve_is_idle_not_low_load for why that matters).
    rows = []
    for wind in (6.5, 9.0, 13.0):
        expected = float(reference_power(pd.Series([wind]), health_config).iloc[0])
        assert expected > thresholds.idle_power_kw
        rows.append({"wind_speed": wind, "power_output": expected, OPERATIONAL_STATUS: "normal"})
    regimes = assign_regimes_threshold(_frame(rows), health_config)

    assert regimes.iloc[0] == str(OperatingRegime.LOW_LOAD)
    assert regimes.iloc[1] == str(OperatingRegime.MEDIUM_LOAD)
    assert regimes.iloc[2] == str(OperatingRegime.HIGH_LOAD)
    assert thresholds.low_medium_edge <= 9.0 < thresholds.medium_high_edge


def test_low_wind_on_the_curve_is_idle_not_low_load(health_config: Config) -> None:
    """Idle is defined by output, not by wind speed, and output wins.

    The power curve is cubic through the ramp, so just above cut-in a perfectly
    healthy machine produces only a few tens of kW. At 5 m/s the reference curve
    gives roughly 19 kW, which is below the 25 kW idle floor, so the observation
    is labelled ``idle`` even though the wind is above cut-in. That is the
    intended reading - a machine making 19 kW is not meaningfully producing - and
    it means the ``low_load`` band effectively starts where the curve crosses the
    idle floor rather than at cut-in.
    """
    thresholds = RegimeThresholds.from_config(health_config)
    expected = float(reference_power(pd.Series([5.0]), health_config).iloc[0])
    assert expected < thresholds.idle_power_kw

    frame = _frame([{"wind_speed": 5.0, "power_output": expected, OPERATIONAL_STATUS: "normal"}])
    assert assign_regimes_threshold(frame, health_config).iloc[0] == str(OperatingRegime.IDLE)


def test_producing_well_below_the_curve_is_curtailed_not_a_load_band(
    health_config: Config,
) -> None:
    # A curtailed machine looks "cold for its wind speed" and would otherwise
    # poison the load-band baseline it was pooled into.
    wind = 13.0
    expected = float(reference_power(pd.Series([wind]), health_config).iloc[0])
    ratio = float(health_config.get("health.operating_regimes.curtailment_power_ratio", 0.55))
    frame = _frame(
        [
            {
                "wind_speed": wind,
                "power_output": expected * (ratio - 0.15),
                OPERATIONAL_STATUS: "normal",
            }
        ]
    )
    assert assign_regimes_threshold(frame, health_config).iloc[0] == str(OperatingRegime.CURTAILED)


def test_every_label_is_a_known_regime(small_health_config: Config, health_prepared) -> None:
    labelled, _, _ = health_prepared
    known = {str(regime) for regime in OperatingRegime}
    assert set(labelled[OPERATING_REGIME].unique()) <= known


def test_unknown_method_is_rejected(health_config: Config) -> None:
    data = health_config.to_dict()
    data["health"]["operating_regimes"]["method"] = "astrology"
    frame = _frame([{"wind_speed": 8.0, "power_output": 700.0, OPERATIONAL_STATUS: "normal"}])
    with pytest.raises(ValueError, match="Unknown health.operating_regimes.method"):
        assign_regimes(frame, Config(data))


# ---------------------------------------------------------------------------
# K-means method
# ---------------------------------------------------------------------------
def test_kmeans_produces_valid_labels_and_respects_controller_state(
    small_health_config: Config, health_prepared
) -> None:
    labelled, _, _ = health_prepared
    subset = labelled.iloc[:2000].copy()
    regimes = assign_regimes_kmeans(subset, small_health_config)

    assert len(regimes) == len(subset)
    assert set(regimes.unique()) <= {str(regime) for regime in OperatingRegime}
    offline = subset[OPERATIONAL_STATUS].astype(str).isin({"maintenance", "fault"})
    if offline.any():
        assert (regimes[offline] == str(OperatingRegime.OFFLINE)).all()


def test_kmeans_orders_clusters_by_power(small_health_config: Config, health_prepared) -> None:
    # The labels only mean something if higher bands really carry more power.
    labelled, _, _ = health_prepared
    subset = labelled.iloc[:3000].copy()
    regimes = assign_regimes_kmeans(subset, small_health_config)
    means = subset.groupby(regimes.to_numpy())["power_output"].mean()

    ladder = [
        str(OperatingRegime.IDLE),
        str(OperatingRegime.LOW_LOAD),
        str(OperatingRegime.MEDIUM_LOAD),
        str(OperatingRegime.HIGH_LOAD),
    ]
    present = [name for name in ladder if name in means.index]
    ordered = means.loc[present].to_numpy()
    assert (np.diff(ordered) >= 0).all(), f"cluster power means are not ordered: {means.to_dict()}"


def test_kmeans_without_any_configured_feature_raises(health_config: Config) -> None:
    data = health_config.to_dict()
    data["health"]["operating_regimes"]["kmeans"]["features"] = ["not_a_column"]
    frame = _frame([{"wind_speed": 8.0, "power_output": 700.0, OPERATIONAL_STATUS: "normal"}])
    with pytest.raises(ValueError, match="at least one of the configured features"):
        assign_regimes_kmeans(frame, Config(data))


# ---------------------------------------------------------------------------
# Attachment and summary
# ---------------------------------------------------------------------------
def test_attach_regimes_does_not_mutate_the_input(health_config: Config) -> None:
    frame = _frame([{"wind_speed": 8.0, "power_output": 700.0, OPERATIONAL_STATUS: "normal"}])
    out = attach_regimes(frame, health_config)
    assert OPERATING_REGIME in out.columns
    assert OPERATING_REGIME not in frame.columns


def test_regime_summary_covers_every_regime_and_sums_to_one(health_prepared) -> None:
    labelled, _, _ = health_prepared
    summary = regime_summary(labelled)
    assert list(summary[OPERATING_REGIME]) == [str(regime) for regime in OperatingRegime]
    assert summary["observations"].sum() == len(labelled)
    assert summary["share"].sum() == pytest.approx(1.0)


def test_regime_summary_requires_the_column() -> None:
    with pytest.raises(KeyError, match=OPERATING_REGIME):
        regime_summary(pd.DataFrame({"a": [1]}))


def test_producing_regimes_are_a_subset_of_all_regimes() -> None:
    assert set(PRODUCING_REGIMES) < set(OperatingRegime)
    assert OperatingRegime.OFFLINE not in PRODUCING_REGIMES
    assert OperatingRegime.IDLE not in PRODUCING_REGIMES
