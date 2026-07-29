"""Tests for health-score feature engineering.

The single most important property is that no feature reads the future: the
health target is derived from a degradation state that trends towards a failure,
so a feature that peeks forward would produce excellent measured error and be
worthless in service.  :func:`assert_no_future_leakage` checks that empirically
rather than by inspection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import OPERATING_REGIME, TIMESTAMP, TURBINE_ID, OperatingRegime
from wind_turbine_pm.features.transformers import assert_no_future_leakage
from wind_turbine_pm.health.health_features import (
    align_to_feature_order,
    build_health_features,
    minimum_history_hours,
)
from wind_turbine_pm.health.regimes import attach_regimes


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------
def test_no_feature_reads_the_future(small_health_config: Config, health_toy_frame) -> None:
    from wind_turbine_pm.data.preprocessing import preprocess

    prepared, _ = preprocess(health_toy_frame, small_health_config)
    prepared = attach_regimes(prepared, small_health_config)

    def build(frame: pd.DataFrame) -> pd.DataFrame:
        features, _ = build_health_features(frame, small_health_config)
        return features

    # Perturbing the final row must not change any earlier row's features.
    assert_no_future_leakage(prepared, build, TURBINE_ID, TIMESTAMP)


def test_features_never_cross_a_turbine_boundary(
    small_health_config: Config, health_toy_frame
) -> None:
    from wind_turbine_pm.data.preprocessing import preprocess

    prepared, _ = preprocess(health_toy_frame, small_health_config)
    baseline, _ = build_health_features(prepared, small_health_config)

    # Change turbine B's history entirely; turbine A's features must not move.
    mutated = prepared.copy()
    mask = mutated[TURBINE_ID] == "B"
    numeric = mutated.select_dtypes(include=[np.number]).columns
    mutated.loc[mask, numeric] = mutated.loc[mask, numeric] + 25.0
    perturbed, _ = build_health_features(mutated, small_health_config)

    rows_a = prepared.index[prepared[TURBINE_ID] == "A"]
    difference = (baseline.loc[rows_a] - perturbed.loc[rows_a]).abs().max().max()
    assert difference == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------
def test_target_and_ground_truth_columns_are_absent_from_the_matrix(
    small_health_config: Config, health_features: pd.DataFrame
) -> None:
    # The label is derived from degradation_level; letting it in as a feature
    # would make the model learn its own input.
    forbidden = set(small_health_config.get("health.features.exclude_columns", []))
    assert forbidden, "the configuration must list excluded columns"
    assert not (forbidden & set(health_features.columns))
    assert "degradation_level" not in health_features.columns
    assert "health_score_target" not in health_features.columns
    assert "failure_event" not in health_features.columns


def test_matrix_is_numeric_finite_typed_and_index_aligned(
    small_health_config: Config, health_prepared
) -> None:
    labelled, features, _ = health_prepared
    assert features.index.equals(labelled.index)
    assert (features.dtypes == np.float32).all()
    assert not np.isinf(features.to_numpy(dtype=np.float64, na_value=0.0)).any()


def test_spec_records_the_exact_feature_order_and_groups(
    small_health_config: Config, health_toy_frame
) -> None:
    from wind_turbine_pm.data.preprocessing import preprocess

    prepared, _ = preprocess(health_toy_frame, small_health_config)
    features, spec = build_health_features(prepared, small_health_config)

    assert list(spec.names) == list(features.columns)
    assert spec.to_dict()["n_features"] == len(features.columns)
    assert spec.step_hours == pytest.approx(1.0)
    for group, members in spec.groups.items():
        assert members, f"group {group} is empty"
        assert set(members) <= set(features.columns)
    # Every feature belongs to exactly one group.
    flattened = [name for members in spec.groups.values() for name in members]
    assert sorted(flattened) == sorted(features.columns)


def test_expected_feature_groups_are_present(small_health_config: Config, health_toy_frame) -> None:
    from wind_turbine_pm.data.preprocessing import preprocess

    prepared, _ = preprocess(health_toy_frame, small_health_config)
    _, spec = build_health_features(prepared, small_health_config)
    assert {"condition", "rolling", "trend", "rule", "physical"} <= set(spec.groups)


def test_regime_is_one_hot_encoded_over_every_regime(
    small_health_config: Config, health_toy_frame
) -> None:
    from wind_turbine_pm.data.preprocessing import preprocess

    prepared, _ = preprocess(health_toy_frame, small_health_config)
    features, _ = build_health_features(prepared, small_health_config)

    columns = [f"regime_{regime.value}" for regime in OperatingRegime]
    assert set(columns) <= set(features.columns)
    # Exactly one regime is active per row, whatever the data looks like.
    assert (features[columns].sum(axis=1) == 1).all()


def test_regime_is_attached_when_absent(small_health_config: Config, health_toy_frame) -> None:
    from wind_turbine_pm.data.preprocessing import preprocess

    prepared, _ = preprocess(health_toy_frame, small_health_config)
    assert OPERATING_REGIME not in prepared.columns
    # Must not raise: the builder attaches the regime itself so a caller cannot
    # accidentally build features against a different assignment than training.
    features, _ = build_health_features(prepared, small_health_config)
    assert len(features) == len(prepared)


def test_building_is_deterministic(small_health_config: Config, health_toy_frame) -> None:
    from wind_turbine_pm.data.preprocessing import preprocess

    prepared, _ = preprocess(health_toy_frame, small_health_config)
    first, _ = build_health_features(prepared, small_health_config)
    second, _ = build_health_features(prepared, small_health_config)
    pd.testing.assert_frame_equal(first, second)


def test_row_order_does_not_change_a_row_s_features(
    small_health_config: Config, health_toy_frame
) -> None:
    from wind_turbine_pm.data.preprocessing import preprocess

    prepared, _ = preprocess(health_toy_frame, small_health_config)
    ordered, _ = build_health_features(prepared, small_health_config)

    shuffled = prepared.sample(frac=1.0, random_state=3)
    shuffled_features, _ = build_health_features(shuffled, small_health_config)

    # The builder sorts internally and reindexes to the input, so the same
    # observation must receive the same values whatever order it arrived in.
    pd.testing.assert_frame_equal(
        ordered.sort_index(), shuffled_features.sort_index(), check_like=True
    )


def test_empty_and_keyless_frames_are_rejected(small_health_config: Config) -> None:
    with pytest.raises(ValueError, match="turbine_id"):
        build_health_features(pd.DataFrame({"vibration": [1.0]}), small_health_config)
    with pytest.raises(ValueError, match="empty"):
        build_health_features(pd.DataFrame({TURBINE_ID: [], TIMESTAMP: []}), small_health_config)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def test_align_to_feature_order_reorders_and_subsets(health_features: pd.DataFrame) -> None:
    expected = list(health_features.columns[:5])[::-1]
    aligned = align_to_feature_order(health_features, expected)
    assert list(aligned.columns) == expected


def test_align_to_feature_order_reports_missing_features(
    health_features: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="missing 1 expected feature"):
        align_to_feature_order(health_features, ["definitely_not_a_feature"])


def test_minimum_history_hours_is_the_longest_configured_window(
    small_health_config: Config,
) -> None:
    hours = minimum_history_hours(small_health_config)
    candidates = [
        *[float(h) for h in small_health_config.require("health.features.windows_hours")],
        *[float(h) for h in small_health_config.require("health.features.slope_windows_hours")],
        *[float(h) for h in small_health_config.require("health.features.diff_hours")],
    ]
    assert hours == max(candidates + [hours])
    assert hours >= max(candidates)


def test_signal_shape_indicators_are_produced(
    small_health_config: Config, health_features: pd.DataFrame
) -> None:
    window = int(small_health_config.get("health.features.signal_shape.window_hours", 24))
    # The classic vibration condition indicators, computed on a trailing window.
    for suffix in ("rms", "crest", "kurtosis", "skew", "p2p", "hf_ratio", "zcr"):
        assert f"vibration_{suffix}_{window}h" in health_features.columns


def test_kurtosis_is_excess_kurtosis(small_health_config: Config) -> None:
    # Excess kurtosis is 0 for a Gaussian; an incipient bearing defect makes the
    # signal impulsive and drives it positive. If the constant were wrong every
    # value would be offset by 3 and the feature would be misread.
    from wind_turbine_pm.data.preprocessing import preprocess

    rng = np.random.default_rng(11)
    hours = 400
    start = pd.Timestamp("2024-01-01")
    frame = pd.DataFrame(
        {
            TIMESTAMP: [start + pd.Timedelta(hours=h) for h in range(hours)],
            TURBINE_ID: ["A"] * hours,
            "vibration": 3.0 + rng.normal(0, 0.3, hours),
        }
    )
    for column, value in (
        ("wind_speed", 8.0),
        ("rotor_speed", 12.0),
        ("generator_speed", 1160.0),
        ("power_output", 700.0),
        ("generator_temperature", 60.0),
        ("gearbox_temperature", 54.0),
        ("bearing_temperature", 47.0),
        ("oil_temperature", 45.0),
        ("oil_pressure", 5.0),
        ("ambient_temperature", 10.0),
        ("nacelle_temperature", 18.0),
        ("hydraulic_pressure", 190.0),
        ("brake_temperature", 16.0),
        ("operational_status", "normal"),
        ("failure_event", 0),
    ):
        frame[column] = value

    prepared, _ = preprocess(frame, small_health_config)
    features, _ = build_health_features(prepared, small_health_config)
    window = int(small_health_config.get("health.features.signal_shape.window_hours", 24))
    kurtosis = features[f"vibration_kurtosis_{window}h"].dropna()

    # Gaussian input: excess kurtosis centred near zero, not near three.
    assert abs(float(kurtosis.median())) < 1.5
