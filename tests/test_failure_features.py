"""Tests for leakage-safe feature engineering.

The three properties that matter most:

1. No information moves between turbines.
2. No information moves backwards in time.
3. The feature set and its order are stable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wind_turbine_pm.constants import TIMESTAMP, TURBINE_ID
from wind_turbine_pm.features.failure_features import (
    align_to_feature_order,
    build_failure_features,
    minimum_history_rows,
)
from wind_turbine_pm.features.transformers import (
    expanding_baseline,
    grouped_lag,
    grouped_rolling,
    robust_zscore,
    rolling_slope,
)


def test_expected_feature_groups_exist(feature_matrix, prepared):
    """All configured feature families must be produced."""
    _, _, _ = prepared
    columns = set(feature_matrix.columns)
    assert "vibration" in columns  # raw
    assert any(c.endswith("_lag_3h") for c in columns)  # lag
    assert any("_roll_mean_" in c for c in columns)  # rolling
    assert any("_slope_" in c for c in columns)  # trend
    assert "gearbox_temp_above_ambient" in columns  # interaction
    assert any(c.endswith("_turbine_robust_z") for c in columns)  # turbine-relative
    assert "hour_sin" in columns  # time


def test_no_target_or_key_columns_leak_into_features(feature_matrix):
    """Identifiers, events and the label must never be model features."""
    forbidden = {
        "failure_within_48h",
        "failure_event",
        "maintenance_event",
        "degradation_level",
        "hours_to_failure",
        "episode_id",
        "failure_mode",
        TURBINE_ID,
        TIMESTAMP,
    }
    assert forbidden.isdisjoint(set(feature_matrix.columns))


def test_features_are_deterministic(labelled_frame, small_config):
    """Rebuilding features on identical input must give identical output."""
    first, spec_a = build_failure_features(labelled_frame, small_config)
    second, spec_b = build_failure_features(labelled_frame, small_config)
    pd.testing.assert_frame_equal(first, second)
    assert spec_a.names == spec_b.names


def test_feature_order_is_stable_under_row_shuffling(labelled_frame, small_config):
    """Input row order must not change the feature column order or values."""
    shuffled = labelled_frame.sample(frac=1.0, random_state=11)
    ordered_features, ordered_spec = build_failure_features(labelled_frame, small_config)
    shuffled_features, shuffled_spec = build_failure_features(shuffled, small_config)

    assert ordered_spec.names == shuffled_spec.names
    # Values must match once realigned to the same index.
    pd.testing.assert_frame_equal(
        ordered_features.sort_index(), shuffled_features.sort_index(), check_like=False
    )


def test_no_leakage_between_turbines(small_config, toy_frame):
    """Changing turbine B must never alter any feature of turbine A."""
    baseline, _ = build_failure_features(toy_frame, small_config)

    perturbed_input = toy_frame.copy()
    mask_b = perturbed_input[TURBINE_ID] == "B"
    numeric = perturbed_input.select_dtypes(include=[np.number]).columns
    perturbed_input.loc[mask_b, numeric] = perturbed_input.loc[mask_b, numeric] + 500.0
    perturbed, _ = build_failure_features(perturbed_input, small_config)

    mask_a = toy_frame[TURBINE_ID] == "A"
    left = baseline.loc[mask_a.to_numpy()]
    right = perturbed.loc[mask_a.to_numpy()]
    difference = (left - right).abs().max().max()
    assert np.isnan(difference) or difference < 1e-5


def test_no_future_leakage(small_config, toy_frame):
    """Perturbing the last observation must not change any earlier feature."""
    baseline, _ = build_failure_features(toy_frame, small_config)

    perturbed_input = toy_frame.copy()
    last_position = perturbed_input.index[perturbed_input[TURBINE_ID] == "A"][-1]
    numeric = perturbed_input.select_dtypes(include=[np.number]).columns
    perturbed_input.loc[last_position, numeric] = (
        perturbed_input.loc[last_position, numeric].astype(float) + 1000.0
    )
    perturbed, _ = build_failure_features(perturbed_input, small_config)

    earlier = baseline.index != last_position
    difference = (baseline.loc[earlier] - perturbed.loc[earlier]).abs().max().max()
    assert np.isnan(difference) or difference < 1e-5, (
        "a past feature changed when the future changed"
    )


def test_truncating_the_future_does_not_change_past_features(small_config, toy_frame):
    """Features for the first N rows must not depend on rows after N."""
    full, _ = build_failure_features(toy_frame, small_config)
    cutoff = 120
    truncated_input = toy_frame.groupby(TURBINE_ID, group_keys=False).head(cutoff)
    truncated, _ = build_failure_features(truncated_input, small_config)

    common = truncated.index
    difference = (full.loc[common] - truncated).abs().max().max()
    assert np.isnan(difference) or difference < 1e-5


# ---------------------------------------------------------------------------
# Primitive-level tests
# ---------------------------------------------------------------------------
def test_grouped_lag_respects_groups():
    """A lag must not pull a value across a group boundary."""
    series = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    groups = pd.Series(["A", "A", "A", "B", "B", "B"])
    lagged = grouped_lag(series, groups, 1)
    assert pd.isna(lagged.iloc[0])
    assert pd.isna(lagged.iloc[3]), "lag crossed a group boundary"
    assert lagged.iloc[4] == 10.0


def test_rolling_window_is_trailing():
    """A rolling mean at position t must use only rows up to t."""
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    groups = pd.Series(["A"] * 5)
    rolled = grouped_rolling(series, groups, window=3, stat="mean", min_periods=1)
    assert rolled.iloc[0] == 1.0
    assert rolled.iloc[2] == pytest.approx(2.0)  # mean(1,2,3), not affected by 100
    assert rolled.iloc[3] == pytest.approx(3.0)  # mean(2,3,4)


def test_rolling_range_is_max_minus_min():
    """The 'range' statistic must equal max - min over the window."""
    series = pd.Series([1.0, 5.0, 3.0, 9.0])
    groups = pd.Series(["A"] * 4)
    rolled = grouped_rolling(series, groups, window=3, stat="range", min_periods=1)
    assert rolled.iloc[2] == pytest.approx(4.0)  # max(1,5,3) - min(1,5,3)
    assert rolled.iloc[3] == pytest.approx(6.0)  # max(5,3,9) - min(5,3,9)


def test_rolling_slope_detects_a_linear_trend():
    """A perfectly linear series must produce a slope equal to its increment."""
    series = pd.Series(np.arange(20, dtype=float) * 2.0)
    groups = pd.Series(["A"] * 20)
    slope = rolling_slope(series, groups, window=6)
    assert slope.iloc[-1] == pytest.approx(2.0, abs=1e-6)


def test_expanding_baseline_excludes_the_current_row():
    """The expanding baseline must be computed from strictly past rows."""
    series = pd.Series([1.0, 1.0, 1.0, 100.0])
    groups = pd.Series(["A"] * 4)
    baseline = expanding_baseline(series, groups, "mean", min_periods=1)
    # At the last row the baseline is the mean of the first three ones, not
    # contaminated by the 100 at that same position.
    assert baseline.iloc[3] == pytest.approx(1.0)


def test_robust_zscore_flags_an_outlier():
    """A large departure from a turbine's own history must score highly."""
    series = pd.Series([5.0] * 30 + [50.0])
    groups = pd.Series(["A"] * 31)
    z = robust_zscore(series, groups, min_periods=5, floor=1e-6)
    assert abs(z.iloc[-1]) > 10.0


def test_missing_values_are_handled(feature_matrix):
    """Features must be finite or NaN, never infinite."""
    numeric = feature_matrix.to_numpy(dtype=float)
    assert not np.isinf(numeric).any()


def test_features_are_float32(feature_matrix):
    """The matrix must use a compact, consistent dtype."""
    assert (feature_matrix.dtypes == np.float32).all()


def test_align_to_feature_order_reorders():
    """Alignment must reorder columns to the model's expected order."""
    frame = pd.DataFrame({"b": [1.0], "a": [2.0], "c": [3.0]})
    aligned = align_to_feature_order(frame, ["a", "b"])
    assert list(aligned.columns) == ["a", "b"]


def test_align_to_feature_order_rejects_missing_features():
    """Alignment must fail loudly when a required feature is absent."""
    frame = pd.DataFrame({"a": [1.0]})
    with pytest.raises(ValueError, match="missing"):
        align_to_feature_order(frame, ["a", "missing_feature"])


def test_minimum_history_rows_matches_longest_window(small_config):
    """Required history must equal the longest configured window."""
    assert minimum_history_rows(small_config, step_hours=1.0) == 12


def test_empty_frame_is_rejected(small_config):
    """Building features from nothing must raise rather than return garbage."""
    with pytest.raises(ValueError, match="empty"):
        build_failure_features(pd.DataFrame(columns=[TURBINE_ID, TIMESTAMP]), small_config)


def test_calendar_components_are_configurable(small_config, toy_frame):
    """Only the configured calendar components may be emitted."""
    from wind_turbine_pm.config import Config

    default_features, _ = build_failure_features(toy_frame, small_config)
    # `month` and `day_of_week` cannot generalise across a chronological split
    # and are excluded by default; see configs/features.yaml.
    assert "hour" in default_features.columns
    assert "hour_sin" in default_features.columns
    assert "month" not in default_features.columns
    assert "day_of_week" not in default_features.columns

    data = small_config.to_dict()
    data["features"]["time_features"]["components"] = ["hour", "month"]
    widened, _ = build_failure_features(toy_frame, Config(data))
    assert "month" in widened.columns
    assert "month_cos" in widened.columns


def test_unknown_calendar_component_is_rejected(small_config, toy_frame):
    """A misconfigured calendar component must fail loudly."""
    from wind_turbine_pm.config import Config

    data = small_config.to_dict()
    data["features"]["time_features"]["components"] = ["fortnight"]
    with pytest.raises(ValueError, match="Unknown time_features.components"):
        build_failure_features(toy_frame, Config(data))
