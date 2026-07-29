"""Tests for the 48-hour failure target.

These are the highest-value tests in the suite: a wrong label silently makes
every downstream metric meaningless.
"""

from __future__ import annotations

import pandas as pd
import pytest

from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import FAILURE_EVENT, TARGET_COLUMN, TIMESTAMP, TURBINE_ID
from wind_turbine_pm.data.preprocessing import (
    apply_modelling_filter,
    create_failure_target,
    infer_step_hours,
)


def test_target_marks_exactly_the_horizon_before_failure(toy_frame, small_config):
    """Rows in ``(failure - 48h, failure)`` are positive; the rest are not."""
    labelled = create_failure_target(toy_frame, small_config)
    turbine_a = (
        labelled.loc[labelled[TURBINE_ID] == "A"].sort_values(TIMESTAMP).reset_index(drop=True)
    )

    failure_time = turbine_a.loc[turbine_a[FAILURE_EVENT] == 1, TIMESTAMP].iloc[0]
    horizon = pd.Timedelta(hours=48)

    expected = (
        (turbine_a[TIMESTAMP] < failure_time) & (turbine_a[TIMESTAMP] >= failure_time - horizon)
    ).astype(int)
    pd.testing.assert_series_equal(
        turbine_a[TARGET_COLUMN].astype(int), expected, check_names=False
    )


def test_failure_row_itself_is_not_labelled_positive(toy_frame, small_config):
    """A failure at time t is not 'within the next 48 hours' of t."""
    labelled = create_failure_target(toy_frame, small_config)
    failure_rows = labelled.loc[labelled[FAILURE_EVENT] == 1]
    assert (failure_rows[TARGET_COLUMN] == 0).all()


def test_target_count_matches_horizon(toy_frame, small_config):
    """With hourly data and a 48-hour horizon there must be exactly 48 positives."""
    labelled = create_failure_target(toy_frame, small_config)
    turbine_a = labelled.loc[labelled[TURBINE_ID] == "A"]
    assert int(turbine_a[TARGET_COLUMN].sum()) == 48


def test_target_does_not_cross_turbine_boundaries(toy_frame, small_config):
    """A failure on turbine A must never label rows on turbine B."""
    labelled = create_failure_target(toy_frame, small_config)
    turbine_b = labelled.loc[labelled[TURBINE_ID] == "B"]
    assert turbine_b[FAILURE_EVENT].sum() == 0
    assert turbine_b[TARGET_COLUMN].sum() == 0


def test_turbine_order_does_not_affect_labels(toy_frame, small_config):
    """Shuffling input row order must not change the resulting labels."""
    shuffled = toy_frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    from_ordered = create_failure_target(toy_frame, small_config)
    from_shuffled = create_failure_target(shuffled, small_config)

    key = [TURBINE_ID, TIMESTAMP]
    left = from_ordered.set_index(key)[TARGET_COLUMN].sort_index()
    right = from_shuffled.set_index(key)[TARGET_COLUMN].sort_index()
    pd.testing.assert_series_equal(left, right)


def test_tail_of_timeline_is_flagged_unreliable(toy_frame, small_config):
    """The final 48 hours of each turbine cannot be labelled and are excluded."""
    labelled = create_failure_target(toy_frame, small_config)
    for _, group in labelled.groupby(TURBINE_ID):
        ordered = group.sort_values(TIMESTAMP)
        last_time = ordered[TIMESTAMP].max()
        tail = ordered.loc[ordered[TIMESTAMP] > last_time - pd.Timedelta(hours=48)]
        assert not tail["label_reliable"].any()
        head = ordered.loc[ordered[TIMESTAMP] <= last_time - pd.Timedelta(hours=48)]
        assert head["label_reliable"].all()


def test_modelling_filter_drops_unreliable_and_excluded_rows(toy_frame, small_config):
    """The eligibility filter must remove exactly the flagged rows."""
    labelled = create_failure_target(toy_frame, small_config)
    eligible = apply_modelling_filter(labelled)
    assert len(eligible) == int(labelled["modelling_eligible"].sum())
    assert "modelling_eligible" not in eligible.columns
    assert "label_reliable" not in eligible.columns


def test_fault_and_maintenance_rows_are_excluded(toy_frame, small_config):
    """Rows recorded during fault/maintenance leak the outcome and are dropped."""
    frame = toy_frame.copy()
    frame.loc[frame.index[50:60], "operational_status"] = "fault"
    labelled = create_failure_target(frame, small_config)
    excluded = labelled.iloc[50:60]
    assert not excluded["modelling_eligible"].any()


def test_post_repair_window_is_excluded(toy_frame, small_config):
    """Rows shortly after a maintenance window are dropped as transient."""
    data = small_config.to_dict()
    data["target"]["post_repair_exclusion_hours"] = 6
    cfg = Config(data)

    frame = toy_frame.copy()
    frame.loc[frame.index[40:44], "operational_status"] = "maintenance"
    labelled = create_failure_target(frame, cfg)

    # The six rows immediately after the maintenance window must be excluded.
    following = labelled.iloc[44:50]
    assert not following["modelling_eligible"].any()
    # Rows well clear of it remain eligible.
    assert labelled.iloc[60:70]["modelling_eligible"].all()


def test_consecutive_failures_produce_disjoint_windows(small_config):
    """Two failures far apart must produce two separate positive windows."""
    start = pd.Timestamp("2024-01-01")
    rows = []
    for hour in range(400):
        rows.append(
            {
                TIMESTAMP: start + pd.Timedelta(hours=hour),
                TURBINE_ID: "A",
                "operational_status": "normal",
                FAILURE_EVENT: int(hour in (100, 300)),
            }
        )
    frame = pd.DataFrame(rows)
    labelled = create_failure_target(frame, small_config)
    assert int(labelled[TARGET_COLUMN].sum()) == 96  # two disjoint 48-hour windows


def test_no_positives_without_failures(small_config):
    """A turbine with no failure events must have an all-zero target."""
    start = pd.Timestamp("2024-01-01")
    frame = pd.DataFrame(
        {
            TIMESTAMP: [start + pd.Timedelta(hours=h) for h in range(200)],
            TURBINE_ID: "Z",
            "operational_status": "normal",
            FAILURE_EVENT: 0,
        }
    )
    labelled = create_failure_target(frame, small_config)
    assert labelled[TARGET_COLUMN].sum() == 0


def test_step_inference(toy_frame):
    """Hourly data must be detected as a one-hour sampling interval."""
    assert infer_step_hours(toy_frame) == pytest.approx(1.0)


def test_target_is_binary(labelled_frame):
    """The target must only ever contain 0/1."""
    assert set(labelled_frame[TARGET_COLUMN].unique()) <= {0, 1}
    assert labelled_frame[TARGET_COLUMN].isna().sum() == 0
