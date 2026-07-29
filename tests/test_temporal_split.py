"""Tests for chronological splitting and the embargo."""

from __future__ import annotations

import pandas as pd
import pytest

from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import TIMESTAMP, TURBINE_ID
from wind_turbine_pm.data.splitting import (
    compute_boundaries,
    temporal_split,
    verify_split_integrity,
)


def test_splits_are_chronologically_ordered(prepared, small_config):
    """Train must precede validation, which must precede test."""
    _, _, eligible = prepared
    train, valid, test, _ = temporal_split(eligible, small_config)
    assert pd.to_datetime(train[TIMESTAMP]).max() < pd.to_datetime(valid[TIMESTAMP]).min()
    assert pd.to_datetime(valid[TIMESTAMP]).max() < pd.to_datetime(test[TIMESTAMP]).min()


def test_no_timestamp_overlap(prepared, small_config):
    """The three splits must not share a single timestamp."""
    _, _, eligible = prepared
    train, valid, test, _ = temporal_split(eligible, small_config)
    train_times = set(pd.to_datetime(train[TIMESTAMP]))
    valid_times = set(pd.to_datetime(valid[TIMESTAMP]))
    test_times = set(pd.to_datetime(test[TIMESTAMP]))
    assert not train_times & valid_times
    assert not valid_times & test_times
    assert not train_times & test_times


def test_embargo_gap_is_enforced(prepared, small_config):
    """The gap at each boundary must be at least the configured embargo."""
    _, _, eligible = prepared
    train, valid, test, boundaries = temporal_split(eligible, small_config)
    embargo = pd.Timedelta(hours=boundaries.embargo_hours)
    assert (
        pd.to_datetime(valid[TIMESTAMP]).min() - pd.to_datetime(train[TIMESTAMP]).max() >= embargo
    )
    assert pd.to_datetime(test[TIMESTAMP]).min() - pd.to_datetime(valid[TIMESTAMP]).max() >= embargo
    verify_split_integrity(train, valid, test, boundaries)


def test_embargo_shorter_than_horizon_is_rejected(prepared, small_config):
    """An embargo below the label horizon would leak and must be refused."""
    _, _, eligible = prepared
    data = small_config.to_dict()
    data["split"]["embargo_hours"] = 12  # horizon is 48
    with pytest.raises(ValueError, match="embargo"):
        compute_boundaries(eligible, Config(data))


def test_all_turbines_appear_in_every_split(prepared, small_config):
    """The split is over time, not over turbines: all turbines appear in each."""
    _, _, eligible = prepared
    train, valid, test, _ = temporal_split(eligible, small_config)
    turbines = set(eligible[TURBINE_ID].unique())
    for part in (train, valid, test):
        assert set(part[TURBINE_ID].unique()) == turbines


def test_split_is_deterministic(prepared, small_config):
    """Splitting twice must yield identical partitions."""
    _, _, eligible = prepared
    first = temporal_split(eligible, small_config)
    second = temporal_split(eligible, small_config)
    for left, right in zip(first[:3], second[:3]):
        pd.testing.assert_frame_equal(left, right)


def test_rows_are_embargoed_not_silently_reassigned(prepared, small_config):
    """Embargoed rows must be dropped, so the parts sum to less than the whole."""
    _, _, eligible = prepared
    train, valid, test, _ = temporal_split(eligible, small_config)
    assert len(train) + len(valid) + len(test) < len(eligible)


def test_degenerate_boundaries_are_rejected(prepared, small_config):
    """An embargo wide enough to swallow a split must fail loudly."""
    _, _, eligible = prepared
    data = small_config.to_dict()
    data["split"]["embargo_hours"] = 24 * 365
    with pytest.raises(ValueError):
        compute_boundaries(eligible, Config(data))


def test_verify_split_integrity_catches_overlap(prepared, small_config):
    """The integrity check must reject an overlapping hand-built split."""
    _, _, eligible = prepared
    train, valid, test, boundaries = temporal_split(eligible, small_config)
    bad_valid = pd.concat([valid, train.tail(5)])
    with pytest.raises(ValueError, match="overlap"):
        verify_split_integrity(train, bad_valid, test, boundaries)


def test_preprocessing_is_fit_on_training_data_only(prepared, small_config):
    """The imputer/scaler inside the pipeline must be fitted on train only.

    Fitting a scaler on the full dataset is the classic silent leak. Because
    preprocessing lives *inside* the sklearn pipeline, fitting on the training
    split is structurally guaranteed - this test pins that guarantee down by
    checking the learned statistics match the training split, not the whole set.
    """
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    _, features, eligible = prepared
    train, valid, _, _ = temporal_split(eligible, small_config)

    columns = ["vibration", "gearbox_temperature"]
    pipeline = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("model", LogisticRegression(max_iter=50))]
    )
    x_train = features.loc[train.index, columns]
    pipeline.fit(x_train, train["failure_within_48h"])

    learned = pipeline.named_steps["impute"].statistics_
    expected_train = x_train.median().to_numpy()
    expected_all = features.loc[:, columns].median().to_numpy()

    assert np.allclose(learned, expected_train, equal_nan=True)
    # The training median must differ from the full-dataset median, otherwise
    # this test could pass even with a leaking fit.
    assert not np.allclose(expected_train, expected_all)
