"""Leakage and feature-contract tests for anomaly detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wind_turbine_pm.anomaly.features import build_anomaly_features
from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import TIMESTAMP, TURBINE_ID
from wind_turbine_pm.data.preprocessing import preprocess
from wind_turbine_pm.features.transformers import assert_no_future_leakage
from wind_turbine_pm.health.regimes import attach_regimes


def _prepared(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    clean, _ = preprocess(frame, cfg)
    return attach_regimes(clean, cfg)


def test_future_row_perturbation_does_not_change_past_features(
    small_anomaly_config: Config, anomaly_toy_frame: pd.DataFrame
) -> None:
    prepared = _prepared(anomaly_toy_frame, small_anomaly_config)

    def build(frame: pd.DataFrame) -> pd.DataFrame:
        return build_anomaly_features(frame, small_anomaly_config)[0]

    assert_no_future_leakage(prepared, build, TURBINE_ID, TIMESTAMP)


def test_truth_and_event_columns_never_enter_feature_matrix(
    small_anomaly_config: Config, anomaly_toy_frame: pd.DataFrame
) -> None:
    features, _ = build_anomaly_features(
        _prepared(anomaly_toy_frame, small_anomaly_config),
        small_anomaly_config,
    )
    forbidden = set(small_anomaly_config.require("anomaly.features.exclude_columns"))
    assert not (forbidden & set(features.columns))
    assert not {
        "degradation_level",
        "failure_event",
        "maintenance_event",
        "failure_mode",
        "episode_id",
    }.intersection(features)


def test_feature_matrix_is_numeric_aligned_and_deterministic(
    small_anomaly_config: Config, anomaly_toy_frame: pd.DataFrame
) -> None:
    prepared = _prepared(anomaly_toy_frame, small_anomaly_config)
    first, spec = build_anomaly_features(prepared, small_anomaly_config)
    second, _ = build_anomaly_features(prepared, small_anomaly_config)
    pd.testing.assert_frame_equal(first, second)
    assert first.index.equals(prepared.index)
    assert all(np.issubdtype(dtype, np.number) for dtype in first.dtypes)
    assert list(first.columns) == list(spec.names)
    assert set(spec.groups) == {
        "raw_and_regime",
        "rolling",
        "trend",
        "physical",
        "regime_relative",
    }


def test_features_never_cross_turbine_boundaries(
    small_anomaly_config: Config, anomaly_toy_frame: pd.DataFrame
) -> None:
    prepared = _prepared(anomaly_toy_frame, small_anomaly_config)
    baseline, _ = build_anomaly_features(prepared, small_anomaly_config)
    mutated = prepared.copy()
    rows_b = mutated[TURBINE_ID] == "B"
    numeric = mutated.select_dtypes(include=[np.number]).columns
    mutated.loc[rows_b, numeric] += 10.0
    changed, _ = build_anomaly_features(mutated, small_anomaly_config)
    rows_a = prepared.index[prepared[TURBINE_ID] == "A"]
    pd.testing.assert_frame_equal(
        baseline.loc[rows_a],
        changed.loc[rows_a],
        check_exact=False,
        atol=1e-6,
        rtol=0.0,
    )
