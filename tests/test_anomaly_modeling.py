"""Candidate, calibration and selection tests for anomaly modeling."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from wind_turbine_pm.anomaly.modeling import (
    anomaly_metrics,
    build_candidates,
    deterministic_reference_sample,
    fit_calibration,
    percentile_scores,
    raw_novelty_score,
    select_best,
    train_candidates,
)
from wind_turbine_pm.config import Config


def _matrix(seed: int = 4) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train = pd.DataFrame(rng.normal(0, 1, (180, 5)))
    healthy = rng.normal(0, 1, (120, 5))
    anomalies = rng.normal(5, 0.8, (40, 5))
    valid = pd.DataFrame(np.vstack([healthy, anomalies]))
    truth = np.r_[np.zeros(len(healthy), dtype=int), np.ones(len(anomalies), dtype=int)]
    valid_healthy = truth == 0
    return train, valid, truth, valid_healthy


def test_all_three_algorithms_have_high_is_anomalous_orientation(
    small_anomaly_config: Config,
) -> None:
    train, valid, _, _ = _matrix()
    normal_probe = valid.iloc[:20]
    anomaly_probe = valid.iloc[-20:]
    candidates = build_candidates(small_anomaly_config)
    assert set(candidates) == {
        "isolation_forest",
        "local_outlier_factor",
        "one_class_svm",
    }
    for estimator in candidates.values():
        estimator.fit(train)
        assert (
            raw_novelty_score(estimator, anomaly_probe).mean()
            > raw_novelty_score(estimator, normal_probe).mean()
        )


def test_empirical_calibration_measures_five_and_one_percent_alert_rates() -> None:
    healthy = np.linspace(-3.0, 3.0, 1000)
    calibration = fit_calibration(healthy, warning_rate=0.05, alarm_rate=0.01)
    assert calibration.achieved_warning_rate == pytest.approx(0.05, abs=0.002)
    assert calibration.achieved_alarm_rate == pytest.approx(0.01, abs=0.002)
    mapped = percentile_scores(np.array([-10.0, 0.0, 10.0]), calibration)
    assert np.all(np.diff(mapped) >= 0.0)
    assert mapped[0] == 0.0
    assert mapped[-1] == 1.0


def test_training_is_deterministic_and_selection_uses_validation(
    small_anomaly_config: Config,
) -> None:
    train, valid, truth, healthy = _matrix()
    first = train_candidates(train, valid, truth, healthy, small_anomaly_config)
    second = train_candidates(train, valid, truth, healthy, small_anomaly_config)
    assert [item.name for item in first] == [item.name for item in second]
    for left, right in zip(first, second):
        np.testing.assert_allclose(left.valid_scores, right.valid_scores)
    winner, rationale = select_best(first)
    assert winner.metrics["recall"] == max(item.metrics["recall"] for item in first)
    assert "Test data did not participate in selection" in rationale


def test_selection_tie_breaks_by_pr_auc_f2_then_latency(
    small_anomaly_config: Config,
) -> None:
    train, valid, truth, healthy = _matrix()
    base = train_candidates(train, valid, truth, healthy, small_anomaly_config)[0]
    stronger = replace(
        base,
        name="stronger",
        metrics={**base.metrics, "recall": 0.8, "pr_auc": 0.9, "f2": 0.7},
        latency_ms=10.0,
    )
    weaker = replace(
        base,
        name="weaker",
        metrics={**base.metrics, "recall": 0.8, "pr_auc": 0.8, "f2": 0.9},
        latency_ms=1.0,
    )
    assert select_best([weaker, stronger])[0].name == "stronger"


def test_reference_sampling_is_bounded_balanced_and_repeatable() -> None:
    features = pd.DataFrame({"x": np.arange(200), "y": np.arange(200) * 2})
    keys = pd.DataFrame(
        {
            "turbine_id": ["A"] * 100 + ["B"] * 100,
            "operating_regime": ["low_load"] * 50
            + ["high_load"] * 50
            + ["low_load"] * 50
            + ["high_load"] * 50,
        }
    )
    first = deterministic_reference_sample(features, keys, maximum=40, seed=7)
    second = deterministic_reference_sample(features, keys, maximum=40, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 40
    assert len(set(first.index).intersection(range(0, 50))) > 0
    assert len(set(first.index).intersection(range(150, 200))) > 0


def test_metrics_use_warning_threshold_and_report_healthy_rate() -> None:
    calibration = fit_calibration(np.linspace(0, 1, 100), 0.05, 0.01)
    truth = np.array([0, 0, 1, 1])
    scores = np.array([0.2, 0.96, 0.97, 1.0])
    metrics = anomaly_metrics(truth, scores, calibration)
    assert metrics["recall"] == 1.0
    assert metrics["healthy_alert_rate"] == 0.5
