"""Tests for model construction, training and selection."""

from __future__ import annotations

import numpy as np
import pytest

from wind_turbine_pm.config import Config
from wind_turbine_pm.models.baselines import build_candidates, compute_sample_weights
from wind_turbine_pm.models.evaluation import build_comparison_table, classification_metrics
from wind_turbine_pm.models.training import (
    LOWER_IS_BETTER,
    TrainedCandidate,
    select_best_candidate,
)


def test_candidates_are_pipelines(small_config):
    """Every candidate must be a pipeline so preprocessing travels with it."""
    from sklearn.pipeline import Pipeline

    for candidate in build_candidates(small_config):
        assert isinstance(candidate.estimator, Pipeline)


def test_disabled_candidates_are_skipped(small_config):
    """A candidate marked disabled must not be built."""
    data = small_config.to_dict()
    data["training"]["candidates"]["random_forest"]["enabled"] = False
    names = {c.name for c in build_candidates(Config(data))}
    assert "random_forest" not in names


def test_unknown_candidate_is_rejected(small_config):
    """An unrecognised candidate name must fail loudly."""
    data = small_config.to_dict()
    data["training"]["candidates"]["quantum_forest"] = {"enabled": True, "params": {}}
    with pytest.raises(ValueError, match="Unknown model candidate"):
        build_candidates(Config(data))


def test_no_enabled_candidates_is_rejected(small_config):
    """Disabling everything must raise rather than train nothing."""
    data = small_config.to_dict()
    for name in data["training"]["candidates"]:
        data["training"]["candidates"][name]["enabled"] = False
    with pytest.raises(ValueError, match="No model candidates"):
        build_candidates(Config(data))


def test_sample_weights_upweight_the_positive_class():
    """'auto' weighting must use the negative/positive ratio."""
    y = np.array([0] * 90 + [1] * 10)
    weights = compute_sample_weights(y, "auto")
    assert weights[y == 1][0] == pytest.approx(9.0)
    assert weights[y == 0][0] == pytest.approx(1.0)


def test_sample_weights_none_returns_none():
    """No weighting configured means no weights."""
    assert compute_sample_weights(np.array([0, 1]), None) is None


def test_sample_weights_reject_bad_string():
    """An unrecognised weighting string must raise."""
    with pytest.raises(ValueError):
        compute_sample_weights(np.array([0, 1]), "magic")


def test_training_completes_and_produces_probabilities(trained_bundle):
    """The fitted model must produce valid probabilities."""
    estimator, feature_names, threshold_result, splits = trained_bundle
    probabilities = estimator.predict_proba(splits["test"].x)[:, 1]
    assert probabilities.min() >= 0.0
    assert probabilities.max() <= 1.0
    assert len(probabilities) == len(splits["test"].x)


def test_predictions_are_deterministic(trained_bundle):
    """Scoring the same rows twice must give identical results."""
    estimator, _, _, splits = trained_bundle
    first = estimator.predict_proba(splits["test"].x)[:, 1]
    second = estimator.predict_proba(splits["test"].x)[:, 1]
    np.testing.assert_array_equal(first, second)


def test_model_beats_the_dummy_baseline(trained_bundle):
    """A real model must rank better than chance on the validation split."""
    estimator, _, threshold_result, splits = trained_bundle
    probabilities = estimator.predict_proba(splits["valid"].x)[:, 1]
    metrics = classification_metrics(splits["valid"].y, probabilities, 0.5)
    assert metrics["roc_auc"] > 0.6


def test_feature_metadata_matches_the_model(trained_bundle):
    """The recorded feature list must match what the estimator was fitted on."""
    estimator, feature_names, _, splits = trained_bundle
    assert list(splits["train"].x.columns) == feature_names
    assert estimator.n_features_in_ == len(feature_names)


def _candidate(name: str, **metrics) -> TrainedCandidate:
    base = {
        "pr_auc": 0.5,
        "recall": 0.7,
        "f2": 0.6,
        "precision": 0.4,
        "cost_per_sample": 0.2,
        "false_negative_rate": 0.3,
    }
    base.update(metrics)
    return TrainedCandidate(
        name=name,
        algorithm="Fake",
        estimator=None,
        train_seconds=1.0,
        valid_probabilities=np.array([0.1]),
        metrics={"train": base, "valid": base},
        latency={"single_row_ms": 1.0},
    )


def test_selection_prefers_lower_cost(small_config):
    """With a lower-is-better primary metric, the cheaper model must win."""
    candidates = [
        _candidate("dummy", pr_auc=0.02, cost_per_sample=0.9),
        _candidate("expensive", cost_per_sample=0.30, pr_auc=0.55),
        _candidate("cheap", cost_per_sample=0.10, pr_auc=0.50),
    ]
    winner, rationale = select_best_candidate(candidates, small_config)
    assert winner.name == "cheap"
    assert "cost_per_sample" in rationale


def test_selection_enforces_the_recall_floor(small_config):
    """A candidate below the recall floor must be rejected however cheap."""
    candidates = [
        _candidate("low_recall", recall=0.05, cost_per_sample=0.01),
        _candidate("good", recall=0.8, cost_per_sample=0.20),
    ]
    winner, _ = select_best_candidate(candidates, small_config)
    assert winner.name == "good"
    assert candidates[0].rejected_reason


def test_selection_enforces_the_pr_auc_guard_rail(small_config):
    """A cheap model that ranks poorly must be rejected."""
    candidates = [
        _candidate("cheap_but_poor_ranking", cost_per_sample=0.01, pr_auc=0.10),
        _candidate("solid", cost_per_sample=0.20, pr_auc=0.60),
    ]
    winner, _ = select_best_candidate(candidates, small_config)
    assert winner.name == "solid"


def test_dummy_is_never_selected(small_config):
    """The reference baseline must never be published as the final model."""
    candidates = [
        _candidate("dummy", cost_per_sample=0.0001, pr_auc=0.99, recall=1.0),
        _candidate("real", cost_per_sample=0.2, pr_auc=0.6),
    ]
    winner, _ = select_best_candidate(candidates, small_config)
    assert winner.name == "real"


def test_selection_fails_when_everything_is_rejected(small_config):
    """If no candidate qualifies, selection must raise rather than guess."""
    candidates = [_candidate("bad", recall=0.0, pr_auc=0.5)]
    with pytest.raises(ValueError, match="rejected"):
        select_best_candidate(candidates, small_config)


def test_lower_is_better_set_contains_cost():
    """Cost-like metrics must be recognised as lower-is-better."""
    assert "cost_per_sample" in LOWER_IS_BETTER
    assert "false_negative_rate" in LOWER_IS_BETTER
    assert "pr_auc" not in LOWER_IS_BETTER


def test_metrics_include_the_required_reporting_set():
    """All metrics the project promises to report must be produced."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.6, 0.3, 0.4])
    metrics = classification_metrics(y_true, y_prob, 0.5)
    for key in (
        "precision",
        "recall",
        "f1",
        "f2",
        "pr_auc",
        "roc_auc",
        "false_negative_rate",
        "false_positive_rate",
        "positive_prediction_rate",
        "true_positives",
        "false_negatives",
    ):
        assert key in metrics


def test_metrics_reject_mismatched_shapes():
    """Mismatched inputs must raise rather than silently broadcast."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        classification_metrics(np.array([0, 1]), np.array([0.5]), 0.5)


def test_comparison_table_ranks_by_validation_pr_auc():
    """The comparison table must be ordered by validation PR-AUC."""
    results = {
        "a": {"valid": {"pr_auc": 0.2, "recall": 0.5}},
        "b": {"valid": {"pr_auc": 0.8, "recall": 0.5}},
    }
    table = build_comparison_table(results)
    assert table.iloc[0]["model"] == "b"


def test_near_ties_are_broken_by_the_tie_breakers(small_config):
    """Sub-percent differences in the primary metric must not decide the winner."""
    candidates = [
        # 0.7% cheaper, but materially worse at catching failures.
        _candidate(
            "marginally_cheaper", cost_per_sample=0.1396, f2=0.607, recall=0.687, pr_auc=0.56
        ),
        _candidate("better_recall", cost_per_sample=0.1409, f2=0.638, recall=0.814, pr_auc=0.52),
    ]
    winner, rationale = select_best_candidate(candidates, small_config)
    assert winner.name == "better_recall"
    assert "tie-breakers" in rationale


def test_clear_primary_metric_wins_outside_the_band(small_config):
    """A difference well beyond the tolerance must still be decisive."""
    candidates = [
        _candidate("much_cheaper", cost_per_sample=0.05, f2=0.50, recall=0.60, pr_auc=0.55),
        _candidate("expensive", cost_per_sample=0.30, f2=0.90, recall=0.95, pr_auc=0.55),
    ]
    winner, _ = select_best_candidate(candidates, small_config)
    assert winner.name == "much_cheaper"
