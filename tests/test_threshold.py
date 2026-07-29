"""Tests for threshold optimisation and risk banding."""

from __future__ import annotations

import numpy as np
import pytest

from wind_turbine_pm.config import ConfigError, validate_config
from wind_turbine_pm.models.threshold import (
    CostMatrix,
    map_risk_level,
    optimise_threshold,
    threshold_curve,
    threshold_grid,
)


@pytest.fixture
def scores():
    """A separable-but-noisy score set with a 10% positive rate."""
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(900, dtype=int), np.ones(100, dtype=int)])
    probabilities = np.concatenate([rng.beta(2, 8, 900), rng.beta(6, 3, 100)])
    return y, np.clip(probabilities, 0.0, 1.0)


def test_threshold_is_within_bounds(scores, small_config):
    """A selected threshold must always be a valid probability."""
    y, p = scores
    result = optimise_threshold(y, p, small_config)
    assert 0.0 <= result.threshold <= 1.0


def test_threshold_is_not_the_naive_half(scores, small_config):
    """Optimisation must actually move away from 0.50 on imbalanced data."""
    y, p = scores
    result = optimise_threshold(y, p, small_config)
    assert result.threshold != 0.5


def test_f2_method_maximises_f2(scores, small_config):
    """The F2 method must return the grid point with the highest F2."""
    y, p = scores
    result = optimise_threshold(y, p, small_config, method="f2")
    best = result.curve["f2"].max()
    assert result.metrics["f2"] == pytest.approx(best, abs=1e-9)


def test_cost_method_minimises_cost(scores, small_config):
    """The cost method must return the grid point with the lowest total cost."""
    y, p = scores
    result = optimise_threshold(y, p, small_config, method="cost")
    assert result.metrics["total_cost"] == pytest.approx(result.curve["total_cost"].min(), abs=1e-9)


def test_cost_method_favours_recall_over_f2(scores, small_config):
    """A 10:2 FN:FP cost ratio must produce a threshold no higher than F2's."""
    y, p = scores
    cost_threshold = optimise_threshold(y, p, small_config, method="cost").threshold
    f2_threshold = optimise_threshold(y, p, small_config, method="f2").threshold
    assert cost_threshold <= f2_threshold + 1e-9


def test_both_methods_are_reported(scores, small_config):
    """Whichever method is used, the alternative must also be recorded."""
    y, p = scores
    result = optimise_threshold(y, p, small_config, method="cost")
    assert set(result.alternatives) >= {"f2", "cost", "default"}


def test_threshold_selection_refuses_the_test_split(scores, small_config):
    """Passing the test split must raise: it may never influence selection."""
    y, p = scores
    with pytest.raises(ValueError, match="test split"):
        optimise_threshold(y, p, small_config, split_name="test")


def test_result_records_the_validation_split(scores, small_config):
    """The artifact must record that validation data was used."""
    y, p = scores
    result = optimise_threshold(y, p, small_config)
    assert result.selected_on == "valid"


def test_unknown_method_is_rejected(scores, small_config):
    """An unrecognised method must raise."""
    y, p = scores
    with pytest.raises(ValueError, match="Unknown threshold method"):
        optimise_threshold(y, p, small_config, method="vibes")


def test_cost_matrix_total():
    """The cost matrix must sum each outcome times its count."""
    costs = CostMatrix(false_negative=10, false_positive=2, true_positive=1, true_negative=0)
    assert costs.total(tn=100, fp=5, fn=3, tp=2) == pytest.approx(5 * 2 + 3 * 10 + 2 * 1)


def test_threshold_curve_covers_the_grid(scores, small_config):
    """The curve must contain one row per grid point."""
    y, p = scores
    grid = threshold_grid(small_config)
    curve = threshold_curve(y, p, grid, CostMatrix.from_config(small_config))
    assert len(curve) == len(grid)
    assert curve["recall"].iloc[0] >= curve["recall"].iloc[-1]


def test_optimisation_is_deterministic(scores, small_config):
    """The same inputs must always select the same threshold."""
    y, p = scores
    assert (
        optimise_threshold(y, p, small_config).threshold
        == optimise_threshold(y, p, small_config).threshold
    )


# ---------------------------------------------------------------------------
# Risk banding
# ---------------------------------------------------------------------------
def test_risk_mapping_boundaries():
    """Bands must be half-open: [0, low), [low, medium), [medium, 1]."""
    assert map_risk_level(0.0, 0.3, 0.65) == "low"
    assert map_risk_level(0.299, 0.3, 0.65) == "low"
    assert map_risk_level(0.3, 0.3, 0.65) == "medium"
    assert map_risk_level(0.649, 0.3, 0.65) == "medium"
    assert map_risk_level(0.65, 0.3, 0.65) == "high"
    assert map_risk_level(1.0, 0.3, 0.65) == "high"


def test_risk_mapping_rejects_inconsistent_bands():
    """Out-of-order band limits must raise."""
    with pytest.raises(ValueError, match="Risk bands"):
        map_risk_level(0.5, 0.7, 0.3)


def test_risk_bands_are_independent_of_the_decision_threshold(small_config):
    """Risk banding must not be tied to the binary threshold."""
    low_max = float(small_config.require("risk_levels.low_max"))
    medium_max = float(small_config.require("risk_levels.medium_max"))
    assert 0.0 < low_max < medium_max < 1.0


def test_config_validation_rejects_bad_risk_bands():
    """Configuration validation must catch inverted risk bands."""
    with pytest.raises(ConfigError, match="risk_levels"):
        validate_config({"risk_levels": {"low_max": 0.8, "medium_max": 0.2}})


def test_config_validation_rejects_bad_split_fractions():
    """Configuration validation must catch inverted split fractions."""
    with pytest.raises(ConfigError, match="split fractions"):
        validate_config({"split": {"train_end_fraction": 0.9, "valid_end_fraction": 0.5}})


def test_config_validation_rejects_bad_threshold_method():
    """Configuration validation must catch an unknown threshold method."""
    with pytest.raises(ConfigError, match="threshold.method"):
        validate_config({"threshold": {"method": "vibes"}})
