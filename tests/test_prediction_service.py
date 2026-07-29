"""Tests for the shared FailurePredictionService.

Where a test needs published artifacts it uses the real ones written by the
pipeline and is skipped when they are absent, so a fresh checkout still runs a
green suite.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from wind_turbine_pm.config import Config, load_config
from wind_turbine_pm.constants import RiskLevel
from wind_turbine_pm.contracts.observations import TurbineObservation, TurbineWindow
from wind_turbine_pm.contracts.predictions import FailurePrediction
from wind_turbine_pm.explainability.narratives import (
    build_advisory,
    build_explanation,
    build_recommendation,
    humanise_feature,
)
from wind_turbine_pm.explainability.shap_explainer import LocalAttribution
from wind_turbine_pm.models.persistence import bundle_available
from wind_turbine_pm.models.prediction import FeatureContractError, apply_threshold
from wind_turbine_pm.services.failure_prediction_service import (
    FailurePredictionService,
    InsufficientHistoryError,
    ServiceNotReadyError,
)

pytestmark_needs_artifacts = pytest.mark.skipif(
    not bundle_available(load_config()),
    reason="Model artifacts not built; run python scripts/run_failure_pipeline.py",
)


@pytest.fixture(scope="module")
def service() -> FailurePredictionService:
    """The real service backed by published artifacts."""
    return FailurePredictionService()


def _observation(turbine: str, timestamp: datetime, **overrides) -> TurbineObservation:
    payload = {
        "turbine_id": turbine,
        "timestamp": timestamp,
        "wind_speed": 8.4,
        "rotor_speed": 12.1,
        "generator_speed": 1174.0,
        "power_output": 720.0,
        "generator_temperature": 62.0,
        "gearbox_temperature": 55.0,
        "bearing_temperature": 48.0,
        "oil_temperature": 46.0,
        "oil_pressure": 5.1,
        "vibration": 3.2,
        "ambient_temperature": 11.0,
        "nacelle_temperature": 19.0,
        "hydraulic_pressure": 188.0,
        "brake_temperature": 17.0,
        "operational_status": "normal",
    }
    payload.update(overrides)
    return TurbineObservation(**payload)


def _window(turbine: str = "T01", hours: int = 96) -> TurbineWindow:
    end = datetime(2025, 6, 1, tzinfo=UTC)
    return TurbineWindow(
        turbine_id=turbine,
        observations=[
            _observation(turbine, end - timedelta(hours=hours - 1 - i)) for i in range(hours)
        ],
    )


# ---------------------------------------------------------------------------
# Missing artifacts
# ---------------------------------------------------------------------------
def test_missing_artifacts_raise_a_clear_error(tmp_path, small_config):
    """A service pointed at an empty artifact directory must fail helpfully."""
    data = small_config.to_dict()
    data["paths"]["artifacts_models"] = str(tmp_path / "models")
    data["paths"]["artifacts_metadata"] = str(tmp_path / "metadata")
    broken = FailurePredictionService(Config(data))

    assert not broken.is_ready
    status = broken.status()
    assert status.model_loaded is False
    assert "run_failure_pipeline" in status.detail

    with pytest.raises(ServiceNotReadyError) as exc:
        broken.predict_from_window(_window())
    assert "run_failure_pipeline" in str(exc.value)


def test_status_never_raises_when_artifacts_are_missing(tmp_path, small_config):
    """``status()`` must always return, so /health can report degraded."""
    data = small_config.to_dict()
    data["paths"]["artifacts_models"] = str(tmp_path / "nope")
    data["paths"]["artifacts_metadata"] = str(tmp_path / "nope")
    status = FailurePredictionService(Config(data)).status()
    assert status.model_loaded is False


# ---------------------------------------------------------------------------
# Real predictions
# ---------------------------------------------------------------------------
@pytestmark_needs_artifacts
def test_single_prediction_from_window(service):
    """A raw-history window must produce a complete prediction."""
    result = service.predict_from_window(_window())
    assert isinstance(result, FailurePrediction)
    assert 0.0 <= result.failure_probability <= 1.0
    assert result.prediction in (0, 1)
    assert result.risk_level in set(RiskLevel)
    assert result.horizon_hours == 48
    assert result.advisory_only is True
    assert result.model_version


@pytestmark_needs_artifacts
def test_prediction_is_deterministic(service):
    """The same window must always produce the same probability."""
    window = _window()
    assert (
        service.predict_from_window(window).failure_probability
        == service.predict_from_window(window).failure_probability
    )


@pytestmark_needs_artifacts
def test_short_window_is_rejected(service):
    """A window below the configured minimum history must raise."""
    with pytest.raises(InsufficientHistoryError, match="history"):
        service.predict_from_window(_window(hours=4))


@pytestmark_needs_artifacts
def test_batch_prediction(service):
    """A batch must return one prediction per window, in order."""
    windows = [_window("T01"), _window("T02"), _window("T03")]
    result = service.predict_batch_from_windows(windows)
    assert result.count == 3
    assert [p.turbine_id for p in result.predictions] == ["T01", "T02", "T03"]
    assert result.advisory_only is True


@pytestmark_needs_artifacts
def test_batch_limit_is_enforced(service):
    """A batch beyond the configured limit must be refused."""
    data = service.config.to_dict()
    data["serving"]["max_batch_turbines"] = 2
    limited = FailurePredictionService(Config(data))
    with pytest.raises(ValueError, match="exceeds"):
        limited.predict_batch_from_windows([_window("T01"), _window("T02"), _window("T03")])


@pytestmark_needs_artifacts
def test_prediction_from_prepared_features(service):
    """The fallback prepared-feature path must work."""
    payload = service.example_feature_payload()
    result = service.predict_from_features(payload, "T07", datetime(2025, 1, 1, tzinfo=UTC))
    assert 0.0 <= result.failure_probability <= 1.0
    assert result.turbine_id == "T07"


@pytestmark_needs_artifacts
def test_prepared_features_reject_a_missing_feature(service):
    """An incomplete feature vector must be refused with a clear message."""
    payload = service.example_feature_payload()
    payload.pop(next(iter(payload)))
    with pytest.raises(FeatureContractError, match="Missing"):
        service.predict_from_features(payload, "T01", datetime(2025, 1, 1, tzinfo=UTC))


@pytestmark_needs_artifacts
def test_explanation_output_schema(service):
    """Risk factors must carry a feature, signed impact and a direction."""
    result = service.predict_from_window(_window())
    for factor in result.top_risk_factors:
        assert factor.feature in service.feature_names
        assert isinstance(factor.impact, float)
        assert factor.direction in {"increases_risk", "decreases_risk"}
        assert factor.description


@pytestmark_needs_artifacts
def test_recommendation_is_advisory(service):
    """Every recommendation must carry the human-review disclaimer."""
    result = service.predict_from_window(_window())
    assert "qualified maintenance engineer" in result.recommendation
    assert result.explanation


@pytestmark_needs_artifacts
def test_risk_level_matches_probability(service):
    """The reported band must match the configured cut-points."""
    from wind_turbine_pm.models.threshold import map_risk_level

    low_max, medium_max = service.risk_bands
    result = service.predict_from_window(_window())
    assert str(result.risk_level) == map_risk_level(result.failure_probability, low_max, medium_max)


@pytestmark_needs_artifacts
def test_prediction_matches_the_threshold(service):
    """The binary flag must be the probability compared to the threshold."""
    result = service.predict_from_window(_window())
    assert result.prediction == int(result.failure_probability >= result.threshold)


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------
def test_window_rejects_mixed_turbines():
    """A window must contain one turbine only."""
    now = datetime(2025, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="turbine_id"):
        TurbineWindow(
            turbine_id="T01",
            observations=[_observation("T01", now), _observation("T02", now + timedelta(hours=1))],
        )


def test_window_rejects_unsorted_observations():
    """Observations must be chronologically ordered."""
    now = datetime(2025, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="sorted"):
        TurbineWindow(
            turbine_id="T01",
            observations=[_observation("T01", now + timedelta(hours=1)), _observation("T01", now)],
        )


def test_window_rejects_duplicate_timestamps():
    """Duplicate timestamps in a window must be refused."""
    now = datetime(2025, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="duplicate"):
        TurbineWindow(
            turbine_id="T01", observations=[_observation("T01", now), _observation("T01", now)]
        )


def test_observation_rejects_impossible_values():
    """Physically impossible input must be rejected at the contract boundary."""
    with pytest.raises(ValueError):
        _observation("T01", datetime(2025, 1, 1, tzinfo=UTC), vibration=9999.0)


def test_window_to_frame_is_sorted():
    """Conversion to a dataframe must preserve chronological order."""
    frame = _window(hours=10).to_frame()
    assert frame["timestamp"].is_monotonic_increasing
    assert len(frame) == 10


def test_apply_threshold_bounds():
    """An out-of-range threshold must be refused."""
    import numpy as np

    with pytest.raises(ValueError):
        apply_threshold(np.array([0.5]), 1.5)


# ---------------------------------------------------------------------------
# Narratives
# ---------------------------------------------------------------------------
def test_humanise_feature_decodes_structure():
    """Feature names must be decoded into readable phrases."""
    assert "12-hour average" in humanise_feature("vibration_roll_mean_12h")
    assert "vibration" in humanise_feature("vibration_roll_mean_12h")
    assert "6-hour trend" in humanise_feature("gearbox_temperature_slope_6h")
    assert "power curve" in humanise_feature("power_curve_residual")


def test_explanation_reports_when_no_attribution_is_available():
    """With no attributions the narrative must say so, not invent drivers."""
    text = build_explanation([], 0.42, RiskLevel.MEDIUM)
    assert "No feature-level attribution" in text
    assert "42" in text


def test_explanation_names_actual_drivers():
    """The narrative must mention the features that were actually attributed."""
    attributions = [
        LocalAttribution("vibration_roll_mean_12h", 0.8, 4.5),
        LocalAttribution("oil_pressure_roll_std_24h", -0.3, 0.1),
    ]
    text = build_explanation(attributions, 0.77, RiskLevel.HIGH)
    assert "vibration" in text
    assert "oil pressure" in text
    assert "77" in text


def test_recommendations_escalate_with_risk():
    """Each risk band must produce a distinct, appropriately worded message."""
    low = build_recommendation(RiskLevel.LOW, [])
    medium = build_recommendation(RiskLevel.MEDIUM, [])
    high = build_recommendation(RiskLevel.HIGH, [])

    assert "routine monitoring" in low
    assert "non-urgent" in medium
    assert "Prioritise" in high
    for message in (low, medium, high):
        assert "qualified maintenance engineer" in message


def test_recommendations_avoid_automation_language():
    """Recommendations must never instruct automatic action on plant."""
    for level in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH):
        message = build_recommendation(level, []).lower()
        for forbidden in ("shut down", "stop the turbine", "automatically", "guaranteed"):
            assert forbidden not in message


def test_advisory_pairs_explanation_and_recommendation():
    """The advisory helper must return both parts."""
    advisory = build_advisory([LocalAttribution("vibration", 0.5, 3.0)], 0.8, RiskLevel.HIGH)
    assert advisory.explanation
    assert advisory.recommendation


# ---------------------------------------------------------------------------
# Runtime / artifact version compatibility
# ---------------------------------------------------------------------------
def _metadata_with_versions(versions: dict[str, str]):
    """Build a minimal ModelMetadata carrying the given library versions."""
    from datetime import datetime

    from wind_turbine_pm.contracts.metadata import (
        DatasetSummary,
        ModelMetadata,
        RiskLevelBands,
        ThresholdInfo,
    )

    return ModelMetadata(
        model_name="failure_prediction",
        model_version="1.0.0",
        algorithm="HistGradientBoostingClassifier",
        training_date=datetime(2026, 1, 1, tzinfo=UTC),
        target="failure_within_48h",
        horizon_hours=48,
        features=["a", "b"],
        threshold=ThresholdInfo(value=0.1, method="cost"),
        risk_levels=RiskLevelBands(low_max=0.3, medium_max=0.65),
        dataset=DatasetSummary(
            source="synthetic", is_synthetic=True, n_turbines=1, n_rows=1, n_features=2
        ),
        library_versions=versions,
    )


def test_matching_versions_produce_no_warning():
    """A runtime matching the artifact must report no mismatch."""
    import importlib.metadata as importlib_metadata

    from wind_turbine_pm.models.persistence import check_runtime_compatibility

    current = {p: importlib_metadata.version(p) for p in ("scikit-learn", "numpy", "joblib")}
    assert check_runtime_compatibility(_metadata_with_versions(current)) == []


def test_mismatched_sklearn_is_reported():
    """A different scikit-learn must be flagged - pickles are version-specific."""
    from wind_turbine_pm.models.persistence import check_runtime_compatibility

    warnings = check_runtime_compatibility(_metadata_with_versions({"scikit-learn": "0.0.1"}))
    assert len(warnings) == 1
    assert "scikit-learn" in warnings[0]
    assert "0.0.1" in warnings[0]


def test_missing_version_record_is_tolerated():
    """An artifact predating version recording must not raise."""
    from wind_turbine_pm.models.persistence import check_runtime_compatibility

    assert check_runtime_compatibility(_metadata_with_versions({})) == []


@pytestmark_needs_artifacts
def test_published_artifact_matches_this_runtime(service):
    """The committed artifact must be loadable by the current environment."""
    from wind_turbine_pm.models.persistence import check_runtime_compatibility

    assert check_runtime_compatibility(service.metadata) == []
    assert service.status().runtime_warnings == []
