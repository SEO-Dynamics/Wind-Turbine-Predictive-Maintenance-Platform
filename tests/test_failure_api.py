"""Tests for the FastAPI layer.

Endpoints that need a published model are skipped when artifacts are absent;
the artifact-missing behaviour itself is tested explicitly with an isolated
service.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from wind_turbine_pm.anomaly.config import load_anomaly_config
from wind_turbine_pm.anomaly.persistence import bundle_available as anomaly_bundle_available
from wind_turbine_pm.api.dependencies import provide_service
from wind_turbine_pm.api.main import app
from wind_turbine_pm.config import Config, load_config
from wind_turbine_pm.health.persistence import health_bundle_available
from wind_turbine_pm.models.persistence import bundle_available
from wind_turbine_pm.services.failure_prediction_service import FailurePredictionService

needs_artifacts = pytest.mark.skipif(
    not bundle_available(load_config()),
    reason="Model artifacts not built; run python scripts/run_failure_pipeline.py",
)
_stage3_config = load_anomaly_config()
needs_all_artifacts = pytest.mark.skipif(
    not (
        bundle_available(_stage3_config)
        and health_bundle_available(_stage3_config)
        and anomaly_bundle_available(_stage3_config)
    ),
    reason="All model artifacts are required; run python scripts/run_all_pipelines.py",
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    """A test client bound to the real application."""
    return TestClient(app)


@pytest.fixture
def broken_client(tmp_path) -> TestClient:
    """A client whose service points at an empty artifact directory."""
    cfg = load_config().to_dict()
    cfg["paths"]["artifacts_models"] = str(tmp_path / "models")
    cfg["paths"]["artifacts_metadata"] = str(tmp_path / "metadata")
    broken = FailurePredictionService(Config(cfg))

    # Only the *base* dependency is overridden. `provide_ready_service` composes
    # on top of it, so the real 503 readiness gate still runs - which is exactly
    # the behaviour under test.
    app.dependency_overrides[provide_service] = lambda: broken
    yield TestClient(app)
    app.dependency_overrides.clear()


class _ReadyStub:
    """Minimal stand-in that satisfies the readiness gate but never predicts.

    Request-body validation happens *after* FastAPI resolves dependencies, so
    without a model the readiness gate short-circuits every request with 503 and
    schema-rejection tests can never reach their assertion. This stub keeps those
    tests meaningful in any environment, including CI before the pipeline runs.
    """

    is_ready = True
    metadata = object()


@pytest.fixture
def schema_client() -> TestClient:
    """A client whose service is 'ready', for testing request-schema rejection."""
    app.dependency_overrides[provide_service] = _ReadyStub
    yield TestClient(app)
    app.dependency_overrides.clear()


def _window_payload(turbine: str = "T01", hours: int = 96) -> dict:
    end = datetime(2025, 6, 1, tzinfo=UTC)
    return {
        "turbine_id": turbine,
        "observations": [
            {
                "turbine_id": turbine,
                "timestamp": (end - timedelta(hours=hours - 1 - i)).isoformat(),
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
            for i in range(hours)
        ],
    }


# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------
def test_app_imports_and_has_a_title():
    """The documented import test must succeed."""
    assert app.title == "Wind Turbine Predictive Maintenance API"


def test_openapi_schema_is_generated(client):
    """Swagger documentation must be available."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    for endpoint in (
        "/health",
        "/failure/model-info",
        "/failure/predict",
        "/failure/predict/batch",
    ):
        assert endpoint in paths


def test_root_lists_all_modules_without_stale_roadmap_entries(client):
    """The root endpoint must advertise all four mounted modules."""
    body = client.get("/").json()
    assert body["modules"][0]["module"] == "failure_prediction"
    mounted = {m["module"] for m in body["modules"]}
    planned = {m["module"] for m in body["planned_modules"]}
    assert "turbine_health_monitoring" in mounted
    assert "anomaly_detection" in mounted
    assert "unified_maintenance_decision" in mounted
    assert not planned
    assert not (mounted & planned), "a module cannot be both mounted and planned"
    assert body["advisory_only"] is True


@needs_all_artifacts
def test_health_reports_ok(client):
    """With artifacts present health must be 'ok'."""
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_artifact_status"] == "loaded"
    assert body["model_version"]
    assert body["timestamp"]


def test_health_reports_degraded_without_artifacts(broken_client):
    """Health must return 200 and 'degraded' when the model is missing."""
    response = broken_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert "run_failure_pipeline" in body["detail"]


def test_prediction_without_artifacts_returns_503(broken_client):
    """Predicting without a model must be a 503 carrying the fix command."""
    response = broken_client.post("/failure/predict", json=_window_payload())
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["error"] == "model_unavailable"
    assert "run_failure_pipeline" in detail["hint"]


# ---------------------------------------------------------------------------
# Model information
# ---------------------------------------------------------------------------
@needs_artifacts
def test_model_info(client):
    """Model info must expose the published metadata."""
    body = client.get("/failure/model-info").json()
    assert body["model_name"] == "failure_prediction"
    assert body["horizon_hours"] == 48
    assert 0.0 < body["threshold"] < 1.0
    assert body["n_features"] > 0
    assert body["advisory_only"] is True
    assert body["selection_rationale"]


@needs_artifacts
def test_features_endpoint(client):
    """The feature contract must list every feature in order."""
    body = client.get("/failure/features").json()
    assert body["n_features"] == len(body["features"])
    assert len(set(body["features"])) == body["n_features"]
    assert "raw" in body["feature_groups"]


@needs_artifacts
def test_metrics_endpoint(client):
    """Metrics must be served and flagged as synthetic."""
    body = client.get("/failure/metrics").json()
    assert body["is_synthetic"] is True
    assert "metrics" in body
    assert "disclaimer" in body


@needs_artifacts
def test_example_endpoint_returns_usable_payloads(client):
    """The example payloads must actually be accepted by the endpoints."""
    example = client.get("/failure/example").json()
    assert client.post("/failure/predict", json=example["window_request"]).status_code == 200
    assert (
        client.post("/failure/predict/prepared", json=example["prepared_request"]).status_code
        == 200
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@needs_artifacts
def test_valid_prediction_response_schema(client):
    """A successful prediction must match the documented response schema."""
    response = client.post("/failure/predict", json=_window_payload())
    assert response.status_code == 200
    body = response.json()

    required = {
        "turbine_id",
        "timestamp",
        "failure_probability",
        "prediction",
        "risk_level",
        "threshold",
        "top_risk_factors",
        "recommendation",
        "model_version",
        "advisory_only",
        "explanation",
        "horizon_hours",
    }
    assert required <= set(body)
    assert 0.0 <= body["failure_probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["risk_level"] in ("low", "medium", "high")
    assert body["advisory_only"] is True
    for factor in body["top_risk_factors"]:
        assert set(factor) >= {"feature", "impact", "direction"}
        assert factor["direction"] in ("increases_risk", "decreases_risk")


@needs_artifacts
def test_batch_prediction(client):
    """Batch prediction must return one result per submitted window."""
    payload = {"windows": [_window_payload("T01"), _window_payload("T02")]}
    response = client.post("/failure/predict/batch", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert [p["turbine_id"] for p in body["predictions"]] == ["T01", "T02"]


@needs_artifacts
def test_short_window_returns_422(client):
    """Too little history must be a 422 with an actionable hint."""
    response = client.post("/failure/predict", json=_window_payload(hours=3))
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "insufficient_history"


@needs_artifacts
def test_prepared_prediction_rejects_incomplete_features(client):
    """A short feature vector must be a 422 pointing at /failure/features."""
    response = client.post(
        "/failure/predict/prepared",
        json={
            "turbine_id": "T01",
            "timestamp": "2025-01-01T00:00:00Z",
            "features": {"vibration": 3.0},
        },
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "feature_contract_violation"
    assert "/failure/features" in detail["hint"]


def test_malformed_body_returns_structured_error(schema_client):
    """A schema violation must return the structured validation envelope."""
    response = schema_client.post("/failure/predict", json={"turbine_id": "T01"})
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert "/failure/example" in body["hint"]
    assert body["errors"]


def test_impossible_sensor_value_is_rejected(schema_client):
    """Out-of-range sensor input must be refused at the schema boundary."""
    payload = _window_payload(hours=3)
    payload["observations"][0]["vibration"] = 99999.0
    assert schema_client.post("/failure/predict", json=payload).status_code == 422


def test_unknown_field_is_rejected(schema_client):
    """Unexpected fields must be refused rather than silently ignored."""
    payload = _window_payload(hours=3)
    payload["surprise"] = 1
    assert schema_client.post("/failure/predict", json=payload).status_code == 422


def test_empty_batch_is_rejected(schema_client):
    """An empty batch must fail schema validation."""
    assert schema_client.post("/failure/predict/batch", json={"windows": []}).status_code == 422
