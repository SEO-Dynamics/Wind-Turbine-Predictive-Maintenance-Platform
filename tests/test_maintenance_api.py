"""HTTP contract tests for unified maintenance decision support."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from wind_turbine_pm.anomaly.config import load_anomaly_config
from wind_turbine_pm.anomaly.persistence import bundle_available
from wind_turbine_pm.api.dependencies import provide_maintenance_service
from wind_turbine_pm.api.main import app
from wind_turbine_pm.config import Config
from wind_turbine_pm.services.anomaly_detection_service import AnomalyDetectionService
from wind_turbine_pm.services.failure_prediction_service import FailurePredictionService
from wind_turbine_pm.services.health_monitoring_service import HealthMonitoringService
from wind_turbine_pm.services.maintenance_service import MaintenanceDecisionService

needs_artifacts = pytest.mark.skipif(
    not bundle_available(load_anomaly_config()),
    reason="Stage 3 artifacts not built; run python scripts/run_all_pipelines.py",
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _unready_service(cfg: Config, tmp_path) -> MaintenanceDecisionService:
    data = cfg.to_dict()
    data["paths"]["artifacts_models"] = str(tmp_path / "models")
    data["paths"]["artifacts_metadata"] = str(tmp_path / "metadata")
    isolated = Config(data)
    return MaintenanceDecisionService(
        isolated,
        failure=FailurePredictionService(isolated),
        health=HealthMonitoringService(isolated),
        anomaly=AnomalyDetectionService(isolated),
    )


def test_openapi_lists_complete_maintenance_surface(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {
        "/maintenance/policy",
        "/maintenance/example",
        "/maintenance/assess",
        "/maintenance/assess/batch",
    } <= set(paths)


def test_policy_is_available_without_models(client: TestClient) -> None:
    policy = client.get("/maintenance/policy")
    assert policy.status_code == 200
    assert policy.json()["weights"] == {"failure": 0.5, "anomaly": 0.3, "health": 0.2}
    assert policy.json()["advisory_only"] is True


def test_assess_returns_503_when_all_models_are_missing(
    client: TestClient, anomaly_config: Config, tmp_path
) -> None:
    app.dependency_overrides[provide_maintenance_service] = lambda: _unready_service(
        anomaly_config, tmp_path
    )
    try:
        payload = client.get("/maintenance/example").json()["window_request"]
        response = client.post("/maintenance/assess", json=payload)
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "all_models_unavailable"
    assert "run_all_pipelines" in response.json()["detail"]["hint"]


@needs_artifacts
def test_assess_and_batch_preserve_three_source_outputs(client: TestClient) -> None:
    payload = client.get("/maintenance/example").json()["window_request"]
    response = client.post("/maintenance/assess", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["coverage"] == 1.0
    assert body["missing_modules"] == []
    assert body["failure"] is not None
    assert body["health"] is not None
    assert body["anomaly"] is not None
    assert body["recommendation"]["action"] in {
        "routine_monitoring",
        "plan_inspection",
        "urgent_review",
        "immediate_engineering_review",
    }
    batch = client.post("/maintenance/assess/batch", json={"windows": [payload, payload]})
    assert batch.status_code == 200, batch.text
    assert batch.json()["count"] == 2
