"""HTTP contract tests for anomaly detection."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from wind_turbine_pm.anomaly.config import load_anomaly_config
from wind_turbine_pm.anomaly.persistence import bundle_available
from wind_turbine_pm.api.dependencies import provide_anomaly_service
from wind_turbine_pm.api.main import app
from wind_turbine_pm.config import Config
from wind_turbine_pm.services.anomaly_detection_service import AnomalyDetectionService

needs_artifacts = pytest.mark.skipif(
    not bundle_available(load_anomaly_config()),
    reason="Anomaly artifacts not built; run python scripts/run_anomaly_pipeline.py",
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_openapi_lists_complete_anomaly_surface(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {
        "/anomaly/model-info",
        "/anomaly/metrics",
        "/anomaly/features",
        "/anomaly/example",
        "/anomaly/detect",
        "/anomaly/detect/batch",
    } <= set(paths)


def test_missing_anomaly_artifact_is_503(
    client: TestClient, anomaly_config: Config, tmp_path
) -> None:
    data = anomaly_config.to_dict()
    data["paths"]["artifacts_models"] = str(tmp_path / "models")
    data["paths"]["artifacts_metadata"] = str(tmp_path / "metadata")
    app.dependency_overrides[provide_anomaly_service] = lambda: AnomalyDetectionService(
        Config(data)
    )
    try:
        response = client.get("/anomaly/model-info")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "anomaly_model_unavailable"
    assert "run_anomaly_pipeline" in response.json()["detail"]["hint"]


@needs_artifacts
def test_model_features_metrics_and_example_contract(client: TestClient) -> None:
    info = client.get("/anomaly/model-info")
    assert info.status_code == 200
    assert info.json()["algorithm"] in {
        "IsolationForest",
        "LocalOutlierFactor",
        "OneClassSVM",
    }
    features = client.get("/anomaly/features").json()
    assert features["n_features"] == len(features["features"])
    assert "degradation_level" in features["excluded_truth_columns"]
    metrics = client.get("/anomaly/metrics").json()
    assert 0.035 <= metrics["warning_healthy_alert_rate"] <= 0.065
    assert 0.004 <= metrics["alarm_healthy_alert_rate"] <= 0.016
    example = client.get("/anomaly/example").json()
    result = client.post("/anomaly/detect", json=example["window_request"])
    assert result.status_code == 200, result.text
    assert result.json()["advisory_only"] is True


@needs_artifacts
def test_batch_and_short_window_validation(client: TestClient) -> None:
    window = client.get("/anomaly/example").json()["window_request"]
    batch = client.post("/anomaly/detect/batch", json={"windows": [window, window]})
    assert batch.status_code == 200
    assert batch.json()["count"] == 2
    short = {**window, "observations": window["observations"][-3:]}
    invalid = client.post("/anomaly/detect", json=short)
    assert invalid.status_code == 422
    assert invalid.json()["detail"]["error"] == "insufficient_history"
