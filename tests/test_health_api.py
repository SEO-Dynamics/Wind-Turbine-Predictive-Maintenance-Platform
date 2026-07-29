"""Tests for the Turbine Health Monitoring HTTP layer.

Endpoints that need a published model are skipped when artifacts are absent; the
artifact-missing behaviour itself is tested explicitly with an isolated service,
and the schema-rejection tests use a stub so they stay meaningful in an
environment where nothing has been trained yet.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from wind_turbine_pm.api.dependencies import provide_health_service
from wind_turbine_pm.api.main import app
from wind_turbine_pm.config import Config
from wind_turbine_pm.health.config import load_health_config
from wind_turbine_pm.health.persistence import health_bundle_available
from wind_turbine_pm.services.health_monitoring_service import HealthMonitoringService

needs_artifacts = pytest.mark.skipif(
    not health_bundle_available(load_health_config()),
    reason="Health artifacts not built; run python scripts/run_health_pipeline.py",
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    """A test client bound to the real application."""
    return TestClient(app)


@pytest.fixture
def unbuilt_client(tmp_path) -> TestClient:
    """A client whose health service points at an empty artifact directory."""
    data = load_health_config().to_dict()
    data["paths"]["artifacts_models"] = str(tmp_path / "models")
    data["paths"]["artifacts_metadata"] = str(tmp_path / "metadata")
    unbuilt = HealthMonitoringService(Config(data))

    # Only the *base* dependency is overridden. `provide_ready_health_service`
    # composes on top of it, so the real 503 readiness gate still runs - which is
    # exactly the behaviour under test.
    app.dependency_overrides[provide_health_service] = lambda: unbuilt
    yield TestClient(app)
    app.dependency_overrides.clear()


class _ReadyStub:
    """Minimal stand-in that satisfies the readiness gate but never assesses.

    Request-body validation happens *after* FastAPI resolves dependencies, so
    without a model the readiness gate short-circuits every request with 503 and
    schema-rejection tests could never reach their assertion.

    The assessment methods exist but must never run.  They are needed because the
    handler resolves ``service.assess_from_window`` *before* evaluating
    ``request.to_window()``, so an absent attribute would raise before the
    window's own validation could reject the payload.
    """

    is_ready = True
    metadata = object()

    def minimum_history_hours(self) -> float:
        return 72.0

    def assess_from_window(self, window):  # pragma: no cover - must not be reached
        raise AssertionError("the request should have been rejected before assessment")

    def assess_batch_from_windows(self, windows):  # pragma: no cover - must not be reached
        raise AssertionError("the request should have been rejected before assessment")


@pytest.fixture
def schema_client() -> TestClient:
    """A client whose health service is 'ready', for schema-rejection tests."""
    app.dependency_overrides[provide_health_service] = _ReadyStub
    yield TestClient(app)
    app.dependency_overrides.clear()


def _window_payload(turbine: str = "T01", hours: int = 100) -> dict:
    """A realistic window: values vary, so the frozen-sensor rule is not tripped."""
    end = datetime(2025, 6, 1, tzinfo=UTC)
    observations = []
    for index in range(hours):
        swing = math.sin(2.0 * math.pi * (index % 24) / 24.0)
        wind = 8.4 + 2.2 * swing
        load = max(0.0, min(1.0, (wind - 3.0) / 9.5))
        observations.append(
            {
                "turbine_id": turbine,
                "timestamp": (end - timedelta(hours=hours - 1 - index)).isoformat(),
                "wind_speed": round(wind, 3),
                "rotor_speed": round(9.5 + 3.4 * load, 3),
                "generator_speed": round(880.0 + 420.0 * load, 2),
                "power_output": round(120.0 + 1350.0 * load, 2),
                "generator_temperature": round(52.0 + 16.0 * load, 3),
                "gearbox_temperature": round(46.0 + 14.0 * load, 3),
                "bearing_temperature": round(41.0 + 11.0 * load, 3),
                "oil_temperature": round(40.0 + 10.0 * load, 3),
                "oil_pressure": round(5.4 - 0.45 * load, 3),
                "vibration": round(2.4 + 1.3 * load, 3),
                "ambient_temperature": round(11.0 + 4.0 * swing, 3),
                "nacelle_temperature": round(18.0 + 5.0 * load, 3),
                "hydraulic_pressure": round(191.0 - 6.0 * load, 2),
                "brake_temperature": round(16.5 + 2.0 * load, 3),
                "operational_status": "normal",
            }
        )
    return {"turbine_id": turbine, "observations": observations}


# ---------------------------------------------------------------------------
# Platform surface
# ---------------------------------------------------------------------------
def test_health_probe_reports_both_modules(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()

    assert body["status"] in {"ok", "degraded"}
    assert {"failure_prediction", "turbine_health_monitoring"} <= set(body["modules"])
    for info in body["modules"].values():
        assert "model_loaded" in info
    # Retained for existing probes that read the flat field.
    assert "model_loaded" in body


def test_health_module_is_mounted_not_planned(client: TestClient) -> None:
    body = client.get("/").json()
    mounted = {module["module"] for module in body["modules"]}
    planned = {module["module"] for module in body["planned_modules"]}

    assert "turbine_health_monitoring" in mounted
    assert "turbine_health_monitoring" not in planned
    descriptor = next(
        module for module in body["modules"] if module["module"] == "turbine_health_monitoring"
    )
    assert descriptor["prefix"] == "/health-monitoring"
    assert descriptor["status"] == "available"


def test_module_prefix_does_not_shadow_the_liveness_probe(client: TestClient) -> None:
    # /health must stay the platform probe; the module lives under its own prefix.
    assert client.get("/health").status_code == 200
    assert "modules" in client.get("/health").json()


def test_openapi_documents_the_health_endpoints(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/health-monitoring/assess" in schema["paths"]
    assert "/health-monitoring/assess/batch" in schema["paths"]
    assert "/health-monitoring/sensor-rules" in schema["paths"]
    tags = {tag["name"] for tag in schema.get("tags", [])}
    assert "turbine-health-monitoring" in tags


# ---------------------------------------------------------------------------
# Degraded mode
# ---------------------------------------------------------------------------
def test_missing_artifacts_yield_503_with_the_fixing_command(
    unbuilt_client: TestClient,
) -> None:
    for path in (
        "/health-monitoring/model-info",
        "/health-monitoring/metrics",
        "/health-monitoring/sensor-rules",
        "/health-monitoring/example",
    ):
        response = unbuilt_client.get(path)
        assert response.status_code == 503, path
        detail = response.json()["detail"]
        assert detail["error"] == "health_model_unavailable"
        assert "run_health_pipeline" in detail["hint"]


def test_assess_without_artifacts_yields_503(unbuilt_client: TestClient) -> None:
    response = unbuilt_client.post("/health-monitoring/assess", json=_window_payload())
    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "health_model_unavailable"


def test_liveness_still_answers_when_the_health_model_is_missing(
    unbuilt_client: TestClient,
) -> None:
    # The whole point of the lazy loading: the process serves /health regardless.
    response = unbuilt_client.get("/health")
    assert response.status_code == 200
    assert response.json()["modules"]["turbine_health_monitoring"]["model_loaded"] is False


def test_failure_module_is_unaffected_by_a_missing_health_model(
    unbuilt_client: TestClient,
) -> None:
    # The two modules are independent; a missing health artifact must not make the
    # failure endpoints report a health problem.
    response = unbuilt_client.get("/failure/model-info")
    assert response.status_code in {200, 503}
    if response.status_code == 503:
        assert response.json()["detail"]["error"] == "model_unavailable"


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------
def test_unknown_field_is_rejected(schema_client: TestClient) -> None:
    payload = _window_payload()
    payload["unexpected"] = 1
    response = schema_client.post("/health-monitoring/assess", json=payload)
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_out_of_range_reading_is_rejected_at_the_boundary(schema_client: TestClient) -> None:
    payload = _window_payload()
    payload["observations"][0]["vibration"] = 500.0  # beyond the contract's bound
    response = schema_client.post("/health-monitoring/assess", json=payload)
    assert response.status_code == 422


def test_single_observation_window_is_rejected(schema_client: TestClient) -> None:
    payload = _window_payload()
    payload["observations"] = payload["observations"][:1]
    response = schema_client.post("/health-monitoring/assess", json=payload)
    assert response.status_code == 422


def test_mismatched_turbine_in_the_window_is_rejected(schema_client: TestClient) -> None:
    payload = _window_payload()
    payload["observations"][0]["turbine_id"] = "T99"
    response = schema_client.post("/health-monitoring/assess", json=payload)
    assert response.status_code == 422


def test_empty_batch_is_rejected(schema_client: TestClient) -> None:
    response = schema_client.post("/health-monitoring/assess/batch", json={"windows": []})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Documentation endpoints
# ---------------------------------------------------------------------------
@needs_artifacts
def test_model_info_reports_the_published_contract(client: TestClient) -> None:
    body = client.get("/health-monitoring/model-info").json()
    assert body["model_name"]
    assert body["algorithm"]
    assert body["n_features"] > 0
    assert body["advisory_only"] is True
    # The independence of the label source is part of the published contract.
    assert body["target_source"]
    bands = body["health_classes"]
    assert bands["degraded_min"] < bands["monitor_min"] < bands["healthy_min"]


@needs_artifacts
def test_model_info_exposes_the_drift_calibration(client: TestClient) -> None:
    drift = client.get("/health-monitoring/model-info").json()["drift"]
    assert "cusum" in drift
    calibration = drift.get("calibration")
    # An uncalibrated artifact is allowed but must be visible as such.
    if calibration is not None:
        assert calibration["warning_at"]
        assert 0.0 < calibration["target_warning_rate"] < 1.0


@needs_artifacts
def test_sensor_rules_are_published_with_their_provenance(client: TestClient) -> None:
    body = client.get("/health-monitoring/sensor-rules").json()
    assert body["n_rules"] == len(body["rules"])
    assert body["n_rules"] > 0
    for rule in body["rules"]:
        assert rule["sensor"]
        assert rule["component"]
        assert rule["direction"] in {"high_is_bad", "low_is_bad"}
        assert rule["source"] in {"expert_judgement", "industry_standard", "data_analysis"}
        assert rule["rationale"]
    assert "re-derived" in body["note"]


@needs_artifacts
def test_metrics_carry_a_synthetic_data_disclaimer(client: TestClient) -> None:
    body = client.get("/health-monitoring/metrics").json()
    assert body["metrics"]
    assert "synthetic" in body["disclaimer"].lower()


@needs_artifacts
def test_example_payload_is_accepted_by_the_assess_endpoint(client: TestClient) -> None:
    example = client.get("/health-monitoring/example").json()
    assert example["min_history_hours"] > 0

    response = client.post("/health-monitoring/assess", json=example["window_request"])
    assert response.status_code == 200, response.text
    body = response.json()
    # The documented example must not look pathological: a flat payload would
    # trip the frozen-sensor rule on every channel.
    assert body["data_quality"] > 0.9
    assert body["rule_violations"] == []


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------
@needs_artifacts
def test_assess_returns_a_complete_assessment(client: TestClient) -> None:
    response = client.post("/health-monitoring/assess", json=_window_payload())
    assert response.status_code == 200, response.text
    body = response.json()

    assert 0.0 <= body["health_score"] <= 100.0
    assert body["health_class"] in {"healthy", "monitor", "degraded", "critical"}
    assert body["advisory_only"] is True
    assert body["explanation"]
    assert body["recommendation"]
    assert body["operating_regime"]
    assert isinstance(body["component_health"], list)
    assert isinstance(body["drift_signals"], list)
    # The published score must be the raw score less the deduction.
    assert body["health_score"] == pytest.approx(
        max(body["raw_health_score"] - body["drift_penalty"], 0.0), abs=1e-6
    )


@needs_artifacts
def test_recommendation_always_carries_the_disclaimer(client: TestClient) -> None:
    from wind_turbine_pm.constants import ADVISORY_DISCLAIMER

    body = client.post("/health-monitoring/assess", json=_window_payload()).json()
    assert ADVISORY_DISCLAIMER in body["recommendation"]


@needs_artifacts
def test_short_window_yields_422_with_the_required_history(client: TestClient) -> None:
    payload = _window_payload(hours=5)
    response = client.post("/health-monitoring/assess", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "insufficient_history"
    assert "hours of observations" in detail["hint"]


@needs_artifacts
def test_batch_assesses_every_window(client: TestClient) -> None:
    payload = {"windows": [_window_payload("T01"), _window_payload("T02")]}
    response = client.post("/health-monitoring/assess/batch", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["count"] == 2
    assert len(body["assessments"]) == 2
    assert {item["turbine_id"] for item in body["assessments"]} == {"T01", "T02"}
    assert body["advisory_only"] is True
    assert body["model_version"]
