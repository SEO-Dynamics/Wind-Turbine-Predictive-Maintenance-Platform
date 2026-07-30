"""Artifact readiness and serving tests for anomaly detection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from wind_turbine_pm.anomaly.config import load_anomaly_config
from wind_turbine_pm.anomaly.persistence import bundle_available
from wind_turbine_pm.api.examples import example_window
from wind_turbine_pm.config import Config
from wind_turbine_pm.contracts.observations import TurbineWindow
from wind_turbine_pm.services.anomaly_detection_service import AnomalyDetectionService
from wind_turbine_pm.services.failure_prediction_service import (
    InsufficientHistoryError,
    ServiceNotReadyError,
)

needs_artifacts = pytest.mark.skipif(
    not bundle_available(load_anomaly_config()),
    reason="Anomaly artifacts not built; run python scripts/run_anomaly_pipeline.py",
)


def _window(hours: int = 97, state: str = "normal") -> TurbineWindow:
    body = example_window(hours - 25)
    body["observations"][-1]["operational_status"] = state
    return TurbineWindow.model_validate(body)


def test_missing_artifacts_report_pipeline_command(anomaly_config: Config, tmp_path) -> None:
    data = anomaly_config.to_dict()
    data["paths"]["artifacts_models"] = str(tmp_path / "models")
    data["paths"]["artifacts_metadata"] = str(tmp_path / "metadata")
    service = AnomalyDetectionService(Config(data))
    assert not service.is_ready
    assert "run_anomaly_pipeline" in service.status().detail
    with pytest.raises(ServiceNotReadyError, match="Anomaly model artifacts"):
        service.detect_from_window(_window())


@needs_artifacts
def test_short_window_is_rejected() -> None:
    service = AnomalyDetectionService(load_anomaly_config())
    end = datetime(2026, 1, 1, tzinfo=UTC)
    body = example_window(72)
    body["observations"] = body["observations"][:4]
    for index, observation in enumerate(body["observations"]):
        observation["timestamp"] = (end - timedelta(hours=3 - index)).isoformat()
    with pytest.raises(InsufficientHistoryError):
        service.detect_from_window(TurbineWindow.model_validate(body))


@needs_artifacts
def test_prediction_contract_and_score_orientation() -> None:
    service = AnomalyDetectionService(load_anomaly_config())
    normal = service.detect_from_window(_window())
    assert normal.assessment_state == "scored"
    assert 0.0 <= normal.anomaly_score <= 1.0
    assert normal.warning_threshold == 0.95
    assert normal.alarm_threshold == 0.99
    assert normal.severity in {"normal", "warning", "alarm"}
    assert normal.advisory_only is True
    for signal in normal.contributing_signals:
        assert "not a causal diagnosis" in (signal.description or "")


@needs_artifacts
@pytest.mark.parametrize("state", ["fault", "maintenance"])
def test_known_controller_state_is_not_anomaly_scored(state: str) -> None:
    result = AnomalyDetectionService(load_anomaly_config()).detect_from_window(_window(state=state))
    assert result.assessment_state == "not_applicable"
    assert result.severity == "not_applicable"
    assert result.anomaly_score is None


@needs_artifacts
def test_batch_validation_and_count() -> None:
    service = AnomalyDetectionService(load_anomaly_config())
    result = service.detect_batch_from_windows([_window(), _window()])
    assert result.count == 2
    assert len(result.predictions) == 2
    limit = int(service.config.require("anomaly.serving.max_batch_turbines"))
    with pytest.raises(ValueError, match="exceeds"):
        service.detect_batch_from_windows([_window()] * (limit + 1))
