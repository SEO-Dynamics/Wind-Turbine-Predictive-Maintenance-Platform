"""Table-driven unified score, guardrail and action-policy tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import HealthClass, OperatingRegime, RiskLevel
from wind_turbine_pm.contracts.anomaly import AnomalyPrediction
from wind_turbine_pm.contracts.health import HealthAssessment
from wind_turbine_pm.contracts.observations import TurbineWindow
from wind_turbine_pm.contracts.predictions import FailurePrediction
from wind_turbine_pm.services.failure_prediction_service import ServiceNotReadyError
from wind_turbine_pm.services.maintenance_service import MaintenanceDecisionService

NOW = datetime(2026, 1, 1, tzinfo=UTC)


@dataclass
class _FailureStub:
    probability: float = 0.1
    positive: bool = False
    is_ready: bool = True

    def minimum_history_hours(self) -> float:
        return 1.0

    def predict_from_window(self, window: TurbineWindow) -> FailurePrediction:
        return FailurePrediction(
            turbine_id=window.turbine_id,
            timestamp=window.end_timestamp,
            model_version="failure-test",
            failure_probability=self.probability,
            prediction=int(self.positive),
            risk_level=RiskLevel.HIGH if self.positive else RiskLevel.LOW,
            threshold=0.5,
            horizon_hours=48,
        )


@dataclass
class _HealthStub:
    score: float = 90.0
    health_class: HealthClass = HealthClass.HEALTHY
    quality: float = 1.0
    is_ready: bool = True

    def minimum_history_hours(self) -> float:
        return 1.0

    def assess_from_window(self, window: TurbineWindow) -> HealthAssessment:
        return HealthAssessment(
            turbine_id=window.turbine_id,
            timestamp=window.end_timestamp,
            model_version="health-test",
            health_score=self.score,
            raw_health_score=self.score,
            drift_penalty=0.0,
            health_class=self.health_class,
            operating_regime=OperatingRegime.MEDIUM_LOAD,
            data_quality=self.quality,
        )


@dataclass
class _AnomalyStub:
    score: float = 0.1
    severity: str = "normal"
    quality: float = 1.0
    is_ready: bool = True

    def minimum_history_hours(self) -> float:
        return 1.0

    def detect_from_window(self, window: TurbineWindow) -> AnomalyPrediction:
        return AnomalyPrediction(
            turbine_id=window.turbine_id,
            timestamp=window.end_timestamp,
            model_version="anomaly-test",
            anomaly_score=self.score,
            raw_anomaly_score=self.score,
            is_anomaly=self.severity != "normal",
            severity=self.severity,
            operating_regime=OperatingRegime.MEDIUM_LOAD,
            data_quality=self.quality,
        )


def _window(state: str = "normal") -> TurbineWindow:
    return TurbineWindow.model_validate(
        {
            "turbine_id": "T01",
            "observations": [
                {
                    "turbine_id": "T01",
                    "timestamp": NOW,
                    "operational_status": state,
                }
            ],
        }
    )


def _service(
    cfg: Config,
    failure: _FailureStub | None = None,
    health: _HealthStub | None = None,
    anomaly: _AnomalyStub | None = None,
) -> MaintenanceDecisionService:
    return MaintenanceDecisionService(
        cfg,
        failure=failure or _FailureStub(),
        health=health or _HealthStub(),
        anomaly=anomaly or _AnomalyStub(),
    )


def test_default_weight_math_is_50_30_20(anomaly_config: Config) -> None:
    result = _service(
        anomaly_config,
        _FailureStub(probability=0.8),
        _HealthStub(score=60.0),
        _AnomalyStub(score=0.4),
    ).assess(_window())
    assert result.unified_risk_score == pytest.approx(0.8 * 0.5 + 0.4 * 0.3 + 0.4 * 0.2)
    assert result.coverage == 1.0
    assert result.component_scores == {"failure": 0.8, "health": 0.4, "anomaly": 0.4}


def test_partial_coverage_renormalizes_available_weights(anomaly_config: Config) -> None:
    result = _service(
        anomaly_config,
        _FailureStub(probability=0.8),
        _HealthStub(is_ready=False),
        _AnomalyStub(score=0.2),
    ).assess(_window())
    assert result.unified_risk_score == pytest.approx((0.8 * 0.5 + 0.2 * 0.3) / 0.8)
    assert result.coverage == 0.8
    assert result.missing_modules == ["health"]
    assert result.decision_confidence in {"low", "medium"}


@pytest.mark.parametrize(
    ("failure", "health", "anomaly", "expected_risk", "expected_action"),
    [
        (_FailureStub(), _HealthStub(), _AnomalyStub(), RiskLevel.LOW, "routine_monitoring"),
        (
            _FailureStub(),
            _HealthStub(health_class=HealthClass.MONITOR, score=75.0),
            _AnomalyStub(),
            RiskLevel.MEDIUM,
            "plan_inspection",
        ),
        (
            _FailureStub(positive=True, probability=0.7),
            _HealthStub(),
            _AnomalyStub(),
            RiskLevel.HIGH,
            "urgent_review",
        ),
        (
            _FailureStub(positive=True, probability=0.7),
            _HealthStub(health_class=HealthClass.CRITICAL, score=20.0),
            _AnomalyStub(),
            RiskLevel.HIGH,
            "immediate_engineering_review",
        ),
    ],
)
def test_four_action_layers_and_guardrails(
    anomaly_config: Config,
    failure: _FailureStub,
    health: _HealthStub,
    anomaly: _AnomalyStub,
    expected_risk: RiskLevel,
    expected_action: str,
) -> None:
    result = _service(anomaly_config, failure, health, anomaly).assess(_window())
    assert result.risk_level is expected_risk
    assert result.recommendation.action == expected_action


def test_anomaly_alarm_is_high_and_two_serious_signals_are_immediate(
    anomaly_config: Config,
) -> None:
    one = _service(
        anomaly_config,
        anomaly=_AnomalyStub(score=0.995, severity="alarm"),
    ).assess(_window())
    assert one.risk_level is RiskLevel.HIGH
    assert one.recommendation.action == "urgent_review"

    two = _service(
        anomaly_config,
        failure=_FailureStub(positive=True, probability=0.8),
        anomaly=_AnomalyStub(score=0.995, severity="alarm"),
    ).assess(_window())
    assert two.recommendation.action == "immediate_engineering_review"


def test_controller_fault_is_immediate(anomaly_config: Config) -> None:
    result = _service(anomaly_config).assess(_window("fault"))
    assert result.recommendation.action == "immediate_engineering_review"


def test_low_data_quality_reduces_confidence_not_machine_risk(anomaly_config: Config) -> None:
    high_quality = _service(anomaly_config).assess(_window())
    low_quality = _service(
        anomaly_config,
        health=_HealthStub(quality=0.2),
        anomaly=_AnomalyStub(quality=0.2),
    ).assess(_window())
    assert low_quality.unified_risk_score == high_quality.unified_risk_score
    assert low_quality.risk_level == high_quality.risk_level
    assert low_quality.decision_confidence == "low"


def test_no_ready_model_raises_503_domain_error(anomaly_config: Config) -> None:
    service = _service(
        anomaly_config,
        _FailureStub(is_ready=False),
        _HealthStub(is_ready=False),
        _AnomalyStub(is_ready=False),
    )
    assert service.status()["model_loaded"] is False
    with pytest.raises(ServiceNotReadyError, match="No component model"):
        service.assess(_window())


def test_batch_is_sorted_by_action_then_score(anomaly_config: Config) -> None:
    service = _service(
        anomaly_config,
        failure=_FailureStub(positive=True, probability=0.7),
    )
    result = service.assess_batch([_window(), _window()])
    assert result.count == 2
    assert all(item.recommendation.action == "urgent_review" for item in result.assessments)
