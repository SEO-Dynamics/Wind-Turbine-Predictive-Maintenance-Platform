"""Unified evidence composition and deterministic maintenance policy."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from wind_turbine_pm.anomaly.config import get_anomaly_config
from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import ADVISORY_DISCLAIMER, HealthClass, RiskLevel
from wind_turbine_pm.contracts.maintenance import (
    BatchUnifiedRiskAssessment,
    MaintenanceRecommendation,
    UnifiedRiskAssessment,
)
from wind_turbine_pm.contracts.observations import TurbineWindow
from wind_turbine_pm.models.threshold import map_risk_level
from wind_turbine_pm.services.anomaly_detection_service import (
    AnomalyDetectionService,
    get_anomaly_service,
)
from wind_turbine_pm.services.failure_prediction_service import (
    FailurePredictionService,
    ServiceNotReadyError,
    get_service,
)
from wind_turbine_pm.services.health_monitoring_service import (
    HealthMonitoringService,
    get_health_service,
)

ALL_MODULES = ("failure", "anomaly", "health")


class MaintenanceDecisionService:
    """Compose module services without reaching into their model internals."""

    def __init__(
        self,
        cfg: Config | None = None,
        failure: FailurePredictionService | None = None,
        health: HealthMonitoringService | None = None,
        anomaly: AnomalyDetectionService | None = None,
    ) -> None:
        self._cfg = cfg if cfg is not None else get_anomaly_config()
        self.failure = failure if failure is not None else get_service()
        self.health = health if health is not None else get_health_service()
        self.anomaly = anomaly if anomaly is not None else get_anomaly_service()

    @property
    def policy_version(self) -> str:
        return str(self._cfg.require("maintenance.version"))

    def policy(self) -> dict[str, object]:
        return {
            "version": self.policy_version,
            "weights": self._cfg.require("maintenance.weights").to_dict(),
            "actions": self._cfg.require("maintenance.actions").to_dict(),
            "guardrails": {
                "high": ["failure positive", "health critical", "anomaly alarm"],
                "medium": ["health monitor/degraded", "anomaly warning"],
                "immediate": ["controller fault", "two or more high-severity signals"],
            },
            "advisory_only": True,
        }

    def status(self) -> dict[str, object]:
        ready = {
            "failure": self.failure.is_ready,
            "health": self.health.is_ready,
            "anomaly": self.anomaly.is_ready,
        }
        weights = self._weights()
        coverage = sum(weights[name] for name, available in ready.items() if available)
        return {
            "model_loaded": any(ready.values()),
            "model_version": self.policy_version,
            "algorithm": "deterministic_policy",
            "n_features": 0,
            "coverage": round(coverage, 4),
            "ready_modules": [name for name, available in ready.items() if available],
            "missing_modules": [name for name, available in ready.items() if not available],
            "detail": (
                "Maintenance policy can assess available evidence."
                if any(ready.values())
                else "No component model is available."
            ),
        }

    def _weights(self) -> dict[str, float]:
        return {
            name: float(self._cfg.require(f"maintenance.weights.{name}")) for name in ALL_MODULES
        }

    @staticmethod
    def _component_from_feature(feature: str) -> str | None:
        mapping = {
            "gearbox": "gearbox",
            "bearing": "drivetrain",
            "vibration": "drivetrain",
            "generator": "generator",
            "oil_": "lubrication",
            "hydraulic": "hydraulic",
            "brake": "brake",
            "rotor": "rotor",
        }
        return next((component for token, component in mapping.items() if token in feature), None)

    def _recommendation(
        self,
        risk_level: RiskLevel,
        severe_count: int,
        controller_state: str,
        reasons: list[str],
        components: list[str],
    ) -> MaintenanceRecommendation:
        actions = self._cfg.require("maintenance.actions")
        if controller_state == "fault" or severe_count >= 2:
            action = "immediate_engineering_review"
            hours = int(actions.require("immediate_review_hours"))
        elif severe_count >= 1 or risk_level is RiskLevel.HIGH:
            action = "urgent_review"
            hours = int(actions.require("urgent_review_hours"))
        elif risk_level is RiskLevel.MEDIUM:
            action = "plan_inspection"
            hours = int(actions.require("planned_inspection_hours"))
        else:
            action = "routine_monitoring"
            hours = None
        phrase = action.replace("_", " ")
        message = (
            f"Recommended action: {phrase}"
            + (f" within {hours} hours." if hours else " at the normal monitoring cadence.")
            + f" {ADVISORY_DISCLAIMER}"
        )
        return MaintenanceRecommendation(
            action=action,
            recommended_within_hours=hours,
            target_components=sorted(set(components)),
            reasons=reasons,
            message=message,
        )

    def assess(self, window: TurbineWindow) -> UnifiedRiskAssessment:
        weights = self._weights()
        missing: list[str] = []
        component_scores: dict[str, float] = {}
        reasons: list[str] = []
        components: list[str] = []
        guardrails: list[str] = []
        severe_count = 0

        failure = None
        if self.failure.is_ready:
            failure = self.failure.predict_from_window(window)
            component_scores["failure"] = failure.failure_probability
            components.extend(
                filter(
                    None,
                    (
                        self._component_from_feature(factor.feature)
                        for factor in failure.top_risk_factors
                    ),
                )
            )
            if failure.prediction == 1:
                severe_count += 1
                guardrails.append("failure_prediction_positive")
                reasons.append(
                    f"Failure model crossed its {failure.threshold:.3f} decision threshold."
                )
        else:
            missing.append("failure")

        health = None
        if self.health.is_ready:
            health = self.health.assess_from_window(window)
            component_scores["health"] = 1.0 - health.health_score / 100.0
            components.extend(
                item.component
                for item in health.component_health
                if item.health_class is not HealthClass.HEALTHY
            )
            if health.health_class is HealthClass.CRITICAL:
                severe_count += 1
                guardrails.append("health_critical")
                reasons.append("Health assessment is Critical.")
            elif health.health_class in {HealthClass.MONITOR, HealthClass.DEGRADED}:
                guardrails.append(f"health_{health.health_class.value}")
                reasons.append(f"Health assessment is {health.health_class.value}.")
        else:
            missing.append("health")

        anomaly = None
        if self.anomaly.is_ready:
            anomaly = self.anomaly.detect_from_window(window)
            if anomaly.anomaly_score is not None:
                component_scores["anomaly"] = anomaly.anomaly_score
            components.extend(
                filter(
                    None,
                    (
                        self._component_from_feature(factor.feature)
                        for factor in anomaly.contributing_signals
                    ),
                )
            )
            if anomaly.severity == "alarm":
                severe_count += 1
                guardrails.append("anomaly_alarm")
                reasons.append("Anomaly score crossed the calibrated 1% healthy alarm limit.")
            elif anomaly.severity == "warning":
                guardrails.append("anomaly_warning")
                reasons.append("Anomaly score crossed the calibrated 5% healthy warning limit.")
        else:
            missing.append("anomaly")

        if not component_scores:
            raise ServiceNotReadyError(
                "No component model is available for maintenance assessment.",
                "python scripts/run_all_pipelines.py",
            )

        available_weight = sum(weights[name] for name in component_scores)
        weighted = (
            sum(component_scores[name] * weights[name] for name in component_scores)
            / available_weight
        )
        low = float(self._cfg.require("risk_levels.low_max"))
        medium = float(self._cfg.require("risk_levels.medium_max"))
        risk_level = RiskLevel(map_risk_level(weighted, low, medium))
        if severe_count:
            risk_level = RiskLevel.HIGH
        elif (
            any(
                guardrail in {"health_monitor", "health_degraded", "anomaly_warning"}
                for guardrail in guardrails
            )
            and risk_level is RiskLevel.LOW
        ):
            risk_level = RiskLevel.MEDIUM

        quality_values = [
            value
            for value in (
                health.data_quality if health is not None else None,
                anomaly.data_quality if anomaly is not None else None,
            )
            if value is not None
        ]
        data_quality = min(quality_values, default=1.0)
        if available_weight >= float(
            self._cfg.require("maintenance.confidence.high_coverage")
        ) and data_quality >= float(self._cfg.require("maintenance.confidence.high_data_quality")):
            confidence = "high"
        elif available_weight >= float(
            self._cfg.require("maintenance.confidence.medium_coverage")
        ) and data_quality >= float(
            self._cfg.require("maintenance.confidence.medium_data_quality")
        ):
            confidence = "medium"
        else:
            confidence = "low"
        if missing:
            reasons.append("Missing module evidence: " + ", ".join(sorted(missing)) + ".")
        if data_quality < 0.75:
            reasons.append(
                "Low sensor data quality reduces confidence; it does not increase machine severity."
            )

        controller_state = str(window.observations[-1].operational_status)
        recommendation = self._recommendation(
            risk_level, severe_count, controller_state, reasons, components
        )
        return UnifiedRiskAssessment(
            turbine_id=window.turbine_id,
            timestamp=window.end_timestamp,
            model_version=self.policy_version,
            unified_risk_score=round(float(np.clip(weighted, 0.0, 1.0)), 6),
            risk_level=risk_level,
            coverage=round(available_weight, 4),
            decision_confidence=confidence,
            data_quality=round(data_quality, 4),
            missing_modules=sorted(missing),
            guardrails_triggered=guardrails,
            component_scores={name: round(value, 6) for name, value in component_scores.items()},
            failure=failure,
            health=health,
            anomaly=anomaly,
            recommendation=recommendation,
        )

    def assess_batch(self, windows: list[TurbineWindow]) -> BatchUnifiedRiskAssessment:
        assessments = [self.assess(window) for window in windows]
        priority = {
            "immediate_engineering_review": 3,
            "urgent_review": 2,
            "plan_inspection": 1,
            "routine_monitoring": 0,
        }
        assessments.sort(
            key=lambda item: (
                priority[item.recommendation.action],
                item.unified_risk_score,
            ),
            reverse=True,
        )
        return BatchUnifiedRiskAssessment(
            assessments=assessments,
            count=len(assessments),
            policy_version=self.policy_version,
        )


@lru_cache(maxsize=1)
def get_maintenance_service() -> MaintenanceDecisionService:
    return MaintenanceDecisionService()
