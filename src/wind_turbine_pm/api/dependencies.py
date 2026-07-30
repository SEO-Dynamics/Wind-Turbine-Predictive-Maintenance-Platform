"""FastAPI dependency providers.

Each module's service is resolved lazily through its own cached accessor
(:func:`~wind_turbine_pm.services.failure_prediction_service.get_service` and
:func:`~wind_turbine_pm.services.health_monitoring_service.get_health_service`),
whose ``lru_cache`` guarantees one loaded model per process.  Importing this
module does not read artifacts and never triggers training, so the API starts
cleanly even when a model has not been built yet - ``/health`` then reports
``degraded`` with the command to fix it.

Model modules are independent: one artifact set being absent does not stop the
other model endpoints. The maintenance router is the exception by design: it
continues with partial evidence and rejects only when every model is unavailable.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, status

from wind_turbine_pm.config import Config, get_config
from wind_turbine_pm.services.anomaly_detection_service import (
    PIPELINE_COMMAND as ANOMALY_PIPELINE_COMMAND,
)
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
    HEALTH_PIPELINE_COMMAND,
    HealthMonitoringService,
    get_health_service,
)
from wind_turbine_pm.services.maintenance_service import (
    MaintenanceDecisionService,
    get_maintenance_service,
)


def provide_config() -> Config:
    """Provide the process-wide configuration.

    Returns:
        The merged configuration.
    """
    return get_config()


def provide_service() -> FailurePredictionService:
    """Provide the shared prediction service without asserting readiness.

    Used by ``/health``, which must be able to report a missing model rather
    than fail.

    Returns:
        The shared service.
    """
    return get_service()


def provide_ready_service(
    service: Annotated[FailurePredictionService, Depends(provide_service)],
) -> FailurePredictionService:
    """Provide the service, rejecting the request when artifacts are missing.

    It resolves the service through :func:`provide_service` rather than calling
    ``get_service()`` directly, so a test overriding the base dependency also
    overrides this one.

    Args:
        service: The shared service, injected.

    Returns:
        A service with loaded artifacts.

    Raises:
        HTTPException: ``503 Service Unavailable`` carrying the command that
            generates the missing artifacts.
    """
    if not service.is_ready:
        status_info = service.status()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "model_unavailable",
                "detail": status_info.detail,
                "hint": "python scripts/run_failure_pipeline.py",
            },
        )
    try:
        _ = service.metadata  # forces the load and surfaces contract errors early
    except ServiceNotReadyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "model_unavailable", "detail": str(exc), "hint": exc.hint},
        ) from exc
    return service


def provide_health_service() -> HealthMonitoringService:
    """Provide the shared health service without asserting readiness.

    Used by ``/health``, which reports each module's artifact status and must be
    able to say "not built yet" rather than fail.

    Returns:
        The shared health service.
    """
    return get_health_service()


def provide_ready_health_service(
    service: Annotated[HealthMonitoringService, Depends(provide_health_service)],
) -> HealthMonitoringService:
    """Provide the health service, rejecting the request when artifacts are missing.

    Resolved through :func:`provide_health_service` rather than by calling
    ``get_health_service()`` directly, so a test overriding the base dependency
    also overrides this one.

    Args:
        service: The shared health service, injected.

    Returns:
        A health service with loaded artifacts.

    Raises:
        HTTPException: ``503 Service Unavailable`` carrying the command that
            generates the missing artifacts.
    """
    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "health_model_unavailable",
                "detail": service.status().detail,
                "hint": HEALTH_PIPELINE_COMMAND,
            },
        )
    try:
        _ = service.metadata  # forces the load and surfaces contract errors early
    except ServiceNotReadyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "health_model_unavailable",
                "detail": str(exc),
                "hint": exc.hint,
            },
        ) from exc
    return service


def provide_anomaly_service() -> AnomalyDetectionService:
    """Provide anomaly detection without asserting artifact readiness."""
    return get_anomaly_service()


def provide_ready_anomaly_service(
    service: Annotated[AnomalyDetectionService, Depends(provide_anomaly_service)],
) -> AnomalyDetectionService:
    """Reject anomaly-only requests until anomaly artifacts are available."""
    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "anomaly_model_unavailable",
                "detail": service.status().detail,
                "hint": ANOMALY_PIPELINE_COMMAND,
            },
        )
    try:
        _ = service.metadata
    except ServiceNotReadyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "anomaly_model_unavailable",
                "detail": str(exc),
                "hint": exc.hint,
            },
        ) from exc
    return service


def provide_maintenance_service() -> MaintenanceDecisionService:
    """Provide the shared unified decision service."""
    return get_maintenance_service()


def provide_ready_maintenance_service(
    service: Annotated[MaintenanceDecisionService, Depends(provide_maintenance_service)],
) -> MaintenanceDecisionService:
    """Require at least one component model while permitting partial coverage."""
    if not bool(service.status()["model_loaded"]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "all_models_unavailable",
                "detail": "No component model is available for maintenance assessment.",
                "hint": "python scripts/run_all_pipelines.py",
            },
        )
    return service


ConfigDep = Annotated[Config, Depends(provide_config)]
ServiceDep = Annotated[FailurePredictionService, Depends(provide_service)]
ReadyServiceDep = Annotated[FailurePredictionService, Depends(provide_ready_service)]
HealthServiceDep = Annotated[HealthMonitoringService, Depends(provide_health_service)]
ReadyHealthServiceDep = Annotated[HealthMonitoringService, Depends(provide_ready_health_service)]
AnomalyServiceDep = Annotated[AnomalyDetectionService, Depends(provide_anomaly_service)]
ReadyAnomalyServiceDep = Annotated[AnomalyDetectionService, Depends(provide_ready_anomaly_service)]
MaintenanceServiceDep = Annotated[MaintenanceDecisionService, Depends(provide_maintenance_service)]
ReadyMaintenanceServiceDep = Annotated[
    MaintenanceDecisionService, Depends(provide_ready_maintenance_service)
]
