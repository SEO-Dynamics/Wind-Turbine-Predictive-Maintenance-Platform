"""FastAPI dependency providers.

The service is resolved lazily through
:func:`~wind_turbine_pm.services.failure_prediction_service.get_service`, whose
``lru_cache`` guarantees one loaded model per process.  Importing this module
does not read artifacts and never triggers training, so the API starts cleanly
even when the model has not been built yet - ``/health`` then reports
``degraded`` with the command to fix it.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, status

from wind_turbine_pm.config import Config, get_config
from wind_turbine_pm.services.failure_prediction_service import (
    FailurePredictionService,
    ServiceNotReadyError,
    get_service,
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


ConfigDep = Annotated[Config, Depends(provide_config)]
ServiceDep = Annotated[FailurePredictionService, Depends(provide_service)]
ReadyServiceDep = Annotated[FailurePredictionService, Depends(provide_ready_service)]
