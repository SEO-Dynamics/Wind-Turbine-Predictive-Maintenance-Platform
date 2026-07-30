"""HTTP surface for unified risk and deterministic maintenance actions."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from wind_turbine_pm.anomaly.config import get_anomaly_config
from wind_turbine_pm.api.dependencies import (
    MaintenanceServiceDep,
    ReadyMaintenanceServiceDep,
)
from wind_turbine_pm.api.examples import example_window
from wind_turbine_pm.api.schemas import (
    BatchPredictRequest,
    MaintenanceExampleResponse,
    MaintenancePolicyResponse,
    WindowPredictRequest,
)
from wind_turbine_pm.contracts.maintenance import (
    BatchUnifiedRiskAssessment,
    UnifiedRiskAssessment,
)
from wind_turbine_pm.services.failure_prediction_service import (
    InsufficientHistoryError,
    ServiceNotReadyError,
)

router = APIRouter(prefix="/maintenance", tags=["maintenance-decision-support"])


def _minimum_history(service: MaintenanceServiceDep) -> float:
    requirements = [
        component.minimum_history_hours()
        for component in (service.failure, service.health, service.anomaly)
        if component.is_ready
    ]
    return max(requirements, default=72.0)


@router.get("/policy", response_model=MaintenancePolicyResponse)
def policy(service: MaintenanceServiceDep) -> MaintenancePolicyResponse:
    """Return weights, guardrails and deterministic action windows."""
    return MaintenancePolicyResponse.model_validate(service.policy())


@router.get("/example", response_model=MaintenanceExampleResponse)
def example(service: MaintenanceServiceDep) -> MaintenanceExampleResponse:
    """Return a request covering the longest currently available model history."""
    minimum = _minimum_history(service)
    return MaintenanceExampleResponse(
        window_request=example_window(minimum, cfg=get_anomaly_config()),
        minimum_history_hours=minimum,
        note=(
            "The same raw window is sent to every available component model. "
            "Missing artifacts are reported through coverage and missing_modules."
        ),
    )


@router.post("/assess", response_model=UnifiedRiskAssessment)
def assess(
    request: WindowPredictRequest, service: ReadyMaintenanceServiceDep
) -> UnifiedRiskAssessment:
    """Combine all available evidence without hiding component outputs."""
    try:
        return service.assess(request.to_window())
    except InsufficientHistoryError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "insufficient_history",
                "detail": str(exc),
                "hint": f"Provide at least {_minimum_history(service):.0f} hours of observations.",
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "invalid_window", "detail": str(exc), "hint": None},
        ) from exc
    except ServiceNotReadyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "all_models_unavailable", "detail": str(exc), "hint": exc.hint},
        ) from exc


@router.post("/assess/batch", response_model=BatchUnifiedRiskAssessment)
def assess_batch(
    request: BatchPredictRequest, service: ReadyMaintenanceServiceDep
) -> BatchUnifiedRiskAssessment:
    """Return a fleet batch ordered by deterministic maintenance priority."""
    try:
        return service.assess_batch([item.to_window() for item in request.windows])
    except InsufficientHistoryError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "insufficient_history", "detail": str(exc), "hint": None},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "invalid_batch", "detail": str(exc), "hint": None},
        ) from exc


def router_description() -> dict[str, Any]:
    """Describe this router for the root module registry."""
    return {
        "module": "unified_maintenance_decision",
        "prefix": router.prefix,
        "status": "available",
        "endpoints": sorted({route.path for route in router.routes}),
    }
