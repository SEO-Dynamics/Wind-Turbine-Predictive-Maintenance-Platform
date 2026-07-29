"""Turbine Health Monitoring HTTP router.

Mounted at ``/health-monitoring``.  The prefix is deliberately not ``/health``:
that path is the platform's liveness probe and is used by Docker's healthcheck
and by CI, so taking it for a module would break both.

The module exposes a single input mode - a window of raw observations - because
a health assessment reports component scores, rule violations and drift signals
that are derived from the raw window rather than from a feature vector.  There
is therefore no prepared-features equivalent of
``POST /failure/predict/prepared``: a caller posting only features could not be
given a complete assessment, and silently returning a partial one would be
worse than not offering the endpoint.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, status

from wind_turbine_pm.api.dependencies import ConfigDep, ReadyHealthServiceDep
from wind_turbine_pm.api.schemas import (
    BatchHealthRequest,
    HealthExampleResponse,
    HealthMetricsResponse,
    HealthModelInfoResponse,
    HealthWindowRequest,
    SensorRulesResponse,
)
from wind_turbine_pm.contracts.health import BatchHealthAssessment, HealthAssessment
from wind_turbine_pm.data.ingestion import is_synthetic
from wind_turbine_pm.health.persistence import try_load_health_metrics
from wind_turbine_pm.logging_config import get_logger
from wind_turbine_pm.services.failure_prediction_service import InsufficientHistoryError

logger = get_logger(__name__)

router = APIRouter(prefix="/health-monitoring", tags=["turbine-health-monitoring"])


@router.get(
    "/model-info",
    response_model=HealthModelInfoResponse,
    summary="Published health model metadata",
)
def model_info(service: ReadyHealthServiceDep) -> HealthModelInfoResponse:
    """Return metadata describing the currently published health model.

    Args:
        service: The ready health service.

    Returns:
        The health model metadata.
    """
    metadata = service.metadata
    return HealthModelInfoResponse(
        model_name=metadata.model_name,
        model_version=metadata.model_version,
        algorithm=metadata.algorithm,
        training_date=metadata.training_date,
        target=metadata.target,
        target_source=metadata.target_source,
        n_features=metadata.n_features,
        health_classes=metadata.health_classes.model_dump(mode="json"),
        drift=metadata.drift,
        dataset=metadata.dataset.model_dump(mode="json"),
        is_synthetic=metadata.dataset.is_synthetic,
        selection_rationale=metadata.selection_rationale,
        library_versions=metadata.library_versions,
    )


@router.get("/metrics", response_model=HealthMetricsResponse, summary="Health model metrics")
def metrics(service: ReadyHealthServiceDep, cfg: ConfigDep) -> HealthMetricsResponse:
    """Return the persisted health metrics document.

    Args:
        service: The ready health service.
        cfg: Merged configuration.

    Returns:
        The metrics document.

    Raises:
        HTTPException: ``404`` when no health metrics artifact exists.
    """
    document = try_load_health_metrics(service.config)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "metrics_unavailable",
                "detail": "No health metrics artifact has been written.",
                "hint": "python scripts/train_health_model.py",
            },
        )
    return HealthMetricsResponse(
        model_version=service.model_version,
        metrics=document,
        is_synthetic=is_synthetic(cfg),
    )


@router.get(
    "/sensor-rules",
    response_model=SensorRulesResponse,
    summary="Sensor validation and operating-envelope rules",
)
def sensor_rules(service: ReadyHealthServiceDep) -> SensorRulesResponse:
    """Return the configured sensor rules with their provenance.

    Exposed because the rules are the auditable part of the assessment: an
    operator reviewing a component score needs to see which limit was applied and
    on what basis it was chosen.

    Args:
        service: The ready health service.

    Returns:
        The rule set.
    """
    records = service.sensor_rule_records()
    return SensorRulesResponse(n_rules=len(records), rules=records)


@router.get(
    "/example", response_model=HealthExampleResponse, summary="Ready-to-post example payload"
)
def example(service: ReadyHealthServiceDep) -> HealthExampleResponse:
    """Return a valid example request body for the assessment endpoint.

    Args:
        service: The ready health service.

    Returns:
        An example payload plus the minimum required history.
    """
    min_hours = service.minimum_history_hours()
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    # A margin above the minimum, so the example still validates if a caller
    # trims a few observations from it.
    n_points = int(min_hours) + 25

    # The values follow a smooth diurnal cycle rather than being constant.
    # A constant channel is exactly what the frozen-sensor rule is designed to
    # catch, so a flat example would come back reporting ten rule violations and
    # near-zero data quality - a correct assessment of an implausible payload,
    # and a thoroughly misleading illustration of the endpoint.
    observations = []
    for index in range(n_points):
        phase = 2.0 * math.pi * (index % 24) / 24.0
        swing = math.sin(phase)
        wind = 8.4 + 2.2 * swing
        load = max(0.0, min(1.0, (wind - 3.0) / 9.5))
        observations.append(
            {
                "turbine_id": "T01",
                "timestamp": (now - timedelta(hours=n_points - 1 - index)).isoformat(),
                "wind_speed": round(wind, 2),
                "rotor_speed": round(9.5 + 3.4 * load, 2),
                "generator_speed": round(880.0 + 420.0 * load, 1),
                "power_output": round(120.0 + 1350.0 * load, 1),
                "generator_temperature": round(52.0 + 16.0 * load, 2),
                "gearbox_temperature": round(46.0 + 14.0 * load, 2),
                "bearing_temperature": round(41.0 + 11.0 * load, 2),
                "oil_temperature": round(40.0 + 10.0 * load, 2),
                "oil_pressure": round(5.4 - 0.45 * load, 3),
                "vibration": round(2.4 + 1.3 * load, 3),
                "ambient_temperature": round(11.0 + 4.0 * swing, 2),
                "nacelle_temperature": round(18.0 + 5.0 * load, 2),
                "hydraulic_pressure": round(191.0 - 6.0 * load, 1),
                "brake_temperature": round(16.5 + 2.0 * load, 2),
                "operational_status": "normal",
            }
        )
    return HealthExampleResponse(
        window_request={"turbine_id": "T01", "observations": observations},
        min_history_hours=min_hours,
    )


@router.post(
    "/assess",
    response_model=HealthAssessment,
    summary="Assess one turbine's health from a window of raw observations",
)
def assess(request: HealthWindowRequest, service: ReadyHealthServiceDep) -> HealthAssessment:
    """Assess turbine health for the last observation in a raw history window.

    Features are computed inside the service with the same code path used during
    training, so callers never reproduce feature engineering themselves.

    Args:
        request: The window request.
        service: The ready health service.

    Returns:
        The health assessment for the final observation.

    Raises:
        HTTPException: ``422`` when the window is too short or internally
            inconsistent.
    """
    try:
        result = service.assess_from_window(request.to_window())
    except InsufficientHistoryError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "insufficient_history",
                "detail": str(exc),
                "hint": (
                    f"Provide at least {service.minimum_history_hours():.0f} hours of observations."
                ),
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "invalid_window", "detail": str(exc), "hint": None},
        ) from exc

    logger.info(
        "Served health assessment",
        extra={
            "turbine_id": result.turbine_id,
            "health_score": result.health_score,
            "health_class": str(result.health_class),
        },
    )
    return result


@router.post(
    "/assess/batch",
    response_model=BatchHealthAssessment,
    summary="Assess several turbines at once",
)
def assess_batch(
    request: BatchHealthRequest, service: ReadyHealthServiceDep
) -> BatchHealthAssessment:
    """Assess health for a batch of turbine windows.

    Args:
        request: The batch request.
        service: The ready health service.

    Returns:
        One assessment per window.

    Raises:
        HTTPException: ``422`` when any window is invalid or too short.
    """
    try:
        result = service.assess_batch_from_windows([item.to_window() for item in request.windows])
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

    logger.info("Served batch health assessment", extra={"count": result.count})
    return result


def router_description() -> dict[str, Any]:
    """Describe this router for the platform's module registry.

    Returns:
        A small descriptor document.
    """
    return {
        "module": "turbine_health_monitoring",
        "prefix": router.prefix,
        "status": "available",
        "endpoints": sorted({route.path for route in router.routes}),
    }
