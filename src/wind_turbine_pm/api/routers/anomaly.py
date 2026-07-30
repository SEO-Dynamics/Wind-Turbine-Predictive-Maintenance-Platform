"""HTTP surface for calibrated novelty detection."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from wind_turbine_pm.api.dependencies import ReadyAnomalyServiceDep
from wind_turbine_pm.api.examples import example_window
from wind_turbine_pm.api.schemas import (
    AnomalyExampleResponse,
    AnomalyFeatureListResponse,
    AnomalyMetricsResponse,
    AnomalyModelInfoResponse,
    BatchPredictRequest,
    WindowPredictRequest,
)
from wind_turbine_pm.constants import GROUND_TRUTH_COLUMNS
from wind_turbine_pm.contracts.anomaly import (
    AnomalyPrediction,
    BatchAnomalyPrediction,
)
from wind_turbine_pm.logging_config import get_logger
from wind_turbine_pm.services.failure_prediction_service import InsufficientHistoryError

logger = get_logger(__name__)
router = APIRouter(prefix="/anomaly", tags=["anomaly-detection"])


@router.get("/model-info", response_model=AnomalyModelInfoResponse)
def model_info(service: ReadyAnomalyServiceDep) -> AnomalyModelInfoResponse:
    """Return the selected detector and its calibration contract."""
    metadata = service.metadata
    calibration = service.calibration
    return AnomalyModelInfoResponse(
        model_name=metadata.model_name,
        model_version=metadata.model_version,
        algorithm=metadata.algorithm,
        training_date=metadata.training_date,
        n_features=metadata.n_features,
        warning_threshold=calibration.warning_percentile,
        alarm_threshold=calibration.alarm_percentile,
        dataset=metadata.dataset,
        selection_rationale=metadata.selection_rationale,
        library_versions=metadata.library_versions,
    )


@router.get("/metrics", response_model=AnomalyMetricsResponse)
def metrics(service: ReadyAnomalyServiceDep) -> AnomalyMetricsResponse:
    """Return candidate comparison, test metrics and measured healthy alert rates."""
    from wind_turbine_pm.anomaly.persistence import try_load_metrics

    document = try_load_metrics(service.config)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "metrics_unavailable",
                "detail": "No anomaly metrics artifact has been written.",
                "hint": "python scripts/train_anomaly_model.py",
            },
        )
    calibration = document["calibration"]
    return AnomalyMetricsResponse(
        model_version=service.model_version,
        metrics=document,
        warning_healthy_alert_rate=calibration["achieved_warning_rate"],
        alarm_healthy_alert_rate=calibration["achieved_alarm_rate"],
        is_synthetic=bool(document["is_synthetic"]),
    )


@router.get("/features", response_model=AnomalyFeatureListResponse)
def features(service: ReadyAnomalyServiceDep) -> AnomalyFeatureListResponse:
    """Publish exact feature order and explicit leakage exclusions."""
    metadata = service.metadata
    configured = list(service.config.require("anomaly.features.exclude_columns"))
    return AnomalyFeatureListResponse(
        n_features=metadata.n_features,
        features=metadata.features,
        feature_groups=metadata.feature_groups,
        excluded_truth_columns=sorted(set(configured) | set(GROUND_TRUTH_COLUMNS)),
        max_lookback_hours=int(service.config.require("anomaly.features.max_history_hours")),
    )


@router.get("/example", response_model=AnomalyExampleResponse)
def example(service: ReadyAnomalyServiceDep) -> AnomalyExampleResponse:
    """Return a ready-to-post raw SCADA window."""
    minimum = service.minimum_history_hours()
    return AnomalyExampleResponse(
        window_request=example_window(minimum),
        min_history_hours=minimum,
    )


@router.post("/detect", response_model=AnomalyPrediction)
def detect(request: WindowPredictRequest, service: ReadyAnomalyServiceDep) -> AnomalyPrediction:
    """Detect novelty at the final observation in a window."""
    try:
        result = service.detect_from_window(request.to_window())
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
        "Served anomaly assessment",
        extra={"turbine_id": result.turbine_id, "severity": result.severity},
    )
    return result


@router.post("/detect/batch", response_model=BatchAnomalyPrediction)
def detect_batch(
    request: BatchPredictRequest, service: ReadyAnomalyServiceDep
) -> BatchAnomalyPrediction:
    """Detect novelty for several independent turbine windows."""
    try:
        return service.detect_batch_from_windows([item.to_window() for item in request.windows])
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
        "module": "anomaly_detection",
        "prefix": router.prefix,
        "status": "available",
        "endpoints": sorted({route.path for route in router.routes}),
    }
