"""Failure Prediction Module HTTP router.

Mounted at ``/failure``.  Later modules add their own routers under their own
prefixes (``/health-monitoring``, ``/anomaly``) and register them in
:mod:`wind_turbine_pm.api.main`; none of them need to touch this file.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, status

from wind_turbine_pm.api.dependencies import ConfigDep, ReadyServiceDep
from wind_turbine_pm.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ExampleResponse,
    FeatureListResponse,
    MetricsResponse,
    ModelInfoResponse,
    PreparedPredictRequest,
    WindowPredictRequest,
)
from wind_turbine_pm.contracts.predictions import FailurePrediction
from wind_turbine_pm.data.ingestion import is_synthetic
from wind_turbine_pm.logging_config import get_logger
from wind_turbine_pm.models.persistence import try_load_metrics
from wind_turbine_pm.models.prediction import FeatureContractError
from wind_turbine_pm.services.failure_prediction_service import InsufficientHistoryError

logger = get_logger(__name__)

router = APIRouter(prefix="/failure", tags=["failure-prediction"])


@router.get("/model-info", response_model=ModelInfoResponse, summary="Published model metadata")
def model_info(service: ReadyServiceDep) -> ModelInfoResponse:
    """Return metadata describing the currently published model.

    Args:
        service: The ready prediction service.

    Returns:
        The model metadata.
    """
    metadata = service.metadata
    return ModelInfoResponse(
        model_name=metadata.model_name,
        model_version=metadata.model_version,
        algorithm=metadata.algorithm,
        training_date=metadata.training_date,
        target=metadata.target,
        horizon_hours=metadata.horizon_hours,
        n_features=metadata.n_features,
        threshold=metadata.threshold.value,
        threshold_method=metadata.threshold.method,
        risk_levels={
            "low_max": metadata.risk_levels.low_max,
            "medium_max": metadata.risk_levels.medium_max,
        },
        dataset=metadata.dataset.model_dump(mode="json"),
        is_synthetic=metadata.dataset.is_synthetic,
        selection_rationale=metadata.selection_rationale,
        library_versions=metadata.library_versions,
    )


@router.get("/metrics", response_model=MetricsResponse, summary="Evaluation metrics")
def metrics(service: ReadyServiceDep, cfg: ConfigDep) -> MetricsResponse:
    """Return the persisted evaluation metrics.

    Args:
        service: The ready prediction service.
        cfg: Merged configuration.

    Returns:
        The metrics document.

    Raises:
        HTTPException: ``404`` when no metrics artifact exists.
    """
    document = try_load_metrics(cfg)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "metrics_unavailable",
                "detail": "No metrics artifact has been written.",
                "hint": "python scripts/evaluate_failure_model.py",
            },
        )
    return MetricsResponse(
        model_version=service.model_version,
        threshold=service.threshold,
        metrics=document,
        is_synthetic=is_synthetic(cfg),
    )


@router.get("/features", response_model=FeatureListResponse, summary="Model feature contract")
def features(service: ReadyServiceDep) -> FeatureListResponse:
    """Return the model's feature names, in order, grouped by family.

    Args:
        service: The ready prediction service.

    Returns:
        The feature contract.
    """
    metadata = service.metadata
    return FeatureListResponse(
        n_features=metadata.n_features,
        features=list(metadata.features),
        feature_groups=metadata.feature_groups,
    )


@router.get("/example", response_model=ExampleResponse, summary="Ready-to-post example payloads")
def example(service: ReadyServiceDep) -> ExampleResponse:
    """Return valid example request bodies for both prediction modes.

    Args:
        service: The ready prediction service.

    Returns:
        Example payloads plus the minimum required history.
    """
    min_hours = service.minimum_history_hours()
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    n_points = int(min_hours) + 1
    observations = [
        {
            "turbine_id": "T01",
            "timestamp": (now - timedelta(hours=n_points - 1 - index)).isoformat(),
            "wind_speed": 8.4,
            "rotor_speed": 12.1,
            "generator_speed": 1174.0,
            "power_output": 720.0,
            "generator_temperature": 62.0,
            "gearbox_temperature": 55.0,
            "bearing_temperature": 48.0,
            "oil_temperature": 46.0,
            "oil_pressure": 5.1,
            "vibration": 3.2,
            "ambient_temperature": 11.0,
            "nacelle_temperature": 19.0,
            "hydraulic_pressure": 188.0,
            "brake_temperature": 17.0,
            "operational_status": "normal",
        }
        for index in range(n_points)
    ]
    return ExampleResponse(
        window_request={"turbine_id": "T01", "observations": observations},
        prepared_request={
            "turbine_id": "T01",
            "timestamp": now.isoformat(),
            "features": service.example_feature_payload(),
        },
        min_history_hours=min_hours,
    )


@router.post(
    "/predict",
    response_model=FailurePrediction,
    summary="Predict from a window of raw observations (preferred)",
)
def predict(request: WindowPredictRequest, service: ReadyServiceDep) -> FailurePrediction:
    """Predict failure risk for the last observation in a raw history window.

    Features are computed inside the service with the same code path used during
    training, so callers never reproduce feature engineering themselves.

    Args:
        request: The window request.
        service: The ready prediction service.

    Returns:
        The failure prediction for the final observation.

    Raises:
        HTTPException: ``422`` when the window is too short or internally
            inconsistent.
    """
    try:
        window = request.to_window()
        result = service.predict_from_window(window)
    except InsufficientHistoryError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "insufficient_history",
                "detail": str(exc),
                "hint": f"Provide at least {service.minimum_history_hours():.0f} hours of observations.",
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "invalid_window", "detail": str(exc), "hint": None},
        ) from exc

    logger.info(
        "Served prediction",
        extra={
            "turbine_id": result.turbine_id,
            "probability": result.failure_probability,
            "risk_level": str(result.risk_level),
        },
    )
    return result


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Predict for several turbines at once",
)
def predict_batch(request: BatchPredictRequest, service: ReadyServiceDep) -> BatchPredictResponse:
    """Predict failure risk for a batch of turbine windows.

    Args:
        request: The batch request.
        service: The ready prediction service.

    Returns:
        One prediction per window.

    Raises:
        HTTPException: ``422`` when any window is invalid or too short.
    """
    try:
        result = service.predict_batch_from_windows([item.to_window() for item in request.windows])
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

    logger.info("Served batch prediction", extra={"count": result.count})
    return BatchPredictResponse(
        predictions=result.predictions,
        count=result.count,
        model_version=result.model_version,
        threshold=result.threshold,
    )


@router.post(
    "/predict/prepared",
    response_model=FailurePrediction,
    summary="Predict from a precomputed feature vector (fallback)",
)
def predict_prepared(
    request: PreparedPredictRequest, service: ReadyServiceDep
) -> FailurePrediction:
    """Predict from features that already match the saved schema.

    Args:
        request: The prepared-feature request.
        service: The ready prediction service.

    Returns:
        The failure prediction.

    Raises:
        HTTPException: ``422`` when the feature vector does not satisfy the
            model's contract.
    """
    try:
        return service.predict_from_features(
            features=request.features, turbine_id=request.turbine_id, timestamp=request.timestamp
        )
    except FeatureContractError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "feature_contract_violation",
                "detail": str(exc),
                "hint": "Fetch the required feature list from GET /failure/features.",
            },
        ) from exc


def router_description() -> dict[str, Any]:
    """Describe this router for the platform's module registry.

    Future modules expose the same helper so the API can advertise which
    modules are mounted without hard-coding them.

    Returns:
        A small descriptor document.
    """
    return {
        "module": "failure_prediction",
        "prefix": router.prefix,
        "status": "available",
        "endpoints": sorted({route.path for route in router.routes}),
    }
