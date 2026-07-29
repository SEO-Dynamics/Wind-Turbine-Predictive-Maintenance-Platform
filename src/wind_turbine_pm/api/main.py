"""FastAPI application for the Wind Turbine Predictive Maintenance Platform.

The Failure Prediction and Turbine Health Monitoring modules are mounted today.
Future modules register themselves in :data:`MODULE_ROUTERS` - see
``docs/OZAN_HANDOFF.md``.

Importing this module must stay free of side effects: it does not load model
artifacts and never trains.  Artifacts are loaded lazily on the first request
that needs them, which is why ``/health`` can report a missing model instead of
the process refusing to start - and why one module's artifacts being absent does
not stop the other from serving.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from wind_turbine_pm import __version__
from wind_turbine_pm.api.dependencies import HealthServiceDep, ServiceDep
from wind_turbine_pm.api.routers import failure, health_monitoring
from wind_turbine_pm.api.schemas import HealthResponse
from wind_turbine_pm.config import get_config
from wind_turbine_pm.logging_config import configure_from_config, get_logger
from wind_turbine_pm.services.failure_prediction_service import ServiceNotReadyError

logger = get_logger(__name__)

#: Routers mounted on the application. Future modules append their own router
#: here; nothing else in this file needs to change.
MODULE_ROUTERS: list[tuple[APIRouter, Callable[[], dict[str, Any]]]] = [
    (failure.router, failure.router_description),
    (health_monitoring.router, health_monitoring.router_description),
]

DESCRIPTION = """
Decision-support API for wind-turbine predictive maintenance.

**Failure Prediction Module** - estimates the probability that a turbine will
experience a failure within the next 48 hours, from SCADA sensor history.

Two prediction modes are available:

* `POST /failure/predict` - **preferred**. Post a window of raw observations;
  the service computes the model features itself using the same code path as
  training.
* `POST /failure/predict/prepared` - fallback. Post a feature vector that
  already matches the schema returned by `GET /failure/features`.

**Turbine Health Monitoring Module** - scores a turbine's current condition
0-100, bands it into Healthy / Monitor / Degraded / Critical, attributes it to
named components, and reports sensor drift and operating-envelope violations.

* `POST /health-monitoring/assess` - assess one turbine from a window of raw
  observations. The same body shape as `POST /failure/predict`.
* `POST /health-monitoring/assess/batch` - assess several turbines at once.
* `GET /health-monitoring/sensor-rules` - the operating-envelope rules, with the
  provenance of every limit.

The two modules answer different questions and are deliberately independent:
failure prediction asks *"is something about to break?"*, health monitoring asks
*"what condition is this machine in, and is it getting worse?"*. Either can serve
while the other's artifacts are missing; `GET /health` reports both.

**All output is advisory.** It is decision support for qualified maintenance
staff. It is not a certified industrial safety system, is not a substitute for
engineering inspection, and must never be wired to automatic control of plant.
Default results are produced on synthetic data.
"""


def create_app() -> FastAPI:
    """Build and configure the FastAPI application.

    Returns:
        The configured application.
    """
    cfg = get_config()
    configure_from_config(cfg)

    application = FastAPI(
        title="Wind Turbine Predictive Maintenance API",
        description=DESCRIPTION,
        version=__version__,
        contact={"name": "Failure Prediction Module"},
        license_info={"name": "MIT"},
        openapi_tags=[
            {"name": "system", "description": "Health and platform information."},
            {
                "name": "failure-prediction",
                "description": "48-hour failure risk prediction, explanation and advisory output.",
            },
            {
                "name": "turbine-health-monitoring",
                "description": (
                    "Condition scoring, health classification, component roll-up and sensor "
                    "drift detection."
                ),
            },
        ],
    )

    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=["system"],
        summary="Liveness and per-module model status",
    )
    def health(service: ServiceDep, health_service: HealthServiceDep) -> HealthResponse:
        """Report API liveness and each module's artifact availability.

        Returns ``200`` in every case; ``status`` distinguishes a fully working
        deployment from one where at least one module's artifacts have not been
        built yet, and ``modules`` says which.  The modules are independent, so a
        missing health model does not prevent failure prediction from serving.

        Args:
            service: The shared failure-prediction service.
            health_service: The shared health-monitoring service.

        Returns:
            The health document.
        """
        failure_info = service.status()
        health_info = health_service.status()
        modules = {
            "failure_prediction": failure_info.to_dict(),
            "turbine_health_monitoring": health_info.to_dict(),
        }
        loaded = [name for name, info in modules.items() if info["model_loaded"]]
        missing = [name for name in modules if name not in loaded]

        if missing:
            detail = (
                f"Loaded: {', '.join(loaded) if loaded else 'none'}. "
                f"Missing artifacts: {', '.join(missing)}. "
                + " ".join(modules[name]["detail"] for name in missing if modules[name]["detail"])
            )
        else:
            detail = "All module artifacts loaded."

        return HealthResponse(
            status="ok" if not missing else "degraded",
            api_version=__version__,
            model_loaded=failure_info.model_loaded,
            model_version=failure_info.model_version,
            model_artifact_status="loaded" if failure_info.model_loaded else "missing",
            timestamp=datetime.now(UTC),
            detail=detail,
            modules=modules,
            runtime_warnings=[
                *failure_info.runtime_warnings,
                *health_info.runtime_warnings,
            ],
        )

    @application.get("/", tags=["system"], summary="Platform and module overview")
    def root() -> dict[str, Any]:
        """List the platform version and the modules currently mounted.

        Returns:
            A descriptor document including the planned future modules.
        """
        return {
            "platform": "Wind Turbine Predictive Maintenance Platform",
            "version": __version__,
            "docs": "/docs",
            "modules": [describe() for _, describe in MODULE_ROUTERS],
            "planned_modules": [
                {"module": "anomaly_detection", "status": "planned"},
                {"module": "unified_maintenance_decision", "status": "planned"},
            ],
            "advisory_only": True,
        }

    for module_router, _ in MODULE_ROUTERS:
        application.include_router(module_router)

    @application.exception_handler(ServiceNotReadyError)
    async def service_not_ready_handler(
        request: Request, exc: ServiceNotReadyError
    ) -> JSONResponse:
        """Turn a mid-request artifact loss into a 503 rather than a 500.

        The dependency layer already rejects unready requests; this is the
        safety net for artifacts that disappear between the check and the call.
        """
        logger.error("Service not ready", extra={"path": request.url.path, "error": str(exc)})
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": "model_unavailable", "detail": str(exc), "hint": exc.hint},
        )

    @application.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Return a structured body for request-schema violations."""
        logger.warning("Request validation failed", extra={"path": request.url.path})
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "validation_error",
                "detail": "The request body does not satisfy the API schema.",
                "hint": "See GET /failure/example for a valid payload.",
                "errors": [
                    {"location": list(error["loc"]), "message": error["msg"]}
                    for error in exc.errors()[:20]
                ],
            },
        )

    logger.info(
        "API initialised",
        extra={"version": __version__, "modules": [r.prefix for r, _ in MODULE_ROUTERS]},
    )
    return application


app = create_app()
