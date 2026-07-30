"""Shared example request generation for the ``/example`` endpoints.

Examples are drawn from the **same synthetic generator the models were trained
on**, not from a hand-written curve.

This matters more than it looks. An earlier version emitted a noiseless
sinusoid in which every sensor was an exact affine function of one sine wave.
That window is physically plausible but statistically unlike real SCADA: it has
zero measurement noise and a rank-1 correlation structure. The novelty detector
- correctly - scored it at the 99.9th percentile and returned ``alarm``, which
then tripped the maintenance guardrail and reported a turbine with a 100/100
health score and a 0.2% failure probability as *high risk, urgent review*.

Sampling the real generator instead keeps the examples in-distribution: they
score ``normal`` on the anomaly model and ``healthy`` on the health model, so
the ``/example`` payloads demonstrate the platform behaving sensibly.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import (
    OPERATIONAL_STATUS,
    SENSOR_COLUMNS,
    TIMESTAMP,
    TURBINE_ID,
    OperationalStatus,
)
from wind_turbine_pm.logging_config import get_logger

logger = get_logger(__name__)

#: Seed for the example window. Fixed so ``/example`` is deterministic.
_EXAMPLE_SEED = 20240101


def example_window(
    min_history_hours: float,
    turbine_id: str = "T01",
    cfg: Config | None = None,
) -> dict[str, Any]:
    """Build a representative hourly SCADA window for every model endpoint.

    Args:
        min_history_hours: Minimum history the endpoint requires; the window is
            made comfortably longer so the 48-hour rolling features are defined.
        turbine_id: Identifier stamped on the returned observations.
        cfg: Configuration carrying a ``synthetic`` section. When omitted or
            unusable, a smooth analytical fallback is returned instead.

    Returns:
        A ``{"turbine_id": ..., "observations": [...]}`` payload that is valid
        for ``/failure/predict``, ``/health-monitoring/assess``,
        ``/anomaly/detect`` and ``/maintenance/assess``.
    """
    n_points = int(min_history_hours) + 25
    if cfg is not None:
        window = _window_from_generator(n_points, turbine_id, cfg)
        if window is not None:
            return window
    return _smooth_fallback_window(n_points, turbine_id)


def _window_from_generator(n_points: int, turbine_id: str, cfg: Config) -> dict[str, Any] | None:
    """Sample a healthy window from the project's synthetic SCADA generator.

    Returns ``None`` when the generator cannot be used - for example when
    ``data.source`` points at a real export and no ``synthetic`` section is
    configured - so the caller can fall back.
    """
    try:
        import numpy as np

        from wind_turbine_pm.data.preprocessing import preprocess
        from wind_turbine_pm.data.synthetic import generate_scada_dataset
    except ImportError:  # pragma: no cover - defensive
        return None

    if cfg.get("synthetic") is None:
        return None

    overrides = cfg.to_dict()
    # One turbine, a couple of months, no injected defects: enough history to
    # sample a clean window from, and fast enough to build per request (~10ms).
    overrides["synthetic"].update(
        {
            "n_turbines": 1,
            "months": 2,
            "missing_rate": 0.0,
            "invalid_rate": 0.0,
            "duplicate_rate": 0.0,
            "seed": _EXAMPLE_SEED,
        }
    )

    try:
        generated = generate_scada_dataset(Config(overrides), inject_issues=False)
        clean, _ = preprocess(generated, Config(overrides))
    except (KeyError, ValueError) as exc:
        logger.warning("Example generator unavailable, using fallback", extra={"error": str(exc)})
        return None

    normal = clean.loc[clean[OPERATIONAL_STATUS].astype(str) == str(OperationalStatus.NORMAL)]
    # Take a window that ends clear of the record tail, so it sits inside a
    # settled operating period rather than mid-transition.
    candidate = normal.tail(n_points * 3).head(n_points)
    if len(candidate) < n_points:
        candidate = normal.tail(n_points)
    if len(candidate) < 2:
        return None

    # Re-stamp onto a recent, contiguous hourly axis so the payload reads as
    # "the last N hours" regardless of the generator's own calendar.
    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    observations: list[dict[str, Any]] = []
    for offset, record in enumerate(candidate.to_dict(orient="records")):
        payload: dict[str, Any] = {
            TURBINE_ID: turbine_id,
            TIMESTAMP: (end - timedelta(hours=len(candidate) - 1 - offset)).isoformat(),
        }
        for column in SENSOR_COLUMNS:
            value = record.get(column)
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                payload[column] = None
            else:
                payload[column] = round(float(value), 3)
        payload[OPERATIONAL_STATUS] = str(record.get(OPERATIONAL_STATUS, OperationalStatus.NORMAL))
        observations.append(payload)

    return {"turbine_id": turbine_id, "observations": observations}


def _smooth_fallback_window(n_points: int, turbine_id: str) -> dict[str, Any]:
    """Analytical fallback used only when the generator is unavailable.

    Physically plausible but statistically idealised: no measurement noise and
    a single shared driver. The anomaly detector may score it as novel, which is
    a property of the payload, not a defect in the detector.
    """
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    observations: list[dict[str, Any]] = []
    for index in range(n_points):
        phase = 2.0 * math.pi * (index % 24) / 24.0
        swing = math.sin(phase)
        wind = 8.4 + 2.2 * swing
        load = max(0.0, min(1.0, (wind - 3.0) / 9.5))
        observations.append(
            {
                "turbine_id": turbine_id,
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
    return {"turbine_id": turbine_id, "observations": observations}
