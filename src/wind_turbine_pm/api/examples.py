"""Shared, physically plausible example request generation."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any


def example_window(min_history_hours: float, turbine_id: str = "T01") -> dict[str, Any]:
    """Build a smooth hourly SCADA window suitable for every model endpoint."""
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    n_points = int(min_history_hours) + 25
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
