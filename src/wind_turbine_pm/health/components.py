"""Component-level health roll-up.

A single fleet number tells an operator that something is wrong; it does not
tell them where to send an engineer.  This module attributes the health score to
named components - gearbox, drivetrain, generator, lubrication, hydraulic,
brake - so an assessment can say *which* part of the machine is driving it.

The component score is deliberately **rule- and baseline-driven rather than
model-driven**.  Two reasons:

1. It is auditable.  Every deduction traces to either "this channel is inside
   its alarm band" or "this channel has moved N sigma from this turbine's own
   baseline", both of which an engineer can check against the raw trend.
2. It degrades honestly.  A regression model trained on a fleet-level target
   cannot be decomposed into per-component contributions without inventing an
   attribution that the training objective never constrained.

The two pieces of evidence are combined by taking the worst, not by averaging:
a component with one channel deep into its alarm band is not healthy because its
other channels are fine.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import COMPONENT_DISPLAY_NAMES, SENSOR_DISPLAY_NAMES
from wind_turbine_pm.contracts.health import ComponentHealth
from wind_turbine_pm.health.health_class import ClassBands, classify_health
from wind_turbine_pm.health.sensor_rules import SensorRule

#: Robust z-score magnitude treated as a fully degraded channel.  Four sigma
#: against a turbine's *own* past-only baseline is a large, sustained move, not
#: ordinary operating variation.
_Z_SATURATION = 4.0


def _bad_direction_z(rule: SensorRule, z_score: float) -> float:
    """Return the deviation magnitude, counting only the degrading direction.

    A gearbox running *colder* than its baseline is not a gearbox problem, so a
    negative deviation on a ``high_is_bad`` channel contributes nothing.

    Args:
        rule: The sensor's rule, carrying its degradation direction.
        z_score: Robust z-score against the turbine's own baseline.

    Returns:
        The non-negative deviation in the bad direction.
    """
    if not np.isfinite(z_score):
        return 0.0
    signed = z_score if rule.direction == "high_is_bad" else -z_score
    return max(float(signed), 0.0)


def component_evidence(
    observation: Mapping[str, float],
    z_scores: Mapping[str, float],
    rules: Mapping[str, SensorRule],
    sensors: list[str],
) -> tuple[float, list[str]]:
    """Score one component's evidence of degradation.

    Args:
        observation: Latest raw sensor values.
        z_scores: Robust z-scores against the turbine's own baseline.
        rules: The loaded sensor rules.
        sensors: Sensors belonging to this component.

    Returns:
        ``(evidence, drivers)`` where ``evidence`` is ``0`` (healthy) to ``1``
        (fully degraded) and ``drivers`` are human-readable phrases naming what
        produced it, worst first.
    """
    worst = 0.0
    drivers: list[tuple[float, str]] = []

    for sensor in sensors:
        rule = rules.get(sensor)
        if rule is None:
            continue
        label = SENSOR_DISPLAY_NAMES.get(sensor, sensor.replace("_", " "))

        value = float(observation.get(sensor, np.nan))
        exceedance = 0.0
        if np.isfinite(value):
            exceedance = float(rule.exceedance(pd.Series([value])).iloc[0])
            if exceedance > 0:
                limit = rule.alarm_limit if exceedance >= 1.0 else rule.warning_limit
                band = "alarm" if exceedance >= 1.0 else "warning"
                comparison = "above" if rule.direction == "high_is_bad" else "below"
                drivers.append(
                    (
                        exceedance,
                        f"{label} at {value:.1f} {rule.unit} is {comparison} the {band} "
                        f"limit of {limit:.1f} {rule.unit}",
                    )
                )

        deviation = _bad_direction_z(rule, float(z_scores.get(sensor, np.nan)))
        normalised_deviation = deviation / _Z_SATURATION
        if normalised_deviation > 0.25:
            drivers.append(
                (
                    normalised_deviation,
                    f"{label} is {deviation:.1f} sigma from this turbine's own baseline",
                )
            )

        worst = max(worst, exceedance, normalised_deviation)

    ordered = [text for _, text in sorted(drivers, key=lambda item: -item[0])]
    return float(min(worst, 1.0)), ordered


def build_component_health(
    observation: Mapping[str, float],
    z_scores: Mapping[str, float],
    rules: Mapping[str, SensorRule],
    cfg: Config,
    bands: ClassBands | None = None,
) -> list[ComponentHealth]:
    """Build the component roll-up for one observation.

    Args:
        observation: Latest raw sensor values for the turbine.
        z_scores: Robust z-scores against the turbine's own baseline, keyed by
            sensor.
        rules: The loaded sensor rules.
        cfg: Merged configuration.
        bands: Pre-resolved class bands; read from ``cfg`` when omitted.

    Returns:
        One :class:`~wind_turbine_pm.contracts.health.ComponentHealth` per
        configured component, worst first.
    """
    configured = dict(cfg.get("health.components", {}) or {})
    resolved = bands if bands is not None else ClassBands.from_config(cfg)

    results: list[ComponentHealth] = []
    for component, sensors in configured.items():
        present = [sensor for sensor in sensors if sensor in rules]
        if not present:
            continue
        evidence, drivers = component_evidence(observation, z_scores, rules, present)
        score = float(np.clip(100.0 * (1.0 - evidence), 0.0, 100.0))
        results.append(
            ComponentHealth(
                component=str(component),
                score=round(score, 2),
                health_class=classify_health(score, cfg, resolved),
                sensors=present,
                drivers=drivers[:3],
            )
        )
    return sorted(results, key=lambda item: item.score)


def worst_component(components: list[ComponentHealth]) -> ComponentHealth | None:
    """Return the lowest-scoring component, if any.

    Args:
        components: The component roll-up.

    Returns:
        The worst component, or ``None`` for an empty list.
    """
    return min(components, key=lambda item: item.score) if components else None


def component_label(component: str) -> str:
    """Return the human-readable name of a component.

    Args:
        component: Component key.

    Returns:
        The display label, falling back to the key itself.
    """
    return COMPONENT_DISPLAY_NAMES.get(component, component.replace("_", " "))
