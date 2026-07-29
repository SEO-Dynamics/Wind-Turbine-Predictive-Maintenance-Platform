"""Health classification: turning a 0-100 score into an operational decision.

The score is what the model produces; the class is what an operator acts on.
Keeping them separate matters, because the boundaries are an *operational*
choice - how much degradation justifies scheduling an inspection - and not a
property of the model.  They live in ``configs/health_model.yaml ->
health.classes`` and can be re-tuned without retraining anything.

Bands (defaults):

===========  ==============  =====================================================
Class        Score           Operational meaning
===========  ==============  =====================================================
healthy      >= 80           No action beyond routine monitoring.
monitor      60 - 80         Watch the trend; review at the next planned visit.
degraded     40 - 60         Schedule an inspection; the machine is off-baseline.
critical     < 40            Prioritise inspection before continued operation.
===========  ==============  =====================================================

Two mechanisms modify the mapping, both deliberately bounded:

* **Drift penalty.**  Detected sensor drift subtracts points from the raw score
  (:func:`wind_turbine_pm.health.drift.drift_penalty`), capped so drift alone can
  never manufacture a Critical.  Drift means the measurement or the baseline is
  suspect; that justifies less confidence, not an assertion of failure.
* **Adaptive boundaries.**  Optionally the bands shift with the fleet's own score
  distribution, so a fleet-wide change (a firmware update, a season) does not
  silently move every turbine into Monitor.  Off by default - a moving boundary
  is harder to explain to an operator than a fixed one - and clamped by
  ``max_shift`` so the bands can never invert.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import HEALTH_CLASS_ORDER, HealthClass
from wind_turbine_pm.contracts.health import HealthClassBands
from wind_turbine_pm.logging_config import get_logger

logger = get_logger(__name__)

#: Advisory action attached to each class by :func:`recommended_action`.
_ACTIONS: dict[HealthClass, str] = {
    HealthClass.HEALTHY: (
        "No action required beyond routine monitoring. Continue normal operation."
    ),
    HealthClass.MONITOR: (
        "Keep under observation. Review the trend at the next scheduled visit; no "
        "unplanned intervention is indicated."
    ),
    HealthClass.DEGRADED: (
        "Schedule an inspection. The machine is operating measurably away from its own "
        "healthy baseline and the condition indicators are trending the wrong way."
    ),
    HealthClass.CRITICAL: (
        "Prioritise inspection before continued operation. Condition indicators are deep "
        "into the degraded range; confirm on site before deciding to derate or stop."
    ),
}


@dataclass(frozen=True)
class ClassBands:
    """Resolved score boundaries for the four health classes."""

    healthy_min: float
    monitor_min: float
    degraded_min: float

    @classmethod
    def from_config(cls, cfg: Config) -> ClassBands:
        """Build the bands from configuration.

        Args:
            cfg: Merged configuration carrying the ``health`` namespace.

        Returns:
            The configured :class:`ClassBands`.
        """
        return cls(
            healthy_min=float(cfg.get("health.classes.healthy_min", 80.0)),
            monitor_min=float(cfg.get("health.classes.monitor_min", 60.0)),
            degraded_min=float(cfg.get("health.classes.degraded_min", 40.0)),
        )

    def to_contract(self) -> HealthClassBands:
        """Return the Pydantic contract form, validating the ordering."""
        return HealthClassBands(
            healthy_min=self.healthy_min,
            monitor_min=self.monitor_min,
            degraded_min=self.degraded_min,
        )

    def shifted(self, offset: float) -> ClassBands:
        """Return the bands moved by ``offset`` points.

        Args:
            offset: Points to add to every boundary.

        Returns:
            The shifted bands, clamped into ``[0, 100]``.
        """
        return ClassBands(
            healthy_min=float(np.clip(self.healthy_min + offset, 1.0, 99.0)),
            monitor_min=float(np.clip(self.monitor_min + offset, 0.5, 98.0)),
            degraded_min=float(np.clip(self.degraded_min + offset, 0.0, 97.0)),
        )

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serialisable representation."""
        return {
            "healthy_min": self.healthy_min,
            "monitor_min": self.monitor_min,
            "degraded_min": self.degraded_min,
        }


def classify_health(score: float, cfg: Config, bands: ClassBands | None = None) -> HealthClass:
    """Map a health score onto its class.

    Args:
        score: Health score in ``[0, 100]``.
        cfg: Merged configuration.
        bands: Pre-resolved bands; read from ``cfg`` when omitted.

    Returns:
        The :class:`~wind_turbine_pm.constants.HealthClass`.

    Raises:
        ValueError: If the resolved bands are not strictly ordered.
    """
    resolved = bands if bands is not None else ClassBands.from_config(cfg)
    resolved.to_contract()  # validates degraded_min < monitor_min < healthy_min

    if score >= resolved.healthy_min:
        return HealthClass.HEALTHY
    if score >= resolved.monitor_min:
        return HealthClass.MONITOR
    if score >= resolved.degraded_min:
        return HealthClass.DEGRADED
    return HealthClass.CRITICAL


def classify_health_series(
    scores: pd.Series, cfg: Config, bands: ClassBands | None = None
) -> pd.Series:
    """Vectorised :func:`classify_health` for a whole column of scores.

    Args:
        scores: Health scores in ``[0, 100]``.
        cfg: Merged configuration.
        bands: Pre-resolved bands; read from ``cfg`` when omitted.

    Returns:
        A string series of health-class labels aligned to ``scores``.
    """
    resolved = bands if bands is not None else ClassBands.from_config(cfg)
    resolved.to_contract()
    values = pd.to_numeric(scores, errors="coerce").astype(float)
    return pd.Series(
        np.select(
            [
                values >= resolved.healthy_min,
                values >= resolved.monitor_min,
                values >= resolved.degraded_min,
            ],
            [
                str(HealthClass.HEALTHY),
                str(HealthClass.MONITOR),
                str(HealthClass.DEGRADED),
            ],
            default=str(HealthClass.CRITICAL),
        ),
        index=scores.index,
        dtype=object,
    )


def adaptive_bands(scores: pd.Series, cfg: Config) -> ClassBands:
    """Shift the class boundaries with the fleet's own score distribution.

    The shift is the gap between the fleet's reference quantile and the midpoint
    of the Monitor band, clamped to ``health.classes.adaptive.max_shift``.  When
    adaptation is disabled - the default - the configured bands are returned
    unchanged.

    Args:
        scores: Recent fleet health scores.
        cfg: Merged configuration.

    Returns:
        The bands to apply.
    """
    bands = ClassBands.from_config(cfg)
    if not bool(cfg.get("health.classes.adaptive.enabled", False)) or scores.empty:
        return bands

    quantile = float(cfg.get("health.classes.adaptive.reference_quantile", 0.5))
    max_shift = float(cfg.get("health.classes.adaptive.max_shift", 10.0))
    reference = float(pd.to_numeric(scores, errors="coerce").dropna().quantile(quantile))
    midpoint = (bands.healthy_min + bands.monitor_min) / 2.0
    offset = float(np.clip(reference - midpoint, -max_shift, max_shift))

    shifted = bands.shifted(offset)
    logger.info(
        "Adapted health class bands to the fleet distribution",
        extra={"offset": round(offset, 3), "bands": shifted.to_dict()},
    )
    return shifted


def apply_drift_penalty(raw_score: float, penalty: float) -> float:
    """Apply the drift deduction to a raw score.

    Args:
        raw_score: The model's own 0-100 estimate.
        penalty: Points to deduct, ``>= 0``.

    Returns:
        The published score, clamped into ``[0, 100]``.
    """
    return float(np.clip(raw_score - max(penalty, 0.0), 0.0, 100.0))


def severity_rank(health_class: HealthClass | str) -> int:
    """Return the severity rank of a class, ``0`` being healthiest.

    Args:
        health_class: The class or its string value.

    Returns:
        Index into :data:`~wind_turbine_pm.constants.HEALTH_CLASS_ORDER`.

    Raises:
        ValueError: If the class is not recognised.
    """
    value = HealthClass(str(health_class))
    return HEALTH_CLASS_ORDER.index(value)


def recommended_action(health_class: HealthClass) -> str:
    """Return the standing advisory action for a class.

    Args:
        health_class: The classified health band.

    Returns:
        The advisory sentence, without the platform disclaimer (callers append
        :data:`~wind_turbine_pm.constants.ADVISORY_DISCLAIMER`).
    """
    return _ACTIONS[health_class]


def class_distribution(classes: pd.Series) -> dict[str, int]:
    """Count turbines per health class in the canonical order.

    Args:
        classes: A column of health-class labels.

    Returns:
        A mapping of class name to count, always containing all four classes.
    """
    counts = classes.astype(str).value_counts()
    return {str(name): int(counts.get(str(name), 0)) for name in HEALTH_CLASS_ORDER}
