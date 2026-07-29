"""Sensor validation rules: physical limits and operating envelopes.

Two different questions are asked about every reading, and conflating them is
how condition monitoring goes wrong:

1. **Is this measurement believable?**  A gearbox at 999 degC is an instrument
   fault, not a hot gearbox.  Hard limits (``hard_min`` / ``hard_max``) mirror
   ``configs/data.yaml -> validation.physical_ranges``, which the shared
   preprocessing layer already uses to null out impossible values.
2. **Is this machine running inside its healthy envelope?**  75 degC in the
   gearbox is perfectly measurable and still means the machine is running hot.
   Envelope limits (``warn_above`` / ``alarm_above`` / ``warn_below`` /
   ``alarm_below``) answer that, and they are what feeds the health score.

Rules are data, not code: they live in ``configs/health_model.yaml ->
health.sensor_rules.rules`` with their provenance (``source``) and reasoning
(``rationale``) attached, so an operator can review and adjust them without
touching Python.  :func:`load_sensor_rules` is the only reader.

Two failure modes that pure range checks miss are covered explicitly:

* **Frozen signals.**  A sensor stuck at a plausible value passes every range
  check forever.  :func:`evaluate_rules` flags a channel whose value moves less
  than ``stuck_tolerance`` across ``stuck_window_hours``.
* **Impossible slew.**  A reading that jumps faster than the physics allows
  (``max_rate_per_hour``) is a spike or a dropout, not a real transient.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from wind_turbine_pm.config import Config, ConfigError
from wind_turbine_pm.constants import TIMESTAMP, TURBINE_ID, DriftSeverity
from wind_turbine_pm.data.preprocessing import hours_to_steps, infer_step_hours
from wind_turbine_pm.logging_config import get_logger

logger = get_logger(__name__)

#: Direction in which a sensor degrades.
RuleDirection = Literal["high_is_bad", "low_is_bad"]

#: Provenance tags a rule may declare.
KNOWN_SOURCES: frozenset[str] = frozenset(
    {"expert_judgement", "industry_standard", "data_analysis"}
)


@dataclass(frozen=True)
class SensorRule:
    """Validation and envelope rules for one sensor channel."""

    sensor: str
    component: str
    unit: str
    hard_min: float
    hard_max: float
    direction: RuleDirection
    source: str
    rationale: str
    warn_above: float | None = None
    alarm_above: float | None = None
    warn_below: float | None = None
    alarm_below: float | None = None
    max_rate_per_hour: float | None = None
    stuck_tolerance: float | None = None

    def __post_init__(self) -> None:
        """Validate the rule's internal consistency.

        Raises:
            ConfigError: If limits are ordered impossibly or the declared
                direction has no matching envelope limits.
        """
        if self.hard_min >= self.hard_max:
            raise ConfigError(
                f"Sensor rule {self.sensor!r}: hard_min ({self.hard_min}) must be below "
                f"hard_max ({self.hard_max})"
            )
        if self.direction == "high_is_bad":
            if self.warn_above is None or self.alarm_above is None:
                raise ConfigError(
                    f"Sensor rule {self.sensor!r} is high_is_bad but declares no "
                    f"warn_above/alarm_above limits"
                )
            if not self.hard_min < self.warn_above < self.alarm_above <= self.hard_max:
                raise ConfigError(
                    f"Sensor rule {self.sensor!r}: expected hard_min < warn_above < "
                    f"alarm_above <= hard_max, got {self.hard_min} / {self.warn_above} / "
                    f"{self.alarm_above} / {self.hard_max}"
                )
        else:
            if self.warn_below is None or self.alarm_below is None:
                raise ConfigError(
                    f"Sensor rule {self.sensor!r} is low_is_bad but declares no "
                    f"warn_below/alarm_below limits"
                )
            if not self.hard_min <= self.alarm_below < self.warn_below < self.hard_max:
                raise ConfigError(
                    f"Sensor rule {self.sensor!r}: expected hard_min <= alarm_below < "
                    f"warn_below < hard_max, got {self.hard_min} / {self.alarm_below} / "
                    f"{self.warn_below} / {self.hard_max}"
                )
        if self.source not in KNOWN_SOURCES:
            raise ConfigError(
                f"Sensor rule {self.sensor!r} declares unknown source {self.source!r}; "
                f"expected one of {sorted(KNOWN_SOURCES)}"
            )

    @property
    def warning_limit(self) -> float:
        """The warning limit in the direction the sensor degrades."""
        return float(self.warn_above if self.direction == "high_is_bad" else self.warn_below)

    @property
    def alarm_limit(self) -> float:
        """The alarm limit in the direction the sensor degrades."""
        return float(self.alarm_above if self.direction == "high_is_bad" else self.alarm_below)

    def exceedance(self, values: pd.Series) -> pd.Series:
        """Signed distance past the warning limit, normalised by the alarm band.

        ``0`` means "at or better than the warning limit", ``1`` means "at the
        alarm limit", and values above 1 mean the alarm limit is exceeded.  The
        normalisation makes channels in different units directly comparable,
        which is what lets a component score combine a temperature and a
        pressure without inventing weights.

        Args:
            values: Raw readings for this sensor.

        Returns:
            The normalised exceedance, floored at zero.
        """
        numeric = pd.to_numeric(values, errors="coerce").astype(float)
        band = abs(self.alarm_limit - self.warning_limit)
        if band <= 0:  # pragma: no cover - prevented by __post_init__
            band = 1.0
        if self.direction == "high_is_bad":
            return ((numeric - self.warning_limit) / band).clip(lower=0.0)
        return ((self.warning_limit - numeric) / band).clip(lower=0.0)

    def severity_of(self, value: float) -> DriftSeverity:
        """Classify a single reading against the envelope limits.

        The warning band is entered strictly past the limit and the alarm band
        at or past it, which is exactly the banding
        :meth:`exceedance` produces (``> 0`` and ``>= 1``).  Keeping the two in
        step matters: the same reading must not be a warning per one method and
        healthy per the other.

        Args:
            value: The reading.

        Returns:
            ``alarm``, ``warning`` or ``none``.
        """
        if not np.isfinite(value):
            return DriftSeverity.NONE
        if self.direction == "high_is_bad":
            if value >= self.alarm_limit:
                return DriftSeverity.ALARM
            return DriftSeverity.WARNING if value > self.warning_limit else DriftSeverity.NONE
        if value <= self.alarm_limit:
            return DriftSeverity.ALARM
        return DriftSeverity.WARNING if value < self.warning_limit else DriftSeverity.NONE

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "sensor": self.sensor,
            "component": self.component,
            "unit": self.unit,
            "hard_min": self.hard_min,
            "hard_max": self.hard_max,
            "direction": self.direction,
            "warn_above": self.warn_above,
            "alarm_above": self.alarm_above,
            "warn_below": self.warn_below,
            "alarm_below": self.alarm_below,
            "max_rate_per_hour": self.max_rate_per_hour,
            "stuck_tolerance": self.stuck_tolerance,
            "source": self.source,
            "rationale": self.rationale.strip(),
        }


@dataclass(frozen=True)
class RuleEvaluation:
    """Outcome of evaluating the rule set against a frame."""

    #: One column per sensor and check, aligned to the input index.
    flags: pd.DataFrame
    #: Normalised exceedance per sensor, aligned to the input index.
    exceedance: pd.DataFrame
    #: Per-sensor counts of each check that fired.
    counts: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def any_flag(self) -> pd.Series:
        """``True`` for rows where at least one rule fired."""
        if self.flags.empty:
            return pd.Series(False, index=self.exceedance.index)
        return self.flags.any(axis=1)

    def data_quality(self) -> float:
        """Share of rows that passed every *validity* check.

        Envelope exceedances are excluded: a hot gearbox is a real measurement
        of an unhealthy machine, not a data-quality problem.

        Returns:
            A fraction in ``[0, 1]``; ``1.0`` for an empty frame.
        """
        validity = [
            c for c in self.flags.columns if c.endswith(("__out_of_range", "__stuck", "__rate"))
        ]
        if not validity or self.flags.empty:
            return 1.0
        return float(1.0 - self.flags[validity].any(axis=1).mean())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "n_rows": int(len(self.exceedance)),
            "data_quality": round(self.data_quality(), 6),
            "counts": self.counts,
        }


def load_sensor_rules(cfg: Config) -> dict[str, SensorRule]:
    """Build the rule set from configuration.

    Args:
        cfg: Merged configuration carrying the ``health`` namespace.

    Returns:
        A mapping of sensor name to its :class:`SensorRule`, in configuration
        order.

    Raises:
        ConfigError: If the rules block is missing or a rule is inconsistent.
    """
    raw = cfg.get("health.sensor_rules.rules")
    if raw is None:
        raise ConfigError(
            "health.sensor_rules.rules is missing. Load the health configuration with "
            "wind_turbine_pm.health.config.load_health_config()."
        )
    rules: dict[str, SensorRule] = {}
    for sensor, body in dict(raw).items():
        payload: Mapping[str, Any] = dict(body)
        rules[str(sensor)] = SensorRule(
            sensor=str(sensor),
            component=str(payload["component"]),
            unit=str(payload.get("unit", "")),
            hard_min=float(payload["hard_min"]),
            hard_max=float(payload["hard_max"]),
            direction=str(payload["direction"]),  # type: ignore[arg-type]
            source=str(payload["source"]),
            rationale=str(payload.get("rationale", "")),
            warn_above=_optional_float(payload.get("warn_above")),
            alarm_above=_optional_float(payload.get("alarm_above")),
            warn_below=_optional_float(payload.get("warn_below")),
            alarm_below=_optional_float(payload.get("alarm_below")),
            max_rate_per_hour=_optional_float(payload.get("max_rate_per_hour")),
            stuck_tolerance=_optional_float(payload.get("stuck_tolerance")),
        )
    return rules


def _optional_float(value: Any) -> float | None:
    """Coerce an optional configuration value to ``float``."""
    return None if value is None else float(value)


def verify_against_physical_ranges(cfg: Config) -> list[str]:
    """Check that hard limits agree with the shared physical ranges.

    The health rules restate the platform's physical ranges so they can be read
    on their own.  Restating a value is a chance to disagree with it, so this
    function exists to be called from a test and from the preparation script.

    Args:
        cfg: Merged configuration.

    Returns:
        One message per disagreement; empty when everything matches.
    """
    ranges = dict(cfg.get("validation.physical_ranges") or {})
    problems: list[str] = []
    for sensor, rule in load_sensor_rules(cfg).items():
        bounds = ranges.get(sensor)
        if bounds is None:
            problems.append(f"{sensor}: no entry in validation.physical_ranges")
            continue
        low, high = float(bounds[0]), float(bounds[1])
        if (rule.hard_min, rule.hard_max) != (low, high):
            problems.append(
                f"{sensor}: health rule hard limits [{rule.hard_min}, {rule.hard_max}] disagree "
                f"with validation.physical_ranges [{low}, {high}]"
            )
    return problems


def evaluate_rules(
    frame: pd.DataFrame, cfg: Config, rules: Mapping[str, SensorRule] | None = None
) -> RuleEvaluation:
    """Evaluate every rule against a preprocessed frame.

    All temporal checks (stuck, rate-of-change) are grouped by turbine and use
    trailing windows only, so the result can be used as a model feature without
    leaking future information.

    Args:
        frame: Frame with ``turbine_id``, ``timestamp`` and sensor columns.
        cfg: Merged configuration.
        rules: Pre-loaded rules; loaded from ``cfg`` when omitted.

    Returns:
        The :class:`RuleEvaluation`.

    Raises:
        ValueError: If the key columns are missing.
    """
    for column in (TURBINE_ID, TIMESTAMP):
        if column not in frame.columns:
            raise ValueError(f"Rule evaluation requires column {column!r}")

    active = dict(rules) if rules is not None else load_sensor_rules(cfg)
    groups = frame[TURBINE_ID]
    step_hours = infer_step_hours(frame) if len(frame) > 1 else 1.0
    stuck_steps = hours_to_steps(
        float(cfg.get("health.sensor_rules.stuck_window_hours", 6)), step_hours
    )

    flags: dict[str, pd.Series] = {}
    exceedance: dict[str, pd.Series] = {}
    counts: dict[str, dict[str, int]] = {}

    for sensor, rule in active.items():
        if sensor not in frame.columns:
            continue
        values = pd.to_numeric(frame[sensor], errors="coerce").astype(float)
        sensor_counts: dict[str, int] = {}

        out_of_range = (values < rule.hard_min) | (values > rule.hard_max)
        flags[f"{sensor}__out_of_range"] = out_of_range.fillna(False)
        sensor_counts["out_of_range"] = int(out_of_range.sum())

        severity_warning = rule.exceedance(values) > 0.0
        severity_alarm = rule.exceedance(values) >= 1.0
        flags[f"{sensor}__warning"] = severity_warning.fillna(False)
        flags[f"{sensor}__alarm"] = severity_alarm.fillna(False)
        sensor_counts["warning"] = int(severity_warning.sum())
        sensor_counts["alarm"] = int(severity_alarm.sum())

        if rule.stuck_tolerance is not None and stuck_steps > 1:
            grouped = values.groupby(groups, sort=False, observed=True)
            spread = grouped.rolling(window=stuck_steps, min_periods=stuck_steps).apply(
                lambda window: float(np.nanmax(window) - np.nanmin(window)), raw=True
            )
            spread = spread.reset_index(level=0, drop=True).reindex(values.index)
            stuck = (spread <= rule.stuck_tolerance).fillna(False)
            flags[f"{sensor}__stuck"] = stuck
            sensor_counts["stuck"] = int(stuck.sum())

        if rule.max_rate_per_hour is not None:
            change = values.groupby(groups, sort=False, observed=True).diff().abs()
            rate = change / max(step_hours, 1e-6)
            too_fast = (rate > rule.max_rate_per_hour).fillna(False)
            flags[f"{sensor}__rate"] = too_fast
            sensor_counts["rate"] = int(too_fast.sum())

        exceedance[sensor] = rule.exceedance(values).fillna(0.0)
        counts[sensor] = sensor_counts

    evaluation = RuleEvaluation(
        flags=pd.DataFrame(flags, index=frame.index),
        exceedance=pd.DataFrame(exceedance, index=frame.index),
        counts=counts,
    )
    logger.info(
        "Evaluated sensor rules",
        extra={
            "rows": len(frame),
            "sensors": len(counts),
            "data_quality": round(evaluation.data_quality(), 4),
        },
    )
    return evaluation


def rules_to_records(rules: Mapping[str, SensorRule]) -> list[dict[str, Any]]:
    """Flatten the rule set for the API and the dashboard table.

    Args:
        rules: The loaded rule set.

    Returns:
        One JSON-serialisable record per rule.
    """
    return [rule.to_dict() for rule in rules.values()]
