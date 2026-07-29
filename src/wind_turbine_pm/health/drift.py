"""Sensor drift detection.

Drift here means: *this channel has moved away from what this turbine's own
history says it should read, and stayed there*.  That is a different question
from "is this value outside its limits" (:mod:`sensor_rules`) and from "is this
machine degrading" (:mod:`health_score`) - a drifting sensor can sit inside
every limit while quietly making the health score wrong, which is exactly why
it gets its own detector and its own penalty.

Three complementary detectors run on every configured channel:

``cusum``
    Cumulative sum of standardised residuals with slack ``k``.  Detects small,
    persistent shifts - the classic slow-drift signature - that a threshold on
    the raw value never catches, because the statistic accumulates evidence
    across many samples instead of judging each one.

``ewma``
    Exponentially weighted moving average of the same residuals.  Reacts faster
    than CUSUM to a moderate step change and is less sensitive to the choice of
    slack.

``isolation_forest``
    Multivariate novelty over all channels at once.  Catches drift that shows up
    only as a broken *relationship* between channels (oil pressure falling while
    oil temperature rises) which per-channel statistics cannot see.

Every statistic is computed inside ``groupby(turbine_id)`` over trailing data
only, so drift columns are safe to use as model features.

The residual the detectors run on is a past-only robust z-score against the
turbine's own baseline **restricted to producing regimes**, so a machine that
simply spent a week idle does not register as drifting.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import (
    OPERATING_REGIME,
    PRODUCING_REGIMES,
    TIMESTAMP,
    TURBINE_ID,
    DriftDirection,
    DriftSeverity,
)
from wind_turbine_pm.contracts.health import SensorDriftSignal
from wind_turbine_pm.data.preprocessing import hours_to_steps, infer_step_hours
from wind_turbine_pm.features.transformers import expanding_baseline
from wind_turbine_pm.logging_config import get_logger

logger = get_logger(__name__)

#: Consistency constant making the MAD an unbiased estimator of sigma for
#: normally distributed data.
_MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class DriftSettings:
    """Detector parameters read from configuration."""

    sensors: tuple[str, ...]
    min_periods: int
    cusum_k: float
    cusum_h: float
    cusum_reset: bool
    signal_window_hours: float
    alarm_signals: int
    ewma_lambda: float
    ewma_l_sigma: float
    warning_ratio: float
    alarm_ratio: float
    isolation_enabled: bool
    isolation_threshold: float

    @classmethod
    def from_config(cls, cfg: Config) -> DriftSettings:
        """Build the settings from configuration.

        Args:
            cfg: Merged configuration carrying the ``health`` namespace.

        Returns:
            The configured :class:`DriftSettings`.
        """
        return cls(
            sensors=tuple(cfg.get("health.drift.sensors", []) or ()),
            min_periods=int(cfg.get("health.drift.min_periods", 72)),
            cusum_k=float(cfg.get("health.drift.cusum.k", 0.5)),
            cusum_h=float(cfg.get("health.drift.cusum.h", 5.0)),
            cusum_reset=bool(cfg.get("health.drift.cusum.reset_after_signal", True)),
            signal_window_hours=float(cfg.get("health.drift.cusum.signal_window_hours", 168)),
            alarm_signals=int(cfg.get("health.drift.severity.alarm_signals", 3)),
            ewma_lambda=float(cfg.get("health.drift.ewma.lambda", 0.1)),
            ewma_l_sigma=float(cfg.get("health.drift.ewma.l_sigma", 3.0)),
            warning_ratio=float(cfg.get("health.drift.severity.warning_ratio", 1.0)),
            alarm_ratio=float(cfg.get("health.drift.severity.alarm_ratio", 1.6)),
            isolation_enabled=bool(cfg.get("health.drift.isolation_forest.enabled", True)),
            isolation_threshold=float(
                cfg.get("health.drift.isolation_forest.anomaly_threshold", 0.62)
            ),
        )

    @property
    def ewma_control_limit(self) -> float:
        """Asymptotic EWMA control limit in standard-error units."""
        lam = self.ewma_lambda
        return float(self.ewma_l_sigma * np.sqrt(lam / (2.0 - lam)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "sensors": list(self.sensors),
            "min_periods": self.min_periods,
            "cusum": {"k": self.cusum_k, "h": self.cusum_h},
            "ewma": {
                "lambda": self.ewma_lambda,
                "l_sigma": self.ewma_l_sigma,
                "control_limit": round(self.ewma_control_limit, 6),
            },
            "severity": {"warning_ratio": self.warning_ratio, "alarm_ratio": self.alarm_ratio},
            "isolation_forest": {
                "enabled": self.isolation_enabled,
                "anomaly_threshold": self.isolation_threshold,
            },
        }


@dataclass(frozen=True)
class DriftCalibration:
    """Per-sensor persistence thresholds fitted on the healthy training period.

    Why this exists
    ---------------
    The textbook CUSUM constants (``k=0.5``, ``h=5``) are derived for
    **independent** residuals.  SCADA channels sampled hourly are strongly
    autocorrelated - thermal inertia means this hour's gearbox temperature is
    largely last hour's - so the effective number of independent observations per
    hour is far below one and the in-control run length collapses.  Measured on
    this dataset, channels whose standardised residual is correctly scaled
    (``std ~ 1.0``) still cross the decision limit in 5-8% of rows, and the
    trailing crossing count does not separate healthy machines from degraded ones
    at all: healthy turbines score *higher* on most channels.

    Inheriting the textbook constant therefore produces a detector that fires
    almost everywhere and means nothing.  Instead the thresholds are set from the
    fleet's own healthy population, so "drift detected" is defined as *"this
    channel is crossing its limit more often than a healthy machine on this fleet
    does"* - an empirical statement with a chosen, documented false-alarm rate
    rather than an assumption the data violates.

    The thresholds are fitted on the **training split only** and persisted in the
    model metadata, so scoring can never re-derive them from the data being
    judged.
    """

    warning_at: dict[str, float]
    alarm_at: dict[str, float]
    ewma_warning_at: dict[str, float]
    ewma_alarm_at: dict[str, float]
    target_warning_rate: float
    target_alarm_rate: float
    fitted_rows: int
    #: Multivariate novelty score at which a warning / an alarm is raised. ``None``
    #: when no anomaly scores were supplied at fit time, in which case the
    #: configured ``isolation_forest.anomaly_threshold`` applies.
    multivariate_warning_at: float | None = None
    multivariate_alarm_at: float | None = None

    @classmethod
    def fit(
        cls,
        statistics: pd.DataFrame,
        healthy: pd.Series,
        cfg: Config,
        anomaly_scores: pd.Series | None = None,
    ) -> DriftCalibration | None:
        """Fit the thresholds from the healthy population's own statistics.

        Args:
            statistics: Output of :func:`compute_drift_statistics` restricted to
                the training split.
            healthy: Boolean mask selecting rows the ground truth says are
                healthy, aligned to ``statistics``.
            cfg: Merged configuration.
            anomaly_scores: Multivariate novelty scores for the same rows, from a
                detector fitted on this training split.  When supplied the
                multivariate threshold is calibrated too; otherwise the
                configured constant applies.

        Returns:
            The fitted calibration, or ``None`` when it is disabled or there are
            too few healthy rows to estimate a quantile from.
        """
        if not bool(cfg.get("health.drift.calibration.enabled", True)):
            return None

        warning_rate = float(cfg.get("health.drift.calibration.target_warning_rate", 0.05))
        alarm_rate = float(cfg.get("health.drift.calibration.target_alarm_rate", 0.01))
        minimum_rows = int(cfg.get("health.drift.calibration.min_healthy_rows", 500))

        subset = statistics.loc[healthy.reindex(statistics.index).fillna(False)]
        if len(subset) < minimum_rows:
            logger.warning(
                "Too few healthy rows to calibrate the drift thresholds; the configured "
                "persistence thresholds will be used as-is",
                extra={"rows": len(subset), "required": minimum_rows},
            )
            return None

        settings = DriftSettings.from_config(cfg)
        warning_at: dict[str, float] = {}
        alarm_at: dict[str, float] = {}
        ewma_warning_at: dict[str, float] = {}
        ewma_alarm_at: dict[str, float] = {}

        for sensor in settings.sensors:
            column = f"{sensor}_cusum_signals"
            if column in subset.columns:
                counts = pd.to_numeric(subset[column], errors="coerce").dropna()
                if not counts.empty:
                    # A threshold must still require at least one crossing,
                    # whatever the quantile says, so a channel that never signals
                    # is never reported as drifting.
                    warning_at[sensor] = max(float(counts.quantile(1.0 - warning_rate)), 1.0)
                    alarm_at[sensor] = max(
                        float(counts.quantile(1.0 - alarm_rate)), warning_at[sensor] + 1.0
                    )

            # The EWMA needs the same treatment and for the same reason. Its
            # asymptotic control limit assumes independent residuals; on
            # autocorrelated channels with a slightly biased baseline the
            # smoothed residual sits past that limit almost permanently, which
            # made the EWMA arm - not the CUSUM - the dominant false-alarm source
            # before it was calibrated.
            ewma_column = f"{sensor}_ewma"
            if ewma_column in subset.columns:
                magnitude = pd.to_numeric(subset[ewma_column], errors="coerce").abs().dropna()
                if not magnitude.empty:
                    ewma_warning_at[sensor] = max(
                        float(magnitude.quantile(1.0 - warning_rate)),
                        settings.ewma_control_limit,
                    )
                    ewma_alarm_at[sensor] = max(
                        float(magnitude.quantile(1.0 - alarm_rate)),
                        ewma_warning_at[sensor] * 1.05,
                    )

        # The Isolation Forest score needs the same treatment. Its configured
        # threshold is a constant on a logistic-squashed score whose location is
        # the training median, so it has no defined false-alarm rate; measured
        # here it fired on 43% of healthy observations.
        multivariate_warning_at: float | None = None
        multivariate_alarm_at: float | None = None
        if anomaly_scores is not None:
            healthy_scores = (
                pd.to_numeric(anomaly_scores.reindex(subset.index), errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if not healthy_scores.empty:
                multivariate_warning_at = float(healthy_scores.quantile(1.0 - warning_rate))
                multivariate_alarm_at = max(
                    float(healthy_scores.quantile(1.0 - alarm_rate)),
                    multivariate_warning_at,
                )

        if not warning_at and not ewma_warning_at:
            return None

        calibration = cls(
            warning_at=warning_at,
            alarm_at=alarm_at,
            ewma_warning_at=ewma_warning_at,
            ewma_alarm_at=ewma_alarm_at,
            target_warning_rate=warning_rate,
            target_alarm_rate=alarm_rate,
            fitted_rows=int(len(subset)),
            multivariate_warning_at=multivariate_warning_at,
            multivariate_alarm_at=multivariate_alarm_at,
        )
        logger.info(
            "Calibrated drift thresholds on the healthy training population",
            extra={
                "rows": len(subset),
                "cusum_warning_at": warning_at,
                "ewma_warning_at": {k: round(v, 3) for k, v in ewma_warning_at.items()},
            },
        )
        return calibration

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> DriftCalibration | None:
        """Rebuild a calibration from its persisted form.

        Args:
            payload: The ``calibration`` block of the model metadata's ``drift``
                document, or ``None``.

        Returns:
            The calibration, or ``None`` when the artifact carries none.
        """
        if not payload:
            return None

        def numeric(key: str) -> dict[str, float]:
            return {str(name): float(value) for name, value in dict(payload.get(key) or {}).items()}

        warning_at = numeric("warning_at")
        ewma_warning_at = numeric("ewma_warning_at")
        if not warning_at and not ewma_warning_at:
            return None
        return cls(
            warning_at=warning_at,
            alarm_at=numeric("alarm_at"),
            ewma_warning_at=ewma_warning_at,
            ewma_alarm_at=numeric("ewma_alarm_at"),
            target_warning_rate=float(payload.get("target_warning_rate", 0.05)),
            target_alarm_rate=float(payload.get("target_alarm_rate", 0.01)),
            fitted_rows=int(payload.get("fitted_rows", 0)),
            multivariate_warning_at=(
                None
                if payload.get("multivariate_warning_at") is None
                else float(payload["multivariate_warning_at"])
            ),
            multivariate_alarm_at=(
                None
                if payload.get("multivariate_alarm_at") is None
                else float(payload["multivariate_alarm_at"])
            ),
        )

    def thresholds_for(self, sensor: str, settings: DriftSettings) -> tuple[float, float]:
        """Return the ``(warning_at, alarm_at)`` crossing counts for a sensor.

        Args:
            sensor: Sensor channel name.
            settings: Detector settings, supplying the fallback thresholds.

        Returns:
            The calibrated thresholds, or the configured ones when this sensor
            was not calibrated.
        """
        warning = self.warning_at.get(sensor)
        if warning is None:
            return 1.0, float(settings.alarm_signals)
        return warning, self.alarm_at.get(sensor, warning + 1.0)

    def ewma_thresholds_for(self, sensor: str, settings: DriftSettings) -> tuple[float, float]:
        """Return the ``(warning_at, alarm_at)`` EWMA magnitudes for a sensor.

        Args:
            sensor: Sensor channel name.
            settings: Detector settings, supplying the uncalibrated fallback.

        Returns:
            The calibrated magnitudes, or the asymptotic control limit scaled by
            the configured ratios when this sensor was not calibrated.
        """
        warning = self.ewma_warning_at.get(sensor)
        if warning is None:
            limit = settings.ewma_control_limit
            return limit * settings.warning_ratio, limit * settings.alarm_ratio
        return warning, self.ewma_alarm_at.get(sensor, warning * 1.05)

    def multivariate_thresholds_for(self, settings: DriftSettings) -> tuple[float, float]:
        """Return the ``(warning_at, alarm_at)`` multivariate novelty scores.

        Args:
            settings: Detector settings, supplying the uncalibrated fallback.

        Returns:
            The calibrated scores, or the configured constant (and the 1.25x
            escalation the uncalibrated path used) when none was fitted.
        """
        if self.multivariate_warning_at is None:
            threshold = settings.isolation_threshold
            return threshold, min(threshold * 1.25, 0.99)
        return (
            self.multivariate_warning_at,
            self.multivariate_alarm_at
            if self.multivariate_alarm_at is not None
            else self.multivariate_warning_at,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "warning_at": {key: round(value, 3) for key, value in self.warning_at.items()},
            "alarm_at": {key: round(value, 3) for key, value in self.alarm_at.items()},
            "ewma_warning_at": {
                key: round(value, 4) for key, value in self.ewma_warning_at.items()
            },
            "ewma_alarm_at": {key: round(value, 4) for key, value in self.ewma_alarm_at.items()},
            "multivariate_warning_at": (
                None
                if self.multivariate_warning_at is None
                else round(self.multivariate_warning_at, 6)
            ),
            "multivariate_alarm_at": (
                None if self.multivariate_alarm_at is None else round(self.multivariate_alarm_at, 6)
            ),
            "target_warning_rate": self.target_warning_rate,
            "target_alarm_rate": self.target_alarm_rate,
            "fitted_rows": self.fitted_rows,
        }


def producing_mask(frame: pd.DataFrame) -> pd.Series:
    """Return ``True`` for rows recorded while the turbine was producing.

    Args:
        frame: Frame carrying ``operating_regime``; when the column is absent
            every row counts as producing.

    Returns:
        A boolean series aligned to ``frame``.
    """
    if OPERATING_REGIME not in frame.columns:
        return pd.Series(True, index=frame.index)
    producing = {str(regime) for regime in PRODUCING_REGIMES}
    return frame[OPERATING_REGIME].astype(str).isin(producing)


def regime_conditioned_z(
    series: pd.Series,
    groups: pd.Series,
    mask: pd.Series,
    min_periods: int,
    floor: float = 1e-6,
) -> pd.Series:
    """Past-only robust z-score against a regime-restricted baseline.

    The location and scale are expanding medians over the turbine's *own past*
    rows that satisfy ``mask``; :func:`~wind_turbine_pm.features.transformers.
    expanding_baseline` shifts by one row, so the current observation never
    contributes to the baseline it is judged against.

    Args:
        series: Sensor values.
        groups: Turbine key aligned to ``series``.
        mask: Rows eligible to form the baseline (e.g. producing regimes).
        min_periods: Minimum baseline history before a score is produced.
        floor: Lower bound on the scale, preventing division by zero.

    Returns:
        The robust z-score, NaN until enough baseline history exists.
    """
    eligible = series.where(mask)
    location = expanding_baseline(eligible, groups, "median", min_periods=min_periods)
    deviation = (eligible - location).abs()
    scale = _MAD_TO_SIGMA * expanding_baseline(deviation, groups, "median", min_periods=min_periods)
    return (series - location) / scale.clip(lower=floor)


def cusum_statistics(
    residuals: pd.Series, groups: pd.Series, k: float, h: float | None = None
) -> tuple[pd.Series, pd.Series]:
    """Two-sided tabular CUSUM over standardised residuals, per turbine.

    ``S+_t = max(0, S+_(t-1) + z_t - k)`` and
    ``S-_t = max(0, S-_(t-1) - z_t - k)``.  The slack ``k`` is what makes the
    statistic ignore noise: shifts smaller than ``k`` sigma decay instead of
    accumulating.  Both arms reset to zero at a turbine boundary and treat a
    missing residual as "no evidence" rather than as zero drift.

    Restart after a signal
    ----------------------
    When ``h`` is supplied, an arm that reaches the decision interval is reported
    and then **reset to zero** - Page's original restart scheme.  This is not
    cosmetic.  Without it the statistic is monotonically unbounded in the
    presence of any sustained residual: measured on this dataset the arms reach
    hundreds to thousands of sigma, every row past the first crossing reports an
    alarm forever, and 92% of all observations take the maximum drift penalty.
    A penalty that applies to almost every row is a constant offset, not a
    signal.

    With the restart the statistic stays bounded and a crossing means "drift was
    detected *here*", so persistence becomes a countable quantity - which is what
    :func:`compute_drift_statistics` turns into the trailing signal count that
    severity is graded on.

    Args:
        residuals: Standardised residuals (robust z-scores).
        groups: Turbine key aligned to ``residuals``.
        k: Slack in sigma units.
        h: Decision interval in sigma units.  When ``None`` the arms accumulate
            without restarting (the textbook non-restart form).

    Returns:
        ``(upward, downward)`` cumulative sums, both non-negative.
    """
    values = residuals.to_numpy(dtype=float)
    keys = groups.to_numpy()
    upward = np.zeros(values.size, dtype=float)
    downward = np.zeros(values.size, dtype=float)

    positive = 0.0
    negative = 0.0
    previous_key: Any = None
    for position in range(values.size):
        key = keys[position]
        if key != previous_key:
            positive, negative = 0.0, 0.0
            previous_key = key
        value = values[position]
        if np.isfinite(value):
            positive = max(0.0, positive + value - k)
            negative = max(0.0, negative - value - k)
        # The crossing value is recorded before the restart, so the reported
        # statistic still shows how far past the limit the evidence went.
        upward[position] = positive
        downward[position] = negative
        if h is not None:
            if positive >= h:
                positive = 0.0
            if negative >= h:
                negative = 0.0

    return (
        pd.Series(upward, index=residuals.index),
        pd.Series(downward, index=residuals.index),
    )


def trailing_signal_count(crossings: pd.Series, groups: pd.Series, window: int) -> pd.Series:
    """Count CUSUM crossings in a trailing window, per turbine.

    Persistence is what separates a channel that wandered once from one that is
    genuinely drifting, and with the restart scheme it is simply the number of
    times the arm re-crossed its limit.

    Args:
        crossings: Boolean/0-1 series marking rows where an arm signalled.
        groups: Turbine key aligned to ``crossings``.
        window: Trailing window length in rows.

    Returns:
        The trailing count, aligned to ``crossings``.
    """
    counted = (
        crossings.astype(float)
        .groupby(groups, sort=False, observed=True)
        .rolling(window=window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .reindex(crossings.index)
    )
    return counted.fillna(0.0)


def ewma_statistic(residuals: pd.Series, groups: pd.Series, lam: float) -> pd.Series:
    """Exponentially weighted moving average of residuals, per turbine.

    Args:
        residuals: Standardised residuals.
        groups: Turbine key aligned to ``residuals``.
        lam: Smoothing constant in ``(0, 1]``; smaller reacts more slowly.

    Returns:
        The EWMA statistic, NaN wherever no residual has been observed yet.
    """
    return (
        residuals.groupby(groups, sort=False, observed=True)
        .ewm(alpha=lam, adjust=False, ignore_na=True)
        .mean()
        .reset_index(level=0, drop=True)
        .reindex(residuals.index)
    )


def compute_drift_statistics(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compute every per-sensor drift statistic for a frame.

    Args:
        frame: Preprocessed frame with ``turbine_id`` and, ideally,
            ``operating_regime`` (see :func:`producing_mask`).
        cfg: Merged configuration.

    Returns:
        A frame aligned to ``frame`` with, per configured sensor, the columns
        ``{sensor}_drift_z``, ``{sensor}_cusum_pos``, ``{sensor}_cusum_neg``,
        ``{sensor}_cusum_signals`` (trailing crossing count) and
        ``{sensor}_ewma``.

    Raises:
        ValueError: If ``turbine_id`` is missing.
    """
    if TURBINE_ID not in frame.columns:
        raise ValueError(f"Drift detection requires column {TURBINE_ID!r}")

    settings = DriftSettings.from_config(cfg)
    groups = frame[TURBINE_ID]
    mask = producing_mask(frame)

    step_hours = infer_step_hours(frame) if len(frame) > 1 else 1.0
    signal_window = hours_to_steps(settings.signal_window_hours, step_hours)
    decision = settings.cusum_h if settings.cusum_reset else None

    columns: dict[str, pd.Series] = {}
    for sensor in settings.sensors:
        if sensor not in frame.columns:
            continue
        values = pd.to_numeric(frame[sensor], errors="coerce").astype(float)
        z = regime_conditioned_z(values, groups, mask, settings.min_periods)
        upward, downward = cusum_statistics(z, groups, settings.cusum_k, decision)
        columns[f"{sensor}_drift_z"] = z
        columns[f"{sensor}_cusum_pos"] = upward
        columns[f"{sensor}_cusum_neg"] = downward
        columns[f"{sensor}_cusum_signals"] = trailing_signal_count(
            (upward >= settings.cusum_h) | (downward >= settings.cusum_h), groups, signal_window
        )
        columns[f"{sensor}_ewma"] = ewma_statistic(z, groups, settings.ewma_lambda)

    statistics = pd.DataFrame(columns, index=frame.index)
    logger.info(
        "Computed drift statistics",
        extra={"rows": len(statistics), "sensors": len(settings.sensors)},
    )
    return statistics


def _severity_for(ratio: float, settings: DriftSettings) -> DriftSeverity:
    """Map a statistic/limit ratio onto a severity band."""
    if not np.isfinite(ratio):
        return DriftSeverity.NONE
    if ratio >= settings.alarm_ratio:
        return DriftSeverity.ALARM
    return DriftSeverity.WARNING if ratio >= settings.warning_ratio else DriftSeverity.NONE


def _severity_from_signals(
    count: float,
    sensor: str,
    settings: DriftSettings,
    calibration: DriftCalibration | None = None,
) -> DriftSeverity:
    """Grade CUSUM severity by how often the arm re-crossed its limit.

    With the restart scheme the statistic is bounded, so "how far past the limit"
    no longer distinguishes a transient excursion from a sustained drift -
    persistence does.  The thresholds come from :class:`DriftCalibration` when the
    artifact carries one, so the comparison is against what a *healthy* machine on
    this fleet does rather than against a constant that assumes independent
    residuals.

    Args:
        count: Trailing count of crossings.
        sensor: Sensor channel name, used to look up its calibrated thresholds.
        settings: The detector settings, supplying the uncalibrated fallback.
        calibration: Fitted thresholds, when available.

    Returns:
        ``alarm``, ``warning`` or ``none``.
    """
    if not np.isfinite(count) or count < 1:
        return DriftSeverity.NONE
    if calibration is not None:
        warning_at, alarm_at = calibration.thresholds_for(sensor, settings)
    else:
        warning_at, alarm_at = 1.0, float(settings.alarm_signals)
    if count >= alarm_at:
        return DriftSeverity.ALARM
    return DriftSeverity.WARNING if count >= warning_at else DriftSeverity.NONE


def _first_crossing(
    statistic: pd.Series, limit: float, timestamps: pd.Series | None
) -> pd.Timestamp | None:
    """Return the earliest timestamp at which ``statistic`` exceeded ``limit``."""
    if timestamps is None:
        return None
    crossed = statistic >= limit
    if not bool(crossed.any()):
        return None
    return pd.Timestamp(timestamps.loc[crossed.idxmax()])


def summarise_drift(
    statistics: pd.DataFrame,
    cfg: Config,
    timestamps: pd.Series | None = None,
    anomaly_score: float | None = None,
    calibration: DriftCalibration | None = None,
) -> list[SensorDriftSignal]:
    """Turn drift statistics for one turbine's window into reportable signals.

    Only signals that actually fired are returned; an empty list means "no drift
    detected", which is what the API and dashboard render.

    Args:
        statistics: Output of :func:`compute_drift_statistics` restricted to one
            turbine, ordered by time.  The **last** row is the assessed state.
        cfg: Merged configuration.
        timestamps: Timestamps aligned to ``statistics``, used to report when
            each signal first crossed its limit.
        anomaly_score: Optional multivariate novelty score in ``[0, 1]``.
        calibration: Persistence thresholds fitted on the healthy training
            population.  Strongly recommended: without it the CUSUM thresholds
            fall back to constants that assume independent residuals, which
            SCADA channels are not - see :class:`DriftCalibration`.

    Returns:
        The fired :class:`~wind_turbine_pm.contracts.health.SensorDriftSignal`
        objects, most severe first.
    """
    if statistics.empty:
        return []

    settings = DriftSettings.from_config(cfg)
    latest = statistics.iloc[-1]
    ewma_limit = settings.ewma_control_limit
    signals: list[SensorDriftSignal] = []

    for sensor in settings.sensors:
        upward_column, downward_column = f"{sensor}_cusum_pos", f"{sensor}_cusum_neg"
        if upward_column not in statistics.columns:
            continue

        # Severity comes from persistence over the window, not from the final
        # row's value: with the restart scheme the arm is zero immediately after
        # it signals, so the last observation alone would under-report a channel
        # that has been crossing repeatedly.
        signal_column = f"{sensor}_cusum_signals"
        count = (
            float(latest[signal_column])
            if signal_column in statistics.columns
            else float(
                (
                    (statistics[upward_column] >= settings.cusum_h)
                    | (statistics[downward_column] >= settings.cusum_h)
                ).sum()
            )
        )
        severity = _severity_from_signals(count, sensor, settings, calibration)
        if severity is not DriftSeverity.NONE:
            peak_upward = float(statistics[upward_column].max())
            peak_downward = float(statistics[downward_column].max())
            direction = (
                DriftDirection.UPWARD if peak_upward >= peak_downward else DriftDirection.DOWNWARD
            )
            column = upward_column if direction is DriftDirection.UPWARD else downward_column
            statistic = max(peak_upward, peak_downward)
            signals.append(
                SensorDriftSignal(
                    sensor=sensor,
                    method="cusum",
                    detected=True,
                    severity=severity,
                    direction=direction,
                    statistic=round(statistic, 4),
                    control_limit=settings.cusum_h,
                    first_detected=_first_crossing(
                        statistics[column], settings.cusum_h, timestamps
                    ),
                    description=(
                        f"Cumulative {direction.value} shift in {sensor.replace('_', ' ')} "
                        f"crossed its {settings.cusum_h:.1f} sigma decision limit "
                        f"{int(count)} time(s) in the assessed window, peaking at "
                        f"{statistic:.1f} sigma"
                    ),
                )
            )

        ewma_column = f"{sensor}_ewma"
        ewma_value = float(latest.get(ewma_column, np.nan))
        if np.isfinite(ewma_value):
            if calibration is not None:
                ewma_warning_at, ewma_alarm_at = calibration.ewma_thresholds_for(sensor, settings)
            else:
                ewma_warning_at = ewma_limit * settings.warning_ratio
                ewma_alarm_at = ewma_limit * settings.alarm_ratio

            magnitude = abs(ewma_value)
            if magnitude >= ewma_alarm_at:
                ewma_severity = DriftSeverity.ALARM
            elif magnitude >= ewma_warning_at:
                ewma_severity = DriftSeverity.WARNING
            else:
                ewma_severity = DriftSeverity.NONE

            if ewma_severity is not DriftSeverity.NONE:
                basis = (
                    f"a limit of {ewma_warning_at:.2f}, above which only "
                    f"{calibration.target_warning_rate:.0%} of this fleet's healthy "
                    "observations sit"
                    if calibration is not None
                    else f"a control limit of {ewma_warning_at:.2f} standard errors"
                )
                signals.append(
                    SensorDriftSignal(
                        sensor=sensor,
                        method="ewma",
                        detected=True,
                        severity=ewma_severity,
                        direction=(
                            DriftDirection.UPWARD if ewma_value > 0 else DriftDirection.DOWNWARD
                        ),
                        statistic=round(ewma_value, 4),
                        control_limit=round(ewma_warning_at, 4),
                        first_detected=_first_crossing(
                            statistics[ewma_column].abs(), ewma_warning_at, timestamps
                        ),
                        description=(
                            f"Smoothed {sensor.replace('_', ' ')} residual is "
                            f"{ewma_value:+.2f} against {basis}"
                        ),
                    )
                )

    multivariate_warning_at, multivariate_alarm_at = (
        calibration.multivariate_thresholds_for(settings)
        if calibration is not None
        else (settings.isolation_threshold, min(settings.isolation_threshold * 1.25, 0.99))
    )
    if (
        anomaly_score is not None
        and settings.isolation_enabled
        and np.isfinite(anomaly_score)
        and anomaly_score >= multivariate_warning_at
    ):
        signals.append(
            SensorDriftSignal(
                sensor="multivariate",
                method="isolation_forest",
                detected=True,
                severity=(
                    DriftSeverity.ALARM
                    if anomaly_score >= multivariate_alarm_at
                    else DriftSeverity.WARNING
                ),
                direction=DriftDirection.NONE,
                statistic=round(float(anomaly_score), 4),
                control_limit=round(float(multivariate_warning_at), 6),
                description=(
                    "the combination of sensor residuals is unlike the turbine's own training "
                    "behaviour, even though no single channel is individually out of band"
                ),
            )
        )

    severity_rank = {DriftSeverity.ALARM: 0, DriftSeverity.WARNING: 1, DriftSeverity.NONE: 2}
    return sorted(signals, key=lambda signal: (severity_rank[signal.severity], -signal.statistic))


def drift_penalty(signals: list[SensorDriftSignal], cfg: Config) -> float:
    """Health-score points to deduct for the detected drift.

    Capped by ``health.drift.penalty.max_points`` so drift alone can never
    manufacture a Critical classification: drift says the measurement or the
    baseline is suspect, which justifies lowering confidence in the score, not
    asserting that the machine has failed.

    Args:
        signals: Fired drift signals.
        cfg: Merged configuration.

    Returns:
        Points to subtract from the raw health score, ``>= 0``.
    """
    warning_points = float(cfg.get("health.drift.penalty.per_warning_points", 2.5))
    alarm_points = float(cfg.get("health.drift.penalty.per_alarm_points", 5.0))
    multivariate_points = float(cfg.get("health.drift.penalty.multivariate_points", 5.0))
    maximum = float(cfg.get("health.drift.penalty.max_points", 15.0))

    total = 0.0
    # One deduction per sensor, at its worst severity: CUSUM and EWMA firing on
    # the same channel is one drifting sensor, not two.
    worst: dict[str, DriftSeverity] = {}
    for signal in signals:
        if signal.method == "isolation_forest":
            total += multivariate_points
            continue
        current = worst.get(signal.sensor)
        if current is None or signal.severity is DriftSeverity.ALARM:
            worst[signal.sensor] = signal.severity

    for severity in worst.values():
        total += alarm_points if severity is DriftSeverity.ALARM else warning_points
    return float(min(total, maximum))


class MultivariateDriftDetector:
    """Isolation Forest over the per-sensor drift residuals.

    Fitted on the **training split only**, so "normal" means "what this fleet
    looked like during the training period" rather than "whatever the scored
    data happens to contain".  Scores are mapped to ``[0, 1]`` where higher is
    more anomalous.
    """

    def __init__(self, columns: list[str], model: Any, offset: float, scale: float) -> None:
        """Store the fitted model and its score normalisation.

        Args:
            columns: Residual columns the model consumes, in order.
            model: The fitted ``IsolationForest``.
            offset: Location used to normalise raw scores.
            scale: Spread used to normalise raw scores.
        """
        self.columns = list(columns)
        self.model = model
        self.offset = float(offset)
        self.scale = float(scale) if scale > 1e-9 else 1.0

    @classmethod
    def fit(cls, statistics: pd.DataFrame, cfg: Config) -> MultivariateDriftDetector | None:
        """Fit the detector on drift residuals.

        Args:
            statistics: Output of :func:`compute_drift_statistics`.
            cfg: Merged configuration.

        Returns:
            The fitted detector, or ``None`` when it is disabled or there are no
            usable residual columns.
        """
        if not bool(cfg.get("health.drift.isolation_forest.enabled", True)):
            return None
        columns = [column for column in statistics.columns if column.endswith("_drift_z")]
        if not columns:
            return None

        from sklearn.ensemble import IsolationForest

        matrix = statistics[columns].replace([np.inf, -np.inf], np.nan).dropna()
        if len(matrix) < 50:
            logger.warning(
                "Too few complete residual rows to fit the multivariate drift detector",
                extra={"rows": len(matrix)},
            )
            return None

        model = IsolationForest(
            n_estimators=int(cfg.get("health.drift.isolation_forest.n_estimators", 200)),
            contamination=float(cfg.get("health.drift.isolation_forest.contamination", 0.02)),
            max_samples=min(
                int(cfg.get("health.drift.isolation_forest.max_samples", 4096)), len(matrix)
            ),
            random_state=int(cfg.get("random_seed", 42)),
            n_jobs=-1,
        ).fit(matrix)

        raw = -model.score_samples(matrix)
        return cls(columns, model, offset=float(np.median(raw)), scale=float(raw.std() or 1.0))

    def score(self, statistics: pd.DataFrame) -> pd.Series:
        """Score rows for multivariate novelty.

        Args:
            statistics: Frame containing this detector's residual columns.

        Returns:
            A ``[0, 1]`` novelty score aligned to ``statistics``; rows with
            incomplete residuals score ``NaN``.
        """
        missing = [column for column in self.columns if column not in statistics.columns]
        if missing:
            raise ValueError(
                f"Drift detector is missing {len(missing)} residual column(s): {missing[:5]}"
            )
        matrix = statistics[self.columns].replace([np.inf, -np.inf], np.nan)
        complete = matrix.dropna()
        scores = pd.Series(np.nan, index=statistics.index, dtype=float)
        if complete.empty:
            return scores
        raw = -self.model.score_samples(complete)
        # Logistic squashing keeps the score interpretable and bounded without
        # depending on the range of any particular scoring batch.
        scores.loc[complete.index] = 1.0 / (1.0 + np.exp(-(raw - self.offset) / self.scale))
        return scores

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the fitted detector."""
        return {
            "n_columns": len(self.columns),
            "columns": list(self.columns),
            "offset": round(self.offset, 6),
            "scale": round(self.scale, 6),
        }


def drift_report(
    statistics: pd.DataFrame,
    frame: pd.DataFrame,
    cfg: Config,
    calibration: DriftCalibration | None = None,
) -> pd.DataFrame:
    """Build a per-turbine drift report over a whole scored frame.

    Args:
        statistics: Output of :func:`compute_drift_statistics`.
        frame: The frame the statistics were computed from (for keys and times).
        cfg: Merged configuration.
        calibration: Persistence thresholds fitted on the healthy population.

    Returns:
        One row per turbine and fired signal, with the sensor, method, severity
        and statistic.  Empty when nothing fired.
    """
    if statistics.empty or frame.empty:
        return pd.DataFrame(
            columns=[TURBINE_ID, TIMESTAMP, "sensor", "method", "severity", "statistic", "limit"]
        )

    rows: list[dict[str, Any]] = []
    times = pd.to_datetime(frame[TIMESTAMP]) if TIMESTAMP in frame.columns else None
    for turbine, positions in frame.groupby(TURBINE_ID, sort=True).groups.items():
        subset = statistics.loc[positions]
        turbine_times = times.loc[positions] if times is not None else None
        for signal in summarise_drift(subset, cfg, turbine_times, calibration=calibration):
            rows.append(
                {
                    TURBINE_ID: str(turbine),
                    TIMESTAMP: None if turbine_times is None else turbine_times.iloc[-1],
                    "sensor": signal.sensor,
                    "method": signal.method,
                    "severity": str(signal.severity),
                    "direction": str(signal.direction),
                    "statistic": signal.statistic,
                    "limit": signal.control_limit,
                    "first_detected": signal.first_detected,
                }
            )
    return pd.DataFrame(rows)
