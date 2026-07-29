"""Health-score feature engineering.

:func:`build_health_features` is the single entry point used by training, by the
API and by the dashboard, for the same reason the failure module has one: a
feature computed at serving time must be computed exactly as it was at training
time, and the only way to guarantee that is to have one function.

Where this differs from the failure feature set - and why it is a separate
pipeline rather than a reuse - is that the health score answers a different
question.  Failure prediction asks *"is something about to break?"* and is
rewarded for finding sharp pre-failure signatures.  Health scoring asks
*"what condition is this machine in right now, and is it getting worse?"*, which
needs slower statistics, condition indicators borrowed from vibration analysis,
and above all **regime conditioning**: a reading is only interpretable against
what the machine was doing when it was taken.

Feature groups
--------------
``condition``
    Current values of the condition channels, plus the one-hot operating regime.
``rolling``
    Trailing mean/std/max over 6/24/72 hours.
``trend``
    Trailing OLS slopes, differences and deviation from a week-long baseline -
    the "getting worse" signal.
``signal_shape``
    Statistical condition indicators over a trailing window: RMS, crest factor,
    kurtosis, skewness, peak-to-peak, high-frequency energy ratio and zero-
    crossing rate.  These are the classic vibration CIs; computed on a trailing
    window they capture spectral character without an FFT and stay leakage-safe.
``regime_relative``
    Deviation and robust z-score against the turbine's own past behaviour
    *within producing regimes*, so a windy site is not scored as unhealthy.
``rule``
    Normalised distance into the operating envelope defined by
    :mod:`wind_turbine_pm.health.sensor_rules`, and recent violation rates.
``drift``
    The CUSUM / EWMA statistics from :mod:`wind_turbine_pm.health.drift`.
``physical``
    Temperature rises above ambient, power-curve ratio, vibration per unit load
    and the lubrication interaction.

Every temporal operation is grouped by ``turbine_id`` and reads current-or-past
rows only; ``tests/test_health_features.py`` asserts this empirically with the
shared :func:`~wind_turbine_pm.features.transformers.assert_no_future_leakage`
helper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import (
    OPERATING_REGIME,
    TIMESTAMP,
    TURBINE_ID,
    OperatingRegime,
)
from wind_turbine_pm.data.preprocessing import hours_to_steps, infer_step_hours
from wind_turbine_pm.features.transformers import (
    expanding_baseline,
    grouped_diff,
    grouped_rolling,
    robust_zscore,
    rolling_slope,
)
from wind_turbine_pm.health.drift import compute_drift_statistics, producing_mask
from wind_turbine_pm.health.regimes import attach_regimes, reference_power
from wind_turbine_pm.health.sensor_rules import evaluate_rules, load_sensor_rules
from wind_turbine_pm.logging_config import get_logger

logger = get_logger(__name__)

EPSILON = 1e-6


@dataclass(frozen=True)
class HealthFeatureSpec:
    """The feature contract produced alongside a health feature matrix."""

    names: tuple[str, ...]
    groups: dict[str, list[str]]
    step_hours: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation."""
        return {
            "names": list(self.names),
            "groups": {key: list(value) for key, value in self.groups.items()},
            "step_hours": self.step_hours,
            "n_features": len(self.names),
        }


def _regime_block(frame: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the operating regime with a fixed, stable column order."""
    regime = frame.get(OPERATING_REGIME)
    if regime is None:
        regime = pd.Series(str(OperatingRegime.MEDIUM_LOAD), index=frame.index)
    regime = regime.astype(str)
    return pd.DataFrame(
        {
            f"regime_{value.value}": (regime == value.value).astype(np.int8)
            for value in OperatingRegime
        },
        index=frame.index,
    )


def _rolling_moments(
    series: pd.Series, groups: pd.Series, window: int, min_periods: int
) -> dict[str, pd.Series]:
    """Trailing raw moments of a series, used to build shape statistics.

    Computing skewness and kurtosis from rolling raw moments keeps everything
    vectorised.  The obvious alternative - ``rolling().apply(scipy.stats.
    kurtosis)`` - is a Python callback per window and is roughly two orders of
    magnitude slower on a fleet-sized frame.

    Args:
        series: Values to aggregate.
        groups: Turbine key aligned to ``series``.
        window: Window length in rows.
        min_periods: Minimum observations required.

    Returns:
        Mapping of ``m1``..``m4`` (raw moments) to trailing series.
    """
    return {
        f"m{power}": grouped_rolling(
            series**power if power > 1 else series,
            groups,
            window,
            "mean",
            min_periods=min_periods,
        )
        for power in (1, 2, 3, 4)
    }


def _signal_shape_block(
    frame: pd.DataFrame, groups: pd.Series, cfg: Config, step_hours: float
) -> pd.DataFrame:
    """Build the statistical / frequency-flavoured condition indicators."""
    sensors = [
        sensor
        for sensor in cfg.get("health.features.signal_shape.sensors", [])
        if sensor in frame.columns
    ]
    stats = set(cfg.get("health.features.signal_shape.stats", []) or ())
    hours = float(cfg.get("health.features.signal_shape.window_hours", 24))
    window = hours_to_steps(hours, step_hours)
    min_periods = max(window // 2, 3)
    label = int(hours)

    out = pd.DataFrame(index=frame.index)
    for sensor in sensors:
        values = frame[sensor].astype(float)
        moments = _rolling_moments(values, groups, window, min_periods)
        mean = moments["m1"]
        variance = (moments["m2"] - mean**2).clip(lower=0.0)
        deviation = np.sqrt(variance)
        rms = np.sqrt(moments["m2"].clip(lower=0.0))

        if "rms" in stats:
            out[f"{sensor}_rms_{label}h"] = rms
        if "peak_to_peak" in stats:
            out[f"{sensor}_p2p_{label}h"] = grouped_rolling(
                values, groups, window, "range", min_periods=min_periods
            )
        if "crest_factor" in stats:
            peak = grouped_rolling(values.abs(), groups, window, "max", min_periods=min_periods)
            out[f"{sensor}_crest_{label}h"] = peak / rms.clip(lower=EPSILON)
        if "skew" in stats:
            third = moments["m3"] - 3.0 * mean * moments["m2"] + 2.0 * mean**3
            out[f"{sensor}_skew_{label}h"] = third / deviation.clip(lower=EPSILON) ** 3
        if "kurtosis" in stats:
            fourth = (
                moments["m4"]
                - 4.0 * mean * moments["m3"]
                + 6.0 * mean**2 * moments["m2"]
                - 3.0 * mean**4
            )
            # Excess kurtosis: 0 for a Gaussian, positive for the impulsive
            # signal an incipient bearing or gear-tooth defect produces.
            out[f"{sensor}_kurtosis_{label}h"] = (fourth / variance.clip(lower=EPSILON) ** 2) - 3.0
        if "hf_ratio" in stats:
            # Sample-to-sample variation against total variation: a proxy for
            # how much of the signal's energy sits at high frequency, which is
            # what rises as a mechanical fault develops.
            high_frequency = grouped_rolling(
                values.groupby(groups, sort=False, observed=True).diff(),
                groups,
                window,
                "std",
                min_periods=min_periods,
            )
            out[f"{sensor}_hf_ratio_{label}h"] = high_frequency / deviation.clip(lower=EPSILON)
        if "zero_crossing_rate" in stats:
            centred = np.sign(values - mean)
            crossings = (
                centred.groupby(groups, sort=False, observed=True).diff().abs() > 0
            ).astype(float)
            out[f"{sensor}_zcr_{label}h"] = grouped_rolling(
                crossings, groups, window, "mean", min_periods=min_periods
            )
    return out


def _regime_relative_block(frame: pd.DataFrame, groups: pd.Series, cfg: Config) -> pd.DataFrame:
    """Build deviations from the turbine's own producing-regime baseline."""
    sensors = [
        sensor
        for sensor in cfg.get("health.features.regime_relative.sensors", [])
        if sensor in frame.columns
    ]
    min_periods = int(cfg.get("health.features.regime_relative.min_periods", 48))
    floor = float(cfg.get("health.features.regime_relative.robust_z_floor", 1e-6))
    mask = producing_mask(frame)

    out = pd.DataFrame(index=frame.index)
    for sensor in sensors:
        values = frame[sensor].astype(float)
        # Baseline built only from rows where the machine was producing, so idle
        # periods cannot depress the reference the current row is judged against.
        baseline = expanding_baseline(values.where(mask), groups, "median", min_periods=min_periods)
        out[f"{sensor}_dev_from_regime_baseline"] = values - baseline
        out[f"{sensor}_regime_robust_z"] = robust_zscore(
            values.where(mask), groups, min_periods=min_periods, floor=floor
        )
    return out


def _rule_block(
    frame: pd.DataFrame, groups: pd.Series, cfg: Config, step_hours: float
) -> pd.DataFrame:
    """Build features describing how close the machine runs to its envelope."""
    rules = load_sensor_rules(cfg)
    evaluation = evaluate_rules(frame, cfg, rules)
    hours = float(cfg.get("health.features.rule_features.window_hours", 24))
    window = hours_to_steps(hours, step_hours)
    min_periods = max(window // 2, 1)
    label = int(hours)

    out = pd.DataFrame(index=frame.index)
    for sensor in evaluation.exceedance.columns:
        exceedance = evaluation.exceedance[sensor]
        out[f"{sensor}_rule_exceedance"] = exceedance
        out[f"{sensor}_rule_exceedance_mean_{label}h"] = grouped_rolling(
            exceedance, groups, window, "mean", min_periods=min_periods
        )
        out[f"{sensor}_rule_exceedance_max_{label}h"] = grouped_rolling(
            exceedance, groups, window, "max", min_periods=min_periods
        )

    if not evaluation.flags.empty:
        warning_columns = [c for c in evaluation.flags.columns if c.endswith("__warning")]
        alarm_columns = [c for c in evaluation.flags.columns if c.endswith("__alarm")]
        validity_columns = [
            c
            for c in evaluation.flags.columns
            if c.endswith(("__out_of_range", "__stuck", "__rate"))
        ]
        for name, columns in (
            ("rule_warning_count", warning_columns),
            ("rule_alarm_count", alarm_columns),
            ("rule_validity_flag_count", validity_columns),
        ):
            if not columns:
                continue
            count = evaluation.flags[columns].sum(axis=1).astype(float)
            out[name] = count
            out[f"{name}_mean_{label}h"] = grouped_rolling(
                count, groups, window, "mean", min_periods=min_periods
            )
    return out


def _physical_block(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Build the physically motivated interaction features."""
    out = pd.DataFrame(index=frame.index)
    ambient = frame["ambient_temperature"].astype(float)

    for column, label in (
        ("gearbox_temperature", "gearbox"),
        ("generator_temperature", "generator"),
        ("bearing_temperature", "bearing"),
        ("oil_temperature", "oil"),
        ("nacelle_temperature", "nacelle"),
    ):
        if column in frame.columns:
            out[f"{label}_temp_above_ambient"] = frame[column].astype(float) - ambient

    wind = frame["wind_speed"].astype(float)
    power = frame["power_output"].astype(float)
    expected = reference_power(wind, cfg)
    out["power_curve_residual"] = power - expected
    ratio = (power / expected.clip(lower=1.0)).clip(-2.0, 3.0)
    out["power_ratio"] = ratio.where(expected > 1.0)

    rated = max(float(cfg.require("features.interactions.rated_power_kw")), EPSILON)
    load = (power / rated).clip(0.0, 1.5)
    out["load_factor"] = load
    out["vibration_per_load"] = frame["vibration"].astype(float) / (load + 0.1)

    oil_pressure = frame["oil_pressure"].astype(float)
    oil_temperature = frame["oil_temperature"].astype(float)
    out["oil_pressure_per_load"] = oil_pressure / (load + 0.1)
    out["oil_pressure_temp_ratio"] = oil_pressure / oil_temperature.abs().clip(lower=1.0)

    thermal_columns = [
        column
        for column in ("gearbox_temperature", "generator_temperature", "bearing_temperature")
        if column in frame.columns
    ]
    thermal = frame[thermal_columns].astype(float)
    out["thermal_stress_index"] = thermal.sub(ambient, axis=0).mean(axis=1)
    out["thermal_spread"] = thermal.max(axis=1) - thermal.min(axis=1)
    return out


def build_health_features(
    frame: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, HealthFeatureSpec]:
    """Build the complete leakage-safe health feature matrix.

    The input must be preprocessed with
    :func:`wind_turbine_pm.data.preprocessing.preprocess`.  The operating regime
    is attached here when it is not already present, so callers cannot
    accidentally build features against a different regime assignment than the
    one the model was trained with.

    Args:
        frame: Preprocessed SCADA frame.
        cfg: Merged configuration carrying the ``health`` namespace.

    Returns:
        ``(features, spec)``.  ``features`` holds only model-facing columns in a
        deterministic order and has the **same index as the input**, so labels
        and metadata can be joined back.  ``spec`` records that order and the
        group membership of every feature.

    Raises:
        ValueError: If required key columns are absent or the frame is empty.
    """
    for column in (TURBINE_ID, TIMESTAMP):
        if column not in frame.columns:
            raise ValueError(f"Health feature engineering requires column {column!r}")
    if frame.empty:
        raise ValueError("Cannot build health features from an empty frame")

    ordered = frame.sort_values([TURBINE_ID, TIMESTAMP])
    if OPERATING_REGIME not in ordered.columns:
        ordered = attach_regimes(ordered, cfg)
    groups = ordered[TURBINE_ID]
    step_hours = infer_step_hours(ordered)

    blocks: list[pd.DataFrame] = []
    feature_groups: dict[str, list[str]] = {}

    def add(name: str, block: pd.DataFrame) -> None:
        """Record a feature block and its group membership."""
        if block.empty:
            return
        feature_groups[name] = list(block.columns)
        blocks.append(block)

    condition = [
        sensor
        for sensor in cfg.require("health.features.condition_sensors")
        if sensor in ordered.columns
    ]

    # --- condition ---------------------------------------------------------
    add("condition", pd.concat([ordered[condition].astype(float), _regime_block(ordered)], axis=1))

    # --- rolling -----------------------------------------------------------
    rolling_columns: dict[str, pd.Series] = {}
    statistics = list(cfg.get("health.features.rolling_stats", ["mean", "std", "max"]))
    for hours in cfg.require("health.features.windows_hours"):
        window = hours_to_steps(float(hours), step_hours)
        min_periods = max(window // 2, 1)
        for sensor in condition:
            values = ordered[sensor].astype(float)
            for statistic in statistics:
                rolling_columns[f"{sensor}_roll_{statistic}_{int(hours)}h"] = grouped_rolling(
                    values, groups, window, statistic, min_periods=min_periods
                )
    add("rolling", pd.DataFrame(rolling_columns, index=ordered.index))

    # --- trend -------------------------------------------------------------
    trend_columns: dict[str, pd.Series] = {}
    for hours in cfg.require("health.features.slope_windows_hours"):
        window = hours_to_steps(float(hours), step_hours)
        for sensor in condition:
            trend_columns[f"{sensor}_slope_{int(hours)}h"] = rolling_slope(
                ordered[sensor].astype(float), groups, window
            )
    for hours in cfg.require("health.features.diff_hours"):
        steps = hours_to_steps(float(hours), step_hours)
        for sensor in condition:
            trend_columns[f"{sensor}_diff_{int(hours)}h"] = grouped_diff(
                ordered[sensor].astype(float), groups, steps
            )
    baseline_hours = float(cfg.require("health.features.baseline_window_hours"))
    baseline_window = hours_to_steps(baseline_hours, step_hours)
    for sensor in condition:
        values = ordered[sensor].astype(float)
        trend_columns[f"{sensor}_dev_from_{int(baseline_hours)}h_baseline"] = values - (
            grouped_rolling(values, groups, baseline_window, "mean")
        )
    add("trend", pd.DataFrame(trend_columns, index=ordered.index))

    # --- signal shape ------------------------------------------------------
    if bool(cfg.get("health.features.signal_shape.enabled", True)):
        add("signal_shape", _signal_shape_block(ordered, groups, cfg, step_hours))

    # --- regime-relative ---------------------------------------------------
    if bool(cfg.get("health.features.regime_relative.enabled", True)):
        add("regime_relative", _regime_relative_block(ordered, groups, cfg))

    # --- rules -------------------------------------------------------------
    if bool(cfg.get("health.features.rule_features.enabled", True)):
        add("rule", _rule_block(ordered, groups, cfg, step_hours))

    # --- drift -------------------------------------------------------------
    if bool(cfg.get("health.features.drift_features.enabled", True)) and bool(
        cfg.get("health.drift.enabled", True)
    ):
        add("drift", compute_drift_statistics(ordered, cfg))

    # --- physical ----------------------------------------------------------
    add("physical", _physical_block(ordered, cfg))

    features = pd.concat(blocks, axis=1)

    excluded = set(cfg.get("health.features.exclude_columns", []))
    features = features[[column for column in features.columns if column not in excluded]]

    duplicated = features.columns[features.columns.duplicated()]
    if len(duplicated):
        raise ValueError(f"Duplicate health feature names produced: {sorted(set(duplicated))}")

    features = features.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    features = features.reindex(frame.index)

    spec = HealthFeatureSpec(
        names=tuple(features.columns),
        groups={
            key: [column for column in value if column in features.columns]
            for key, value in feature_groups.items()
        },
        step_hours=step_hours,
    )
    logger.info(
        "Built health features",
        extra={
            "rows": len(features),
            "features": len(spec.names),
            "groups": {key: len(value) for key, value in spec.groups.items()},
        },
    )
    return features, spec


def align_to_feature_order(features: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    """Reorder/subset a health feature frame to the order a model expects.

    Args:
        features: Feature frame produced by :func:`build_health_features`.
        expected: The feature names, in order, recorded in the model metadata.

    Returns:
        A frame whose columns are exactly ``expected``, in that order.

    Raises:
        ValueError: If any expected feature is absent.
    """
    missing = [name for name in expected if name not in features.columns]
    if missing:
        raise ValueError(
            f"Health feature matrix is missing {len(missing)} expected feature(s): {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
        )
    return features.loc[:, list(expected)]


def minimum_history_hours(cfg: Config) -> float:
    """Hours of history needed before every health feature is defined.

    Args:
        cfg: Merged configuration.

    Returns:
        The longest window any feature block depends on, in hours.
    """
    candidates = [
        *[float(h) for h in cfg.require("health.features.windows_hours")],
        *[float(h) for h in cfg.require("health.features.slope_windows_hours")],
        *[float(h) for h in cfg.require("health.features.diff_hours")],
        float(cfg.get("health.features.signal_shape.window_hours", 24)),
        float(cfg.get("health.features.rule_features.window_hours", 24)),
    ]
    return max(candidates)
