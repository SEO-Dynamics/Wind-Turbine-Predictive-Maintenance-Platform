"""Failure-prediction feature engineering.

:func:`build_failure_features` is the single entry point used by training, by
the API and by the dashboard.  Because all three call the same function, a
feature computed at serving time is computed exactly as it was at training time.

Every temporal operation is grouped by ``turbine_id`` and reads only current or
past rows.  The produced feature list and its order are persisted with the model
so the service can refuse to score a mismatched matrix.

Feature groups
--------------
``raw``
    Current sensor readings.
``lag``
    Values 1/3/6/12/24 hours ago.
``rolling``
    Trailing mean/std/min/max/median/range over 3-48 hour windows.
``trend``
    Differences, guarded percentage changes, trailing OLS slopes and deviation
    from a recent rolling baseline.
``interaction``
    Physically motivated combinations: temperature rises above ambient, power
    ratio against the reference power curve, drivetrain speed ratio, vibration
    normalised by load, lubrication interaction, thermal stress.
``turbine_relative``
    Deviation from the turbine's own past-only expanding median/mean and a
    robust z-score, so a turbine that simply runs hotter than its neighbours is
    not flagged for it.
``time``
    Calendar components with cyclical encodings. Only hour-of-day is enabled by
    default; see ``configs/features.yaml`` for why ``month`` and ``day_of_week``
    are excluded under a chronological split.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import TIMESTAMP, TURBINE_ID, OperationalStatus
from wind_turbine_pm.data.preprocessing import hours_to_steps, infer_step_hours
from wind_turbine_pm.features.transformers import (
    cyclical_encode,
    expanding_baseline,
    grouped_diff,
    grouped_lag,
    grouped_pct_change,
    grouped_rolling,
    robust_zscore,
    rolling_slope,
)
from wind_turbine_pm.logging_config import get_logger

logger = get_logger(__name__)

EPSILON = 1e-6


@dataclass(frozen=True)
class FeatureSpec:
    """The feature contract produced alongside a feature matrix."""

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


def _status_features(frame: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the operating state with a fixed, stable column order."""
    status = frame.get("operational_status")
    if status is None:
        status = pd.Series(str(OperationalStatus.NORMAL), index=frame.index)
    status = status.astype(str)
    return pd.DataFrame(
        {
            f"status_{state.value}": (status == state.value).astype(np.int8)
            for state in OperationalStatus
        },
        index=frame.index,
    )


def _reference_power(wind: pd.Series, cfg: Config) -> pd.Series:
    """Evaluate the reference power curve used for the power-curve residual."""
    rated_power = float(cfg.require("features.interactions.rated_power_kw"))
    rated_wind = float(cfg.require("features.interactions.rated_wind_speed"))
    cut_in = float(cfg.require("features.interactions.cut_in_wind_speed"))
    cut_out = float(cfg.require("features.interactions.cut_out_wind_speed"))

    values = wind.astype(float)
    expected = pd.Series(0.0, index=wind.index)
    ramp = (values >= cut_in) & (values < rated_wind)
    expected[ramp] = rated_power * ((values[ramp] - cut_in) / (rated_wind - cut_in)) ** 3
    expected[(values >= rated_wind) & (values <= cut_out)] = rated_power
    return expected


def _interaction_features(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Build the physically motivated interaction block."""
    out = pd.DataFrame(index=frame.index)
    ambient = frame["ambient_temperature"].astype(float)

    for column, label in (
        ("generator_temperature", "generator"),
        ("gearbox_temperature", "gearbox"),
        ("bearing_temperature", "bearing"),
        ("nacelle_temperature", "nacelle"),
        ("oil_temperature", "oil"),
    ):
        if column in frame.columns:
            out[f"{label}_temp_above_ambient"] = frame[column].astype(float) - ambient

    # Drivetrain consistency: a slipping or damaged coupling changes this ratio.
    rotor = frame["rotor_speed"].astype(float)
    generator = frame["generator_speed"].astype(float)
    out["rotor_generator_speed_ratio"] = generator / rotor.abs().clip(lower=EPSILON)
    out["rotor_generator_speed_ratio"] = out["rotor_generator_speed_ratio"].clip(0.0, 400.0)

    # Power-curve behaviour: how much of the theoretically available power is
    # actually produced, and the absolute shortfall.
    wind = frame["wind_speed"].astype(float)
    power = frame["power_output"].astype(float)
    expected_power = _reference_power(wind, cfg)
    out["expected_power"] = expected_power
    out["power_ratio"] = (power / expected_power.clip(lower=1.0)).clip(-2.0, 3.0)
    out["power_curve_residual"] = power - expected_power
    # Only meaningful while producing; zero it out when the curve predicts idle.
    producing = expected_power > 1.0
    out.loc[~producing, "power_ratio"] = np.nan
    out["power_per_wind_cubed"] = (power / (wind.clip(lower=0.5) ** 3)).clip(-50.0, 200.0)

    # Vibration relative to what the current operating point justifies.
    load = (power / max(float(cfg.require("features.interactions.rated_power_kw")), EPSILON)).clip(
        0.0, 1.5
    )
    out["load_factor"] = load
    out["vibration_per_load"] = frame["vibration"].astype(float) / (load + 0.1)
    out["vibration_per_rotor_speed"] = frame["vibration"].astype(float) / rotor.clip(lower=0.5)

    # Lubrication: pressure falling while temperature rises is the classic
    # signature of a degrading oil film.
    oil_pressure = frame["oil_pressure"].astype(float)
    oil_temperature = frame["oil_temperature"].astype(float)
    out["oil_pressure_temp_ratio"] = oil_pressure / oil_temperature.abs().clip(lower=1.0)
    out["oil_pressure_temp_product"] = oil_pressure * oil_temperature
    out["oil_pressure_per_load"] = oil_pressure / (load + 0.1)

    # Aggregate thermal stress across the drivetrain.
    thermal_columns = [
        column
        for column in ("gearbox_temperature", "generator_temperature", "bearing_temperature")
        if column in frame.columns
    ]
    out["thermal_stress_index"] = (
        frame[thermal_columns].astype(float).sub(ambient, axis=0).mean(axis=1)
    )
    out["thermal_spread"] = frame[thermal_columns].astype(float).max(axis=1) - frame[
        thermal_columns
    ].astype(float).min(axis=1)

    if "hydraulic_pressure" in frame.columns:
        out["hydraulic_per_load"] = frame["hydraulic_pressure"].astype(float) / (load + 0.1)
    if "brake_temperature" in frame.columns:
        out["brake_temp_above_ambient"] = frame["brake_temperature"].astype(float) - ambient

    return out


#: Calendar components the time block can emit, with their cycle lengths.
_CALENDAR_COMPONENTS: dict[str, tuple[str, float]] = {
    "hour": ("hour", 24.0),
    "day_of_week": ("dayofweek", 7.0),
    "month": ("month", 12.0),
}


def _time_features(frame: pd.DataFrame, cyclical: bool, components: Sequence[str]) -> pd.DataFrame:
    """Build calendar features from the timestamp.

    Only the requested components are emitted. See
    ``configs/features.yaml -> features.time_features.components`` for why
    ``month`` and ``day_of_week`` are excluded by default under a chronological
    split.

    Args:
        frame: Frame carrying the timestamp column.
        cyclical: Whether to add sine/cosine encodings.
        components: Calendar component names to emit.

    Returns:
        The calendar feature block.

    Raises:
        ValueError: If an unknown component is requested.
    """
    unknown = [name for name in components if name not in _CALENDAR_COMPONENTS]
    if unknown:
        raise ValueError(
            f"Unknown time_features.components {unknown}; "
            f"known components: {sorted(_CALENDAR_COMPONENTS)}"
        )

    times = pd.to_datetime(frame[TIMESTAMP])
    out = pd.DataFrame(index=frame.index)
    for name in components:
        attribute, period = _CALENDAR_COMPONENTS[name]
        out[name] = getattr(times.dt, attribute).astype(np.int16)
        if cyclical:
            out = pd.concat([out, cyclical_encode(out[name], period, name)], axis=1)
    return out


def build_failure_features(frame: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, FeatureSpec]:
    """Build the complete leakage-safe feature matrix.

    The input frame must be preprocessed (:func:`wind_turbine_pm.data.
    preprocessing.preprocess`) and sorted by turbine and time.  The output has
    the same index as the input, so labels and metadata can be joined back.

    Args:
        frame: Preprocessed SCADA frame.
        cfg: Merged configuration.

    Returns:
        ``(features, spec)`` where ``features`` holds only model-facing columns
        in a deterministic order and ``spec`` records that order and the group
        membership of each feature.

    Raises:
        ValueError: If required key columns are absent or the frame is empty.
    """
    for column in (TURBINE_ID, TIMESTAMP):
        if column not in frame.columns:
            raise ValueError(f"Feature engineering requires column {column!r}")
    if frame.empty:
        raise ValueError("Cannot build features from an empty frame")

    ordered = frame.sort_values([TURBINE_ID, TIMESTAMP])
    groups = ordered[TURBINE_ID]
    step_hours = infer_step_hours(ordered)

    blocks: list[pd.DataFrame] = []
    feature_groups: dict[str, list[str]] = {}

    # --- raw --------------------------------------------------------------
    raw_sensors = [c for c in cfg.require("features.raw_sensors") if c in ordered.columns]
    raw_block = ordered[raw_sensors].astype(float)
    status_block = _status_features(ordered)
    raw_block = pd.concat([raw_block, status_block], axis=1)
    feature_groups["raw"] = list(raw_block.columns)
    blocks.append(raw_block)

    dynamic = [c for c in cfg.require("features.dynamic_sensors") if c in ordered.columns]

    # --- lag --------------------------------------------------------------
    lag_columns: dict[str, pd.Series] = {}
    for hours in cfg.require("features.lags_hours"):
        steps = hours_to_steps(float(hours), step_hours)
        for sensor in dynamic:
            lag_columns[f"{sensor}_lag_{int(hours)}h"] = grouped_lag(
                ordered[sensor].astype(float), groups, steps
            )
    lag_block = pd.DataFrame(lag_columns, index=ordered.index)
    feature_groups["lag"] = list(lag_block.columns)
    blocks.append(lag_block)

    # --- rolling ----------------------------------------------------------
    full_stat_sensors = set(cfg.get("features.rolling.full_stat_sensors", []))
    stats = list(cfg.require("features.rolling.stats"))
    min_fraction = float(cfg.get("features.rolling.min_periods_fraction", 0.5))
    rolling_columns: dict[str, pd.Series] = {}
    for hours in cfg.require("features.rolling.windows_hours"):
        window = hours_to_steps(float(hours), step_hours)
        min_periods = max(int(window * min_fraction), 1)
        for sensor in dynamic:
            sensor_stats = stats if sensor in full_stat_sensors else ["mean", "std"]
            series = ordered[sensor].astype(float)
            for stat in sensor_stats:
                rolling_columns[f"{sensor}_roll_{stat}_{int(hours)}h"] = grouped_rolling(
                    series, groups, window, stat, min_periods=min_periods
                )
    rolling_block = pd.DataFrame(rolling_columns, index=ordered.index)
    feature_groups["rolling"] = list(rolling_block.columns)
    blocks.append(rolling_block)

    # --- trend ------------------------------------------------------------
    trend_columns: dict[str, pd.Series] = {}
    for hours in cfg.require("features.trend.diff_hours"):
        steps = hours_to_steps(float(hours), step_hours)
        for sensor in dynamic:
            series = ordered[sensor].astype(float)
            trend_columns[f"{sensor}_diff_{int(hours)}h"] = grouped_diff(series, groups, steps)
            trend_columns[f"{sensor}_rate_{int(hours)}h"] = grouped_diff(series, groups, steps) / (
                float(hours)
            )
    for hours in cfg.require("features.trend.pct_change_hours"):
        steps = hours_to_steps(float(hours), step_hours)
        for sensor in dynamic:
            trend_columns[f"{sensor}_pct_change_{int(hours)}h"] = grouped_pct_change(
                ordered[sensor].astype(float), groups, steps
            )
    for hours in cfg.require("features.trend.slope_windows_hours"):
        window = hours_to_steps(float(hours), step_hours)
        for sensor in dynamic:
            trend_columns[f"{sensor}_slope_{int(hours)}h"] = rolling_slope(
                ordered[sensor].astype(float), groups, window
            )
    baseline_hours = float(cfg.require("features.trend.baseline_window_hours"))
    baseline_window = hours_to_steps(baseline_hours, step_hours)
    for sensor in dynamic:
        series = ordered[sensor].astype(float)
        baseline = grouped_rolling(series, groups, baseline_window, "mean")
        trend_columns[f"{sensor}_dev_from_{int(baseline_hours)}h_baseline"] = series - baseline
    trend_block = pd.DataFrame(trend_columns, index=ordered.index)
    feature_groups["trend"] = list(trend_block.columns)
    blocks.append(trend_block)

    # --- interactions -----------------------------------------------------
    if bool(cfg.get("features.interactions.enabled", True)):
        interaction_block = _interaction_features(ordered, cfg)
        feature_groups["interaction"] = list(interaction_block.columns)
        blocks.append(interaction_block)

    # --- turbine-relative -------------------------------------------------
    if bool(cfg.get("features.turbine_relative.enabled", True)):
        min_periods = int(cfg.get("features.turbine_relative.min_periods", 48))
        floor = float(cfg.get("features.turbine_relative.robust_z_floor", 1e-6))
        relative_columns: dict[str, pd.Series] = {}
        for sensor in dynamic:
            series = ordered[sensor].astype(float)
            relative_columns[f"{sensor}_dev_from_turbine_median"] = series - expanding_baseline(
                series, groups, "median", min_periods=min_periods
            )
            relative_columns[f"{sensor}_dev_from_turbine_mean"] = series - expanding_baseline(
                series, groups, "mean", min_periods=min_periods
            )
            relative_columns[f"{sensor}_turbine_robust_z"] = robust_zscore(
                series, groups, min_periods=min_periods, floor=floor
            )
        # Deviation from the "normal regime" baseline: the turbine's own past
        # behaviour restricted to rows where the controller reported normal
        # operation, so idle periods do not depress the reference.
        normal_mask = (
            ordered.get("operational_status", pd.Series("normal", index=ordered.index))
            .astype(str)
            .eq(str(OperationalStatus.NORMAL))
        )
        for sensor in ("vibration", "gearbox_temperature", "oil_pressure"):
            if sensor not in ordered.columns:
                continue
            series = ordered[sensor].astype(float)
            normal_only = series.where(normal_mask)
            normal_baseline = expanding_baseline(
                normal_only, groups, "median", min_periods=min_periods
            )
            relative_columns[f"{sensor}_dev_from_normal_regime"] = series - normal_baseline
        relative_block = pd.DataFrame(relative_columns, index=ordered.index)
        feature_groups["turbine_relative"] = list(relative_block.columns)
        blocks.append(relative_block)

    # --- time -------------------------------------------------------------
    if bool(cfg.get("features.time_features.enabled", True)):
        time_block = _time_features(
            ordered,
            cyclical=bool(cfg.get("features.time_features.cyclical", True)),
            components=list(cfg.get("features.time_features.components", ["hour"])),
        )
        feature_groups["time"] = list(time_block.columns)
        blocks.append(time_block)

    features = pd.concat(blocks, axis=1)

    excluded = set(cfg.get("features.exclude_columns", []))
    keep = [column for column in features.columns if column not in excluded]
    features = features[keep]

    duplicated = features.columns[features.columns.duplicated()]
    if len(duplicated):
        raise ValueError(f"Duplicate feature names produced: {sorted(set(duplicated))}")

    features = features.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    features = features.reindex(frame.index)

    spec = FeatureSpec(
        names=tuple(features.columns),
        groups={
            key: [c for c in value if c in features.columns]
            for key, value in feature_groups.items()
        },
        step_hours=step_hours,
    )
    logger.info(
        "Built failure features",
        extra={
            "rows": len(features),
            "features": len(spec.names),
            "groups": {key: len(value) for key, value in spec.groups.items()},
        },
    )
    return features, spec


def align_to_feature_order(features: pd.DataFrame, expected: Sequence[str]) -> pd.DataFrame:
    """Reorder/subset a feature frame to the exact order a model expects.

    Args:
        features: Feature frame produced by :func:`build_failure_features`.
        expected: The feature names, in order, recorded in model metadata.

    Returns:
        A frame whose columns are exactly ``expected``, in that order.

    Raises:
        ValueError: If any expected feature is absent.
    """
    missing = [name for name in expected if name not in features.columns]
    if missing:
        raise ValueError(
            f"Feature matrix is missing {len(missing)} expected feature(s): {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
        )
    return features.loc[:, list(expected)]


def minimum_history_rows(cfg: Config, step_hours: float) -> int:
    """Number of trailing observations needed before features are well-defined.

    Args:
        cfg: Merged configuration.
        step_hours: Sampling interval in hours.

    Returns:
        The required history length in rows.
    """
    longest_window = max(
        [*cfg.require("features.rolling.windows_hours"), *cfg.require("features.lags_hours")]
    )
    return hours_to_steps(float(longest_window), step_hours)
