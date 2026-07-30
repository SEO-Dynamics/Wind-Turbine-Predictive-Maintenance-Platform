"""Leakage-safe features for one-class anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import OPERATING_REGIME, TIMESTAMP, TURBINE_ID, OperatingRegime
from wind_turbine_pm.data.preprocessing import hours_to_steps, infer_step_hours
from wind_turbine_pm.features.transformers import (
    grouped_diff,
    grouped_rolling,
    rolling_slope,
)
from wind_turbine_pm.health.drift import regime_conditioned_z
from wind_turbine_pm.health.regimes import attach_regimes, reference_power


@dataclass(frozen=True)
class AnomalyFeatureSpec:
    """Persisted feature order and grouping."""

    names: tuple[str, ...]
    groups: dict[str, list[str]]
    step_hours: float

    def to_dict(self) -> dict[str, object]:
        return {
            "names": list(self.names),
            "groups": {name: list(columns) for name, columns in self.groups.items()},
            "step_hours": self.step_hours,
            "n_features": len(self.names),
        }


def _physical_features(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Build operating-point residuals rather than absolute-load shortcuts."""
    output = pd.DataFrame(index=frame.index)
    ambient = frame["ambient_temperature"].astype(float)
    for column in (
        "generator_temperature",
        "gearbox_temperature",
        "bearing_temperature",
        "oil_temperature",
        "nacelle_temperature",
        "brake_temperature",
    ):
        output[f"{column}_above_ambient"] = frame[column].astype(float) - ambient

    expected = reference_power(frame["wind_speed"].astype(float).fillna(0.0), cfg)
    power = frame["power_output"].astype(float)
    output["power_curve_residual"] = power - expected
    output["power_ratio"] = power / expected.clip(lower=1.0)
    output.loc[expected <= 1.0, "power_ratio"] = np.nan
    output["generator_rotor_ratio"] = frame["generator_speed"].astype(float) / frame[
        "rotor_speed"
    ].astype(float).abs().clip(lower=0.5)
    output["vibration_per_load"] = frame["vibration"].astype(float) / (
        power.clip(lower=0.0) / 2000.0 + 0.1
    )
    output["oil_pressure_temperature_ratio"] = frame["oil_pressure"].astype(float) / frame[
        "oil_temperature"
    ].astype(float).abs().clip(lower=1.0)
    return output.replace([np.inf, -np.inf], np.nan)


def build_anomaly_features(
    frame: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, AnomalyFeatureSpec]:
    """Create deterministic, past-only novelty features.

    The input may already carry ``operating_regime``; otherwise the shared
    health-regime implementation attaches it. The output preserves the original
    row index so hidden evaluation truth can be joined without entering the
    feature matrix.
    """
    if frame.empty:
        raise ValueError("Cannot build anomaly features from an empty frame")
    for column in (TURBINE_ID, TIMESTAMP):
        if column not in frame:
            raise ValueError(f"Anomaly features require column {column!r}")

    ordered = frame.sort_values([TURBINE_ID, TIMESTAMP]).copy()
    if OPERATING_REGIME not in ordered:
        ordered = attach_regimes(ordered, cfg)
    groups = ordered[TURBINE_ID].astype(str)
    step_hours = infer_step_hours(ordered)
    sensors = [name for name in cfg.require("anomaly.features.sensors") if name in ordered]
    dynamic = [name for name in cfg.require("anomaly.features.dynamic_sensors") if name in ordered]

    blocks: list[pd.DataFrame] = []
    feature_groups: dict[str, list[str]] = {}

    raw = ordered[sensors].astype(float)
    regimes = pd.get_dummies(ordered[OPERATING_REGIME].astype(str), prefix="regime", dtype=np.int8)
    raw = pd.concat([raw, regimes], axis=1)
    feature_groups["raw_and_regime"] = list(raw.columns)
    blocks.append(raw)

    rolling_columns: dict[str, pd.Series] = {}
    for hours in cfg.require("anomaly.features.windows_hours"):
        steps = hours_to_steps(float(hours), step_hours)
        for sensor in dynamic:
            values = ordered[sensor].astype(float)
            for stat in cfg.require("anomaly.features.rolling_stats"):
                rolling_columns[f"{sensor}_{stat}_{int(hours)}h"] = grouped_rolling(
                    values, groups, steps, str(stat)
                )
    rolling = pd.DataFrame(rolling_columns, index=ordered.index)
    feature_groups["rolling"] = list(rolling.columns)
    blocks.append(rolling)

    trend_columns: dict[str, pd.Series] = {}
    for hours in cfg.require("anomaly.features.diff_hours"):
        steps = hours_to_steps(float(hours), step_hours)
        for sensor in dynamic:
            trend_columns[f"{sensor}_diff_{int(hours)}h"] = grouped_diff(
                ordered[sensor].astype(float), groups, steps
            )
    for hours in cfg.require("anomaly.features.slope_hours"):
        steps = hours_to_steps(float(hours), step_hours)
        for sensor in dynamic:
            trend_columns[f"{sensor}_slope_{int(hours)}h"] = rolling_slope(
                ordered[sensor].astype(float), groups, steps
            )
    trend = pd.DataFrame(trend_columns, index=ordered.index)
    feature_groups["trend"] = list(trend.columns)
    blocks.append(trend)

    physical = _physical_features(ordered, cfg)
    feature_groups["physical"] = list(physical.columns)
    blocks.append(physical)

    regime_groups = groups + "::" + ordered[OPERATING_REGIME].astype(str)
    eligible = ordered[OPERATING_REGIME].astype(str) != str(OperatingRegime.OFFLINE)
    baseline_columns: dict[str, pd.Series] = {}
    minimum = int(cfg.require("anomaly.features.regime_baseline_min_periods"))
    for sensor in dynamic:
        baseline_columns[f"{sensor}_regime_z"] = regime_conditioned_z(
            ordered[sensor].astype(float),
            regime_groups,
            eligible,
            min_periods=minimum,
        )
    baseline = pd.DataFrame(baseline_columns, index=ordered.index)
    feature_groups["regime_relative"] = list(baseline.columns)
    blocks.append(baseline)

    features = pd.concat(blocks, axis=1).replace([np.inf, -np.inf], np.nan)
    excluded = set(cfg.require("anomaly.features.exclude_columns"))
    leaked = sorted(excluded.intersection(features.columns))
    if leaked:
        raise ValueError(f"Anomaly feature matrix contains forbidden truth columns: {leaked}")
    features = features.astype(float)
    spec = AnomalyFeatureSpec(
        names=tuple(features.columns),
        groups=feature_groups,
        step_hours=float(step_hours),
    )
    return features.reindex(frame.index), spec
