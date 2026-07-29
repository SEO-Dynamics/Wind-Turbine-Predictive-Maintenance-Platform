"""Operating-regime detection.

A turbine's sensor values only mean something relative to what the machine is
doing.  60 degC in the gearbox is unremarkable at rated power and a warning
sign while idling; 2 mm/s of vibration is healthy at low load and suspiciously
low at high load.  Every regime-relative feature, every drift baseline and
every component score in this module is conditioned on the label assigned here.

Two methods are implemented:

``threshold``
    Deterministic rules over wind speed, power output and the controller state.
    This is the default because an operator can audit it: every boundary is a
    number in ``configs/health_model.yaml``, and the same input always produces
    the same label with no fitted artifact in between.

``kmeans``
    Clusters ``(wind_speed, power_output, rotor_speed)`` and maps the resulting
    centres onto the same ordered labels by mean power.  Offered for fleets
    whose power curve is not known, at the cost of needing a fitted artifact and
    of labels that can move when the model is refitted.

Both return the same :class:`~wind_turbine_pm.constants.OperatingRegime` values,
so everything downstream is indifferent to which was used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import (
    OPERATING_REGIME,
    OPERATIONAL_STATUS,
    OperatingRegime,
    OperationalStatus,
)
from wind_turbine_pm.logging_config import get_logger

logger = get_logger(__name__)

#: Controller states in which the machine is not available to produce.
_OFFLINE_STATES: frozenset[str] = frozenset(
    {str(OperationalStatus.MAINTENANCE), str(OperationalStatus.FAULT)}
)


@dataclass(frozen=True)
class RegimeThresholds:
    """The boundaries used by the threshold method."""

    cut_in_wind_speed: float
    cut_out_wind_speed: float
    idle_power_kw: float
    low_medium_edge: float
    medium_high_edge: float
    curtailment_power_ratio: float

    @classmethod
    def from_config(cls, cfg: Config) -> RegimeThresholds:
        """Build the thresholds from configuration.

        Args:
            cfg: Merged configuration carrying the ``health`` namespace.

        Returns:
            The configured :class:`RegimeThresholds`.
        """
        return cls(
            cut_in_wind_speed=float(cfg.get("health.operating_regimes.cut_in_wind_speed", 3.0)),
            cut_out_wind_speed=float(cfg.get("health.operating_regimes.cut_out_wind_speed", 25.0)),
            idle_power_kw=float(cfg.get("health.operating_regimes.idle_power_kw", 25.0)),
            low_medium_edge=float(cfg.get("health.operating_regimes.low_medium_edge", 7.0)),
            medium_high_edge=float(cfg.get("health.operating_regimes.medium_high_edge", 11.0)),
            curtailment_power_ratio=float(
                cfg.get("health.operating_regimes.curtailment_power_ratio", 0.55)
            ),
        )


def reference_power(wind_speed: pd.Series, cfg: Config) -> pd.Series:
    """Evaluate the reference power curve for a wind-speed series.

    Mirrors the curve the Failure Prediction Module uses for its power-curve
    residual and reads the *same* configuration keys
    (``features.interactions.*``), so the two modules can never describe the
    same turbine with two different rated curves.  ``tests/test_operating_
    regimes.py`` pins the agreement.

    Args:
        wind_speed: Wind speed in m/s.
        cfg: Merged configuration.

    Returns:
        Expected power output in kW, aligned to ``wind_speed``.
    """
    rated_power = float(cfg.require("features.interactions.rated_power_kw"))
    rated_wind = float(cfg.require("features.interactions.rated_wind_speed"))
    cut_in = float(cfg.require("features.interactions.cut_in_wind_speed"))
    cut_out = float(cfg.require("features.interactions.cut_out_wind_speed"))

    values = wind_speed.astype(float)
    expected = pd.Series(0.0, index=wind_speed.index)
    ramp = (values >= cut_in) & (values < rated_wind)
    expected[ramp] = rated_power * ((values[ramp] - cut_in) / (rated_wind - cut_in)) ** 3
    expected[(values >= rated_wind) & (values <= cut_out)] = rated_power
    return expected


def assign_regimes_threshold(frame: pd.DataFrame, cfg: Config) -> pd.Series:
    """Label each observation using the deterministic threshold rules.

    Precedence, highest first:

    1. ``offline`` - the controller reports maintenance or fault.
    2. ``idle`` - wind below cut-in, above cut-out, or output at/below the idle
       power floor.
    3. ``curtailed`` - producing, but well below the reference power curve. This
       is deliberately separated from the load bands: a curtailed machine looks
       "cold for its wind speed" and would otherwise poison a load-band
       baseline.
    4. ``low_load`` / ``medium_load`` / ``high_load`` by wind speed.

    Args:
        frame: Frame with ``wind_speed``, ``power_output`` and optionally
            ``operational_status``.
        cfg: Merged configuration.

    Returns:
        A string series of :class:`~wind_turbine_pm.constants.OperatingRegime`
        values, aligned to ``frame``.
    """
    thresholds = RegimeThresholds.from_config(cfg)
    wind = pd.to_numeric(frame.get("wind_speed"), errors="coerce").astype(float)
    power = pd.to_numeric(frame.get("power_output"), errors="coerce").astype(float)

    regime = pd.Series(str(OperatingRegime.MEDIUM_LOAD), index=frame.index, dtype=object)
    regime[wind < thresholds.low_medium_edge] = str(OperatingRegime.LOW_LOAD)
    regime[wind >= thresholds.medium_high_edge] = str(OperatingRegime.HIGH_LOAD)

    expected = reference_power(wind.fillna(0.0), cfg)
    producing = expected > 1.0
    ratio = pd.Series(np.nan, index=frame.index, dtype=float)
    ratio[producing] = power[producing] / expected[producing]
    regime[producing & (ratio < thresholds.curtailment_power_ratio)] = str(
        OperatingRegime.CURTAILED
    )

    idle = (
        (wind < thresholds.cut_in_wind_speed)
        | (wind > thresholds.cut_out_wind_speed)
        | (power <= thresholds.idle_power_kw)
    )
    regime[idle.fillna(True)] = str(OperatingRegime.IDLE)

    if OPERATIONAL_STATUS in frame.columns:
        offline = frame[OPERATIONAL_STATUS].astype(str).isin(_OFFLINE_STATES)
        regime[offline] = str(OperatingRegime.OFFLINE)

    return regime


def assign_regimes_kmeans(frame: pd.DataFrame, cfg: Config) -> pd.Series:
    """Label each observation by clustering the operating point.

    The clusters are ordered by mean power output and mapped onto
    ``idle`` / ``low_load`` / ``medium_load`` / ``high_load``.  ``offline`` is
    still assigned from the controller state, because that is a fact rather than
    something to be discovered, and rows the controller reports as offline are
    excluded from the fit so they cannot drag a centre down.

    Args:
        frame: Frame carrying the clustering features.
        cfg: Merged configuration.

    Returns:
        A string series of regime labels aligned to ``frame``.

    Raises:
        ValueError: If none of the configured clustering features are present.
    """
    from sklearn.cluster import KMeans

    columns = [
        column
        for column in cfg.get("health.operating_regimes.kmeans.features", ["wind_speed"])
        if column in frame.columns
    ]
    if not columns:
        raise ValueError(
            "K-means regime detection needs at least one of the configured features "
            f"{list(cfg.get('health.operating_regimes.kmeans.features', []))} in the frame"
        )

    n_clusters = int(cfg.get("health.operating_regimes.kmeans.n_clusters", 4))
    max_samples = int(cfg.get("health.operating_regimes.kmeans.max_samples", 20000))
    seed = int(cfg.get("random_seed", 42))

    matrix = frame[columns].astype(float).fillna(0.0)
    online = pd.Series(True, index=frame.index)
    if OPERATIONAL_STATUS in frame.columns:
        online = ~frame[OPERATIONAL_STATUS].astype(str).isin(_OFFLINE_STATES)

    fit_rows = matrix.loc[online]
    if fit_rows.empty:
        fit_rows = matrix
    if len(fit_rows) > max_samples:
        fit_rows = fit_rows.sample(max_samples, random_state=seed)

    # Standardise so power (kW, ~1e3) does not dominate wind speed (m/s, ~1e1).
    centre = fit_rows.mean()
    scale = fit_rows.std().replace(0.0, 1.0)
    model = KMeans(
        n_clusters=min(n_clusters, max(len(fit_rows), 1)),
        random_state=seed,
        n_init=int(cfg.get("health.operating_regimes.kmeans.n_init", 10)),
    ).fit((fit_rows - centre) / scale)

    labels = pd.Series(model.predict((matrix - centre) / scale), index=frame.index)
    power_column = "power_output" if "power_output" in frame.columns else columns[0]
    ordering = frame[power_column].astype(float).groupby(labels).mean().sort_values().index.tolist()
    ladder = [
        str(OperatingRegime.IDLE),
        str(OperatingRegime.LOW_LOAD),
        str(OperatingRegime.MEDIUM_LOAD),
        str(OperatingRegime.HIGH_LOAD),
    ]
    mapping = {
        cluster: ladder[min(position, len(ladder) - 1)] for position, cluster in enumerate(ordering)
    }
    regime = labels.map(mapping).astype(object)

    if OPERATIONAL_STATUS in frame.columns:
        regime[frame[OPERATIONAL_STATUS].astype(str).isin(_OFFLINE_STATES)] = str(
            OperatingRegime.OFFLINE
        )
    return regime


def assign_regimes(frame: pd.DataFrame, cfg: Config) -> pd.Series:
    """Assign an operating regime to every observation.

    Args:
        frame: Preprocessed SCADA frame.
        cfg: Merged configuration.

    Returns:
        A string series of regime labels aligned to ``frame``.

    Raises:
        ValueError: If ``health.operating_regimes.method`` is not recognised.
    """
    method = str(cfg.get("health.operating_regimes.method", "threshold"))
    if method == "threshold":
        regime = assign_regimes_threshold(frame, cfg)
    elif method == "kmeans":
        regime = assign_regimes_kmeans(frame, cfg)
    else:
        raise ValueError(
            f"Unknown health.operating_regimes.method {method!r}; expected 'threshold' or 'kmeans'"
        )

    logger.info(
        "Assigned operating regimes",
        extra={
            "method": method,
            "counts": {str(key): int(value) for key, value in regime.value_counts().items()},
        },
    )
    return regime


def attach_regimes(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Return a copy of ``frame`` with the ``operating_regime`` column attached.

    Args:
        frame: Preprocessed SCADA frame.
        cfg: Merged configuration.

    Returns:
        The frame plus the regime column.
    """
    out = frame.copy()
    out[OPERATING_REGIME] = assign_regimes(frame, cfg)
    return out


def regime_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise how much time the fleet spends in each regime.

    Args:
        frame: Frame carrying ``operating_regime``.

    Returns:
        One row per regime with the observation count and share, ordered by the
        canonical regime ladder.

    Raises:
        KeyError: If the regime column is absent.
    """
    if OPERATING_REGIME not in frame.columns:
        raise KeyError(f"regime_summary requires the {OPERATING_REGIME!r} column")
    counts = frame[OPERATING_REGIME].value_counts()
    order = [str(regime) for regime in OperatingRegime]
    counts = counts.reindex(order).fillna(0).astype(int)
    return pd.DataFrame(
        {
            OPERATING_REGIME: counts.index,
            "observations": counts.to_numpy(),
            "share": (counts / max(int(counts.sum()), 1)).to_numpy(),
        }
    ).reset_index(drop=True)
