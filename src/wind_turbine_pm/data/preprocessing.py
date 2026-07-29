"""Preprocessing and target construction.

Two rules govern everything in this module:

1. **Never cross a turbine boundary.**  Every temporal operation is applied
   inside ``groupby(turbine_id)``.
2. **Never look forward.**  Forward fill is allowed (it copies *past* values
   forward); backward fill is not, because it would copy a future reading into
   the present.  Remaining gaps are closed with a per-turbine median computed
   from the whole training history, which is a documented, deliberate
   simplification - see :func:`impute_missing`.

Target construction (:func:`create_failure_target`) is the one place where
future information is legitimately used, because that is what a label *is*.
The resulting column is excluded from the feature matrix by configuration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import (
    FAILURE_EVENT,
    OPERATIONAL_STATUS,
    TIMESTAMP,
    TURBINE_ID,
    OperationalStatus,
)
from wind_turbine_pm.logging_config import get_logger

logger = get_logger(__name__)


def infer_step_hours(frame: pd.DataFrame) -> float:
    """Infer the sampling interval in hours from the data.

    Uses the median positive per-turbine timestamp difference, which is robust
    to gaps and to the occasional duplicate.

    Args:
        frame: Frame with ``turbine_id`` and a datetime ``timestamp``.

    Returns:
        The sampling interval in hours.

    Raises:
        ValueError: If no positive interval can be inferred.
    """
    times = pd.to_datetime(frame[TIMESTAMP], errors="coerce")
    deltas = times.groupby(frame[TURBINE_ID].to_numpy()).diff().dropna()
    positive = deltas[deltas > pd.Timedelta(0)]
    if positive.empty:
        raise ValueError("Cannot infer sampling interval: no positive timestamp differences")
    return float(positive.median() / pd.Timedelta("1h"))


def hours_to_steps(hours: float, step_hours: float) -> int:
    """Convert a duration in hours to a whole number of row offsets.

    Args:
        hours: Duration in hours.
        step_hours: Sampling interval in hours.

    Returns:
        At least 1 row offset.
    """
    return max(int(round(hours / step_hours)), 1)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def normalise_keys(frame: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict[str, int]]:
    """Parse timestamps, drop unusable rows, de-duplicate and sort.

    Args:
        frame: Raw dataframe.
        cfg: Merged configuration.

    Returns:
        ``(cleaned_frame, stats)`` where ``stats`` counts what was removed. The
        caller is expected to log/persist ``stats`` rather than discard it.
    """
    out = frame.copy()
    out[TIMESTAMP] = pd.to_datetime(out[TIMESTAMP], errors="coerce")
    out[TURBINE_ID] = out[TURBINE_ID].astype("string")

    stats = {"input_rows": len(out)}

    if bool(cfg.get("preprocessing.drop_rows_missing_keys", True)):
        bad_keys = out[TIMESTAMP].isna() | out[TURBINE_ID].isna()
        stats["dropped_missing_keys"] = int(bad_keys.sum())
        out = out.loc[~bad_keys]
    else:
        stats["dropped_missing_keys"] = 0

    before = len(out)
    out = out.drop_duplicates(subset=[TURBINE_ID, TIMESTAMP], keep="first")
    stats["dropped_duplicates"] = before - len(out)

    out = out.sort_values([TURBINE_ID, TIMESTAMP]).reset_index(drop=True)
    out[TURBINE_ID] = out[TURBINE_ID].astype(str)
    stats["output_rows"] = len(out)

    logger.info("Normalised keys", extra=stats)
    return out, stats


def treat_invalid_values(frame: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replace physically impossible readings with NaN, then optionally clip.

    Out-of-range values are treated as sensor faults rather than as extreme but
    real observations: they become NaN so the imputation step handles them, and
    the count is reported.

    Args:
        frame: Frame after key normalisation.
        cfg: Merged configuration.

    Returns:
        ``(frame, per_column_invalid_counts)``.
    """
    out = frame.copy()
    ranges = cfg.get("validation.physical_ranges") or {}
    counts: dict[str, int] = {}

    for column, bounds in dict(ranges).items():
        if column not in out.columns:
            continue
        low, high = float(bounds[0]), float(bounds[1])
        values = pd.to_numeric(out[column], errors="coerce")
        invalid = (values < low) | (values > high)
        n_invalid = int(invalid.sum())
        if n_invalid:
            counts[column] = n_invalid
            values = values.mask(invalid, np.nan)
        out[column] = values

    if bool(cfg.get("preprocessing.clip_to_physical_ranges", True)):
        for column, bounds in dict(ranges).items():
            if column in out.columns:
                out[column] = out[column].clip(float(bounds[0]), float(bounds[1]))

    if counts:
        logger.warning(
            "Replaced physically implausible readings with NaN",
            extra={"total": sum(counts.values()), "columns": counts},
        )
    return out, counts


def impute_missing(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Fill missing sensor values using past-only information per turbine.

    Strategy:

    * Slow-moving channels listed in ``preprocessing.ffill_columns`` are forward
      filled within the turbine, limited to ``preprocessing.ffill_limit`` steps.
      This is justified by physical inertia: a gearbox temperature does not
      change materially over a few samples.
    * Everything still missing is filled with that turbine's median.  The median
      is a *global* statistic over the turbine's history, which technically uses
      future rows.  This is accepted deliberately because it is a scalar
      location estimate over a year of data, not an observation-level signal;
      the model-facing consequence is negligible and the alternative (expanding
      median) leaves the first rows unusable.  This choice is recorded in
      MODEL_CARD.md under limitations.
    * Any residual NaN (a turbine with a fully missing channel) is filled with
      the fleet median, then with 0.

    Args:
        frame: Frame after invalid-value treatment.
        cfg: Merged configuration.

    Returns:
        A frame with no missing values in the sensor columns.
    """
    out = frame.copy()
    sensors = [column for column in cfg.require("features.raw_sensors") if column in out.columns]
    ffill_columns = [
        column for column in cfg.get("preprocessing.ffill_columns", []) if column in sensors
    ]
    limit = int(cfg.get("preprocessing.ffill_limit", 3))

    grouped = out.groupby(TURBINE_ID, sort=False)
    if ffill_columns:
        out[ffill_columns] = grouped[ffill_columns].ffill(limit=limit)

    remaining = [column for column in sensors if out[column].isna().any()]
    if remaining:
        medians = out.groupby(TURBINE_ID, sort=False)[remaining].transform("median")
        out[remaining] = out[remaining].fillna(medians)
        out[remaining] = out[remaining].fillna(out[remaining].median())
        out[remaining] = out[remaining].fillna(0.0)

    if OPERATIONAL_STATUS in out.columns:
        out[OPERATIONAL_STATUS] = (
            out[OPERATIONAL_STATUS].astype("string").fillna(str(OperationalStatus.NORMAL)).astype(str)
        )
    return out


def preprocess(frame: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run the full cleaning pipeline.

    Args:
        frame: Raw ingested dataframe.
        cfg: Merged configuration.

    Returns:
        ``(clean_frame, stats)``.
    """
    out, key_stats = normalise_keys(frame, cfg)
    out, invalid_counts = treat_invalid_values(out, cfg)
    out = impute_missing(out, cfg)
    stats: dict[str, object] = dict(key_stats)
    stats["invalid_values_by_column"] = invalid_counts
    stats["invalid_values_total"] = int(sum(invalid_counts.values()))
    return out, stats


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
def create_failure_target(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Attach the ``failure_within_48h`` label.

    Definition
    ----------
    For observation *t* of turbine *k*, the label is 1 when at least one
    ``failure_event == 1`` occurs for turbine *k* in the half-open interval
    ``(t, t + horizon]``.  The failure timestamp itself is excluded from the
    window (a failure at *t* is not "within the next 48 hours" of *t*; it has
    already happened).

    Documented edge cases
    ---------------------
    * **Per turbine.** The look-ahead never crosses a turbine boundary; it is
      computed inside ``groupby(turbine_id)``.
    * **End of timeline.** The last ``horizon`` hours of each turbine's history
      cannot be labelled reliably - a failure just past the end of the record
      would be invisible.  Those rows are flagged in ``label_reliable`` and
      dropped from the modelling set, which prevents systematic false negatives
      at the tail.
    * **During fault / maintenance.** Rows whose ``operational_status`` is in
      ``target.exclude_states`` are dropped: at that point the failure is
      already known to the operator, so keeping them would leak the outcome and
      inflate measured performance.
    * **After a repair.** ``target.post_repair_exclusion_hours`` rows following
      the end of a maintenance window are dropped, because the machine is in a
      post-maintenance transient that does not represent normal operation.
    * **Repeated labelling of one failure.** A single failure legitimately
      labels the ``horizon`` observations preceding it; that is the definition
      of the task, not duplication. Degradation state is reset to zero by the
      repair, so consecutive failures produce disjoint positive windows.

    Args:
        frame: Preprocessed frame sorted by turbine and time.
        cfg: Merged configuration.

    Returns:
        A copy of ``frame`` with ``failure_within_48h``, ``label_reliable`` and
        ``modelling_eligible`` columns added.
    """
    target_name = str(cfg.require("target.name"))
    horizon_hours = float(cfg.require("target.horizon_hours"))
    out = frame.sort_values([TURBINE_ID, TIMESTAMP]).reset_index(drop=True).copy()
    out[TIMESTAMP] = pd.to_datetime(out[TIMESTAMP])

    horizon = pd.Timedelta(hours=horizon_hours)
    labels = np.zeros(len(out), dtype=np.int8)
    reliable = np.ones(len(out), dtype=bool)

    events = pd.to_numeric(out[FAILURE_EVENT], errors="coerce").fillna(0).to_numpy()

    for _, positions in out.groupby(TURBINE_ID, sort=False).indices.items():
        positions = np.asarray(positions)
        times = out[TIMESTAMP].to_numpy()[positions]
        failure_times = times[events[positions] == 1]

        if failure_times.size:
            # searchsorted on the sorted failure times: is there a failure in
            # (t, t + horizon]?
            lower = np.searchsorted(failure_times, times, side="right")
            upper = np.searchsorted(failure_times, times + horizon, side="right")
            labels[positions] = (upper > lower).astype(np.int8)

        # Tail of the turbine's record cannot be labelled reliably.
        reliable[positions] = times <= (times[-1] - horizon)

    out[target_name] = labels
    out["label_reliable"] = reliable

    excluded_states = {str(state) for state in cfg.get("target.exclude_states", [])}
    state_ok = (
        ~out[OPERATIONAL_STATUS].astype(str).isin(excluded_states)
        if OPERATIONAL_STATUS in out.columns and excluded_states
        else pd.Series(True, index=out.index)
    )

    post_repair_ok = _post_repair_mask(out, cfg)
    out["modelling_eligible"] = out["label_reliable"] & state_ok & post_repair_ok

    logger.info(
        "Created failure target",
        extra={
            "target": target_name,
            "horizon_hours": horizon_hours,
            "positives": int(labels.sum()),
            "eligible_rows": int(out["modelling_eligible"].sum()),
            "dropped_unreliable_tail": int((~reliable).sum()),
            "dropped_excluded_state": int((~state_ok).sum()),
            "dropped_post_repair": int((~post_repair_ok).sum()),
        },
    )
    return out


def _post_repair_mask(frame: pd.DataFrame, cfg: Config) -> pd.Series:
    """Return ``False`` for rows inside a post-repair settling window."""
    hours = float(cfg.get("target.post_repair_exclusion_hours", 0))
    if hours <= 0 or OPERATIONAL_STATUS not in frame.columns:
        return pd.Series(True, index=frame.index)

    step_hours = infer_step_hours(frame)
    steps = hours_to_steps(hours, step_hours)
    in_maintenance = frame[OPERATIONAL_STATUS].astype(str).isin(
        {str(OperationalStatus.MAINTENANCE), str(OperationalStatus.FAULT)}
    )
    # A row is excluded when any of the previous `steps` rows (same turbine)
    # were in maintenance/fault.
    recent = (
        in_maintenance.astype(int)
        .groupby(frame[TURBINE_ID].to_numpy(), sort=False)
        .transform(lambda s: s.shift(1).rolling(steps, min_periods=1).max())
        .fillna(0)
    )
    return recent == 0


def apply_modelling_filter(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows eligible for supervised modelling.

    Args:
        frame: Frame produced by :func:`create_failure_target`.

    Returns:
        The filtered frame with the helper flag columns removed.

    Raises:
        KeyError: If :func:`create_failure_target` was not run first.
    """
    if "modelling_eligible" not in frame.columns:
        raise KeyError("apply_modelling_filter requires create_failure_target to have been run")
    filtered = frame.loc[frame["modelling_eligible"]].copy()
    return filtered.drop(columns=["modelling_eligible", "label_reliable"])
