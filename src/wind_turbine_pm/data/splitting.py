"""Chronological train / validation / test splitting with embargo gaps.

A random split would be invalid here for two reasons: consecutive SCADA rows are
strongly autocorrelated, and the label looks 48 hours into the future.  Both
would let information about a test-period failure appear in training.

Strategy
--------
The global time axis is cut at two quantiles of the *distinct timestamps*
(``split.train_end_fraction`` and ``split.valid_end_fraction``).  Around each
cut an **embargo** of ``split.embargo_hours`` is removed from the data entirely.
The embargo must be at least the target horizon: an observation at
``train_end - 10h`` carries a label determined by events up to
``train_end + 38h``, which is validation-period information.  Dropping the
embargo band makes that impossible.

All three splits share the same wall-clock boundaries across turbines, so the
evaluation answers the operationally relevant question: *given everything known
up to time T, how well does the model do afterwards?*
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import SPLIT_COLUMN, TIMESTAMP, SplitName
from wind_turbine_pm.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SplitBoundaries:
    """Wall-clock boundaries produced by the chronological split."""

    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    embargo_hours: float

    def to_dict(self) -> dict[str, str | float]:
        """Return a JSON-serialisable representation."""
        return {
            "train_end": str(self.train_end),
            "valid_start": str(self.valid_start),
            "valid_end": str(self.valid_end),
            "test_start": str(self.test_start),
            "embargo_hours": self.embargo_hours,
        }


def compute_boundaries(frame: pd.DataFrame, cfg: Config) -> SplitBoundaries:
    """Derive split boundaries from the data's time axis.

    Args:
        frame: Frame with a datetime ``timestamp`` column.
        cfg: Merged configuration.

    Returns:
        The computed :class:`SplitBoundaries`.

    Raises:
        ValueError: If the frame has no usable timestamps, or if the configured
            embargo is shorter than the target horizon.
    """
    times = pd.to_datetime(frame[TIMESTAMP], errors="coerce").dropna()
    if times.empty:
        raise ValueError("Cannot split: no valid timestamps")

    embargo_hours = float(cfg.get("split.embargo_hours", 0.0))
    horizon_hours = float(cfg.get("target.horizon_hours", 0.0))
    if embargo_hours < horizon_hours:
        raise ValueError(
            f"split.embargo_hours ({embargo_hours}) must be >= target.horizon_hours "
            f"({horizon_hours}); a shorter embargo lets future labels leak across the boundary"
        )

    unique_times = pd.Series(times.unique()).sort_values()
    train_end = pd.Timestamp(unique_times.quantile(float(cfg.require("split.train_end_fraction"))))
    valid_end = pd.Timestamp(unique_times.quantile(float(cfg.require("split.valid_end_fraction"))))
    embargo = pd.Timedelta(hours=embargo_hours)

    boundaries = SplitBoundaries(
        train_end=train_end,
        valid_start=train_end + embargo,
        valid_end=valid_end,
        test_start=valid_end + embargo,
        embargo_hours=embargo_hours,
    )
    if not boundaries.train_end < boundaries.valid_start < boundaries.valid_end < boundaries.test_start:
        raise ValueError(
            "Split boundaries are degenerate; reduce split.embargo_hours or widen the "
            f"split fractions. Got {boundaries.to_dict()}"
        )
    return boundaries


def assign_splits(frame: pd.DataFrame, boundaries: SplitBoundaries) -> pd.Series:
    """Label each row with its split partition.

    Args:
        frame: Frame with a datetime ``timestamp`` column.
        boundaries: Boundaries from :func:`compute_boundaries`.

    Returns:
        A string series of split names, using ``"embargo"`` for the gap bands.
    """
    times = pd.to_datetime(frame[TIMESTAMP])
    labels = pd.Series(str(SplitName.EMBARGO), index=frame.index, dtype="object")
    labels[times <= boundaries.train_end] = str(SplitName.TRAIN)
    labels[(times >= boundaries.valid_start) & (times <= boundaries.valid_end)] = str(SplitName.VALID)
    labels[times >= boundaries.test_start] = str(SplitName.TEST)
    return labels


def temporal_split(
    frame: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitBoundaries]:
    """Split a frame chronologically into train, validation and test parts.

    Rows falling inside an embargo band are returned in none of the three
    frames.

    Args:
        frame: Frame with a datetime ``timestamp`` column.
        cfg: Merged configuration.

    Returns:
        ``(train, valid, test, boundaries)``.

    Raises:
        ValueError: If any split ends up empty.
    """
    boundaries = compute_boundaries(frame, cfg)
    labelled = frame.copy()
    labelled[SPLIT_COLUMN] = assign_splits(labelled, boundaries)

    train = labelled.loc[labelled[SPLIT_COLUMN] == str(SplitName.TRAIN)].drop(columns=[SPLIT_COLUMN])
    valid = labelled.loc[labelled[SPLIT_COLUMN] == str(SplitName.VALID)].drop(columns=[SPLIT_COLUMN])
    test = labelled.loc[labelled[SPLIT_COLUMN] == str(SplitName.TEST)].drop(columns=[SPLIT_COLUMN])

    empty = [name for name, part in (("train", train), ("valid", valid), ("test", test)) if part.empty]
    if empty:
        raise ValueError(f"Chronological split produced empty partition(s): {empty}")

    logger.info(
        "Chronological split complete",
        extra={
            "train_rows": len(train),
            "valid_rows": len(valid),
            "test_rows": len(test),
            "embargoed_rows": int((labelled[SPLIT_COLUMN] == str(SplitName.EMBARGO)).sum()),
            **boundaries.to_dict(),
        },
    )
    return train, valid, test, boundaries


def verify_split_integrity(
    train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, boundaries: SplitBoundaries
) -> None:
    """Assert that the split respects chronology and the embargo.

    Args:
        train: Training partition.
        valid: Validation partition.
        test: Test partition.
        boundaries: The boundaries used to build the split.

    Raises:
        ValueError: If ordering or embargo width is violated.
    """
    train_max = pd.to_datetime(train[TIMESTAMP]).max()
    valid_min = pd.to_datetime(valid[TIMESTAMP]).min()
    valid_max = pd.to_datetime(valid[TIMESTAMP]).max()
    test_min = pd.to_datetime(test[TIMESTAMP]).min()

    if not train_max < valid_min:
        raise ValueError(f"Training data ({train_max}) overlaps validation ({valid_min})")
    if not valid_max < test_min:
        raise ValueError(f"Validation data ({valid_max}) overlaps test ({test_min})")

    embargo = pd.Timedelta(hours=boundaries.embargo_hours)
    if (valid_min - train_max) < embargo:
        raise ValueError(
            f"Train/validation gap {valid_min - train_max} is narrower than the "
            f"required embargo {embargo}"
        )
    if (test_min - valid_max) < embargo:
        raise ValueError(
            f"Validation/test gap {test_min - valid_max} is narrower than the "
            f"required embargo {embargo}"
        )
