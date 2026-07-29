"""Low-level, leakage-safe temporal primitives.

Every function here takes the *grouping key* explicitly and applies its
operation inside that group.  None of them ever reads a row with a larger index
than the one being computed, which is what makes the resulting features safe to
use with a forward-looking label.

The rolling helpers use pandas' default *trailing* windows (the window ends at
the current row, inclusive), so a value at position *t* is a function of rows
``[t - w + 1, t]`` only.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _grouped(series: pd.Series, groups: pd.Series):
    """Return a groupby handle that preserves the original index order."""
    return series.groupby(groups, sort=False, observed=True)


def grouped_lag(series: pd.Series, groups: pd.Series, periods: int) -> pd.Series:
    """Shift a series backwards in time within each group.

    Args:
        series: Values to lag.
        groups: Group key aligned to ``series``.
        periods: Number of rows to shift (positive = look into the past).

    Returns:
        The lagged series, NaN for the first ``periods`` rows of each group.
    """
    return _grouped(series, groups).shift(periods)


def grouped_diff(series: pd.Series, groups: pd.Series, periods: int) -> pd.Series:
    """Difference against the value ``periods`` rows earlier, within group.

    Args:
        series: Values to difference.
        groups: Group key aligned to ``series``.
        periods: Lookback in rows.

    Returns:
        ``series[t] - series[t - periods]`` within each group.
    """
    return _grouped(series, groups).diff(periods)


def grouped_pct_change(
    series: pd.Series, groups: pd.Series, periods: int, epsilon: float = 1e-6
) -> pd.Series:
    """Relative change over ``periods`` rows, numerically guarded.

    A plain percentage change explodes when the denominator approaches zero
    (idle turbines produce near-zero power and speed).  The denominator is
    therefore floored by ``epsilon`` in magnitude and the result is clipped.

    Args:
        series: Values to compare.
        groups: Group key aligned to ``series``.
        periods: Lookback in rows.
        epsilon: Minimum absolute denominator.

    Returns:
        The guarded relative change.
    """
    previous = grouped_lag(series, groups, periods)
    denominator = previous.abs().clip(lower=epsilon)
    change = (series - previous) / denominator
    return change.replace([np.inf, -np.inf], np.nan).clip(-50.0, 50.0)


def grouped_rolling(
    series: pd.Series,
    groups: pd.Series,
    window: int,
    stat: str,
    min_periods: int | None = None,
) -> pd.Series:
    """Compute a trailing rolling statistic within each group.

    Args:
        series: Values to aggregate.
        groups: Group key aligned to ``series``.
        window: Window length in rows (inclusive of the current row).
        stat: One of ``mean``, ``std``, ``min``, ``max``, ``median``, ``sum``,
            ``range``.
        min_periods: Minimum observations required; defaults to half the window.

    Returns:
        The rolling statistic.

    Raises:
        ValueError: If ``stat`` is not supported.
    """
    periods = min_periods if min_periods is not None else max(window // 2, 1)
    rolling = _grouped(series, groups).rolling(window=window, min_periods=periods)

    if stat == "range":
        result = rolling.max() - rolling.min()
    elif stat in {"mean", "std", "min", "max", "median", "sum"}:
        result = getattr(rolling, stat)()
    else:
        raise ValueError(f"Unsupported rolling statistic {stat!r}")

    # `groupby(...).rolling(...)` returns a MultiIndex; drop the group level.
    return result.reset_index(level=0, drop=True).reindex(series.index)


def rolling_slope(
    series: pd.Series, groups: pd.Series, window: int, min_periods: int | None = None
) -> pd.Series:
    """Ordinary-least-squares slope over a trailing window, per group.

    The slope is expressed in *units per row*.  It is computed with the closed
    form ``cov(x, y) / var(x)`` where ``x = 0..window-1`` is fixed, which makes
    it a cheap linear combination rather than a per-window regression fit.

    Args:
        series: Values to fit.
        groups: Group key aligned to ``series``.
        window: Window length in rows.
        min_periods: Minimum observations required; defaults to half the window.

    Returns:
        The trailing OLS slope.
    """
    periods = min_periods if min_periods is not None else max(window // 2, 2)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_centred = x - x_mean
    denominator = float((x_centred**2).sum())

    def _slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            mask = ~np.isnan(values)
            if mask.sum() < 2:
                return np.nan
            xs = x[-values.size :][mask]
            ys = values[mask]
            xs_centred = xs - xs.mean()
            local_denominator = float((xs_centred**2).sum())
            if local_denominator <= 0:
                return np.nan
            return float((xs_centred * (ys - ys.mean())).sum() / local_denominator)
        if values.size < window:
            xs = x[: values.size]
            xs_centred = xs - xs.mean()
            local_denominator = float((xs_centred**2).sum())
            if local_denominator <= 0:
                return np.nan
            return float((xs_centred * (values - values.mean())).sum() / local_denominator)
        return float((x_centred * (values - values.mean())).sum() / denominator)

    result = (
        _grouped(series, groups).rolling(window=window, min_periods=periods).apply(_slope, raw=True)
    )
    return result.reset_index(level=0, drop=True).reindex(series.index)


def expanding_baseline(
    series: pd.Series, groups: pd.Series, stat: str, min_periods: int = 24
) -> pd.Series:
    """Expanding statistic over strictly *past* rows within each group.

    The series is shifted by one row before expanding, so the current
    observation never contributes to its own baseline.  This matters: without
    the shift, an extreme reading would partly explain itself.

    Args:
        series: Values to aggregate.
        groups: Group key aligned to ``series``.
        stat: ``mean``, ``median`` or ``std``.
        min_periods: Minimum history required before a baseline is produced.

    Returns:
        The past-only expanding statistic.

    Raises:
        ValueError: If ``stat`` is not supported.
    """
    if stat not in {"mean", "median", "std"}:
        raise ValueError(f"Unsupported expanding statistic {stat!r}")
    shifted = _grouped(series, groups).shift(1)
    result = getattr(
        shifted.groupby(groups, sort=False, observed=True).expanding(min_periods=min_periods), stat
    )()
    return result.reset_index(level=0, drop=True).reindex(series.index)


def robust_zscore(
    series: pd.Series, groups: pd.Series, min_periods: int = 24, floor: float = 1e-6
) -> pd.Series:
    """Past-only robust z-score against each turbine's own history.

    Uses the expanding median as location and the expanding median absolute
    deviation (scaled by 1.4826) as scale, so a few extreme readings do not
    inflate the denominator the way a standard deviation would.

    Args:
        series: Values to standardise.
        groups: Group key aligned to ``series``.
        min_periods: Minimum history before a score is produced.
        floor: Lower bound on the scale, preventing division by zero.

    Returns:
        The robust z-score.
    """
    median = expanding_baseline(series, groups, "median", min_periods=min_periods)
    absolute_deviation = (series - median).abs()
    scale = 1.4826 * expanding_baseline(
        absolute_deviation, groups, "median", min_periods=min_periods
    )
    return (series - median) / scale.clip(lower=floor)


def cyclical_encode(values: pd.Series, period: float, name: str) -> pd.DataFrame:
    """Encode a cyclical quantity as sine/cosine components.

    Args:
        values: Raw cyclical values (e.g. hour of day).
        period: Length of one full cycle.
        name: Base name for the produced columns.

    Returns:
        A two-column frame ``{name}_sin`` / ``{name}_cos``.
    """
    angle = 2.0 * np.pi * values.astype(float) / period
    return pd.DataFrame(
        {f"{name}_sin": np.sin(angle), f"{name}_cos": np.cos(angle)}, index=values.index
    )


def assert_no_future_leakage(
    frame: pd.DataFrame,
    feature_fn,
    group_column: str,
    time_column: str,
    feature_columns: Sequence[str] | None = None,
    tolerance: float = 1e-8,
) -> None:
    """Verify that mutating the final row does not change earlier features.

    This is the empirical leakage check used by the test-suite: build features
    on the full frame, then rebuild them with the last row's sensor values
    perturbed.  Any feature at an earlier timestamp that moves is reading the
    future.

    Args:
        frame: Input frame.
        feature_fn: Callable mapping a frame to a feature frame with the same
            index.
        group_column: Grouping key column name.
        time_column: Timestamp column name.
        feature_columns: Subset of features to check; defaults to all numeric.
        tolerance: Absolute difference treated as equal.

    Raises:
        AssertionError: If any earlier feature value changes.
    """
    baseline = feature_fn(frame)
    perturbed_input = frame.copy()
    numeric = perturbed_input.select_dtypes(include=[np.number]).columns
    last_index = perturbed_input.index[-1]
    perturbed_input.loc[last_index, numeric] = (
        perturbed_input.loc[last_index, numeric].astype(float) + 1000.0
    )
    perturbed = feature_fn(perturbed_input)

    columns = list(feature_columns or baseline.select_dtypes(include=[np.number]).columns)
    earlier = baseline.index != last_index
    difference = (baseline.loc[earlier, columns] - perturbed.loc[earlier, columns]).abs()
    offenders = difference.max()[lambda s: s > tolerance]
    if not offenders.empty:
        raise AssertionError(
            f"Future leakage detected: perturbing the last row changed earlier values of "
            f"{list(offenders.index)[:10]} (group={group_column}, time={time_column})"
        )
