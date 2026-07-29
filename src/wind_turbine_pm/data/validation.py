"""Reusable data-validation layer.

Validation never mutates the frame and never silently drops records.  It
returns a :class:`ValidationReport` holding structured findings plus summary
statistics; callers decide whether to stop (``report.raise_for_errors()``) or to
continue and let preprocessing repair what it can.

The layer is written to be reused by the Health Monitoring and Anomaly
Detection modules: :func:`validate_scada_frame` covers the shared raw contract,
and :func:`validate_target` is the failure-prediction-specific addition.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import (
    FAILURE_EVENT,
    OPERATIONAL_STATUS,
    REQUIRED_RAW_COLUMNS,
    TIMESTAMP,
    TURBINE_ID,
    ValidationSeverity,
)
from wind_turbine_pm.logging_config import get_logger

logger = get_logger(__name__)


class DataValidationError(ValueError):
    """Raised when validation produced findings of ``ERROR`` severity."""

    def __init__(self, findings: Sequence[Finding]) -> None:
        """Store the failing findings.

        Args:
            findings: The error-severity findings.
        """
        self.findings = list(findings)
        joined = "\n  - ".join(finding.message for finding in self.findings)
        super().__init__(f"Data validation failed with {len(self.findings)} error(s):\n  - {joined}")


@dataclass(frozen=True)
class Finding:
    """A single validation observation."""

    check: str
    severity: ValidationSeverity
    message: str
    count: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        payload = asdict(self)
        payload["severity"] = str(self.severity)
        return payload


@dataclass
class ValidationReport:
    """Structured, machine-readable outcome of a validation run."""

    findings: list[Finding] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def add(
        self,
        check: str,
        severity: ValidationSeverity,
        message: str,
        count: int = 0,
        **details: Any,
    ) -> None:
        """Record a finding and log it at a matching level.

        Args:
            check: Short identifier of the check.
            severity: Finding severity.
            message: Human-readable description.
            count: Number of affected records, when meaningful.
            **details: Extra structured context.
        """
        finding = Finding(check=check, severity=severity, message=message, count=count, details=details)
        self.findings.append(finding)
        level = {
            ValidationSeverity.ERROR: logger.error,
            ValidationSeverity.WARNING: logger.warning,
            ValidationSeverity.INFO: logger.info,
        }[severity]
        level(message, extra={"check": check, "count": count})

    @property
    def errors(self) -> list[Finding]:
        """Findings with ``ERROR`` severity."""
        return [f for f in self.findings if f.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        """Findings with ``WARNING`` severity."""
        return [f for f in self.findings if f.severity == ValidationSeverity.WARNING]

    @property
    def is_valid(self) -> bool:
        """``True`` when no error-severity findings were recorded."""
        return not self.errors

    def raise_for_errors(self) -> None:
        """Raise :class:`DataValidationError` if any error findings exist."""
        if self.errors:
            raise DataValidationError(self.errors)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable report document."""
        return {
            "is_valid": self.is_valid,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "summary": self.summary,
            "findings": [finding.to_dict() for finding in self.findings],
        }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def _check_required_columns(frame: pd.DataFrame, required: Iterable[str], report: ValidationReport) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        report.add(
            "required_columns",
            ValidationSeverity.ERROR,
            f"Missing required columns: {missing}",
            count=len(missing),
            missing=missing,
        )


def _check_keys(frame: pd.DataFrame, report: ValidationReport) -> None:
    if TURBINE_ID in frame.columns:
        n_missing = int(frame[TURBINE_ID].isna().sum())
        if n_missing:
            report.add(
                "missing_turbine_id",
                ValidationSeverity.WARNING,
                f"{n_missing} row(s) have no turbine_id and cannot be attributed",
                count=n_missing,
            )
    if TIMESTAMP in frame.columns:
        parsed = pd.to_datetime(frame[TIMESTAMP], errors="coerce")
        n_unparseable = int(parsed.isna().sum())
        if n_unparseable:
            report.add(
                "unparseable_timestamp",
                ValidationSeverity.WARNING,
                f"{n_unparseable} row(s) have a missing or unparseable timestamp",
                count=n_unparseable,
            )


def _check_duplicates(frame: pd.DataFrame, report: ValidationReport) -> None:
    if not {TURBINE_ID, TIMESTAMP}.issubset(frame.columns):
        return
    keyed = frame.dropna(subset=[TURBINE_ID, TIMESTAMP])
    n_duplicates = int(keyed.duplicated(subset=[TURBINE_ID, TIMESTAMP]).sum())
    if n_duplicates:
        report.add(
            "duplicate_records",
            ValidationSeverity.WARNING,
            f"{n_duplicates} duplicate (turbine_id, timestamp) record(s) found",
            count=n_duplicates,
        )


def _check_chronology(frame: pd.DataFrame, report: ValidationReport) -> None:
    if not {TURBINE_ID, TIMESTAMP}.issubset(frame.columns):
        return
    keyed = frame.dropna(subset=[TURBINE_ID, TIMESTAMP])
    if keyed.empty:
        return
    times = pd.to_datetime(keyed[TIMESTAMP], errors="coerce")
    out_of_order = (
        times.groupby(keyed[TURBINE_ID].to_numpy()).apply(lambda s: bool((s.diff().dropna() < pd.Timedelta(0)).any()))
    )
    offenders = [str(t) for t, flag in out_of_order.items() if flag]
    if offenders:
        report.add(
            "chronological_order",
            ValidationSeverity.WARNING,
            f"{len(offenders)} turbine(s) have out-of-order timestamps: {offenders[:5]}",
            count=len(offenders),
            turbines=offenders,
        )


def _check_missingness(frame: pd.DataFrame, cfg: Config, report: ValidationReport) -> None:
    error_threshold = float(cfg.get("validation.max_missing_fraction", 0.15))
    warn_threshold = float(cfg.get("validation.warn_missing_fraction", 0.05))
    sensors = [column for column in cfg.require("features.raw_sensors") if column in frame.columns]
    fractions = frame[sensors].isna().mean()
    for column, fraction in fractions.items():
        if fraction > error_threshold:
            report.add(
                "excessive_missingness",
                ValidationSeverity.ERROR,
                f"Column {column!r} is {fraction:.1%} missing (limit {error_threshold:.1%})",
                count=int(frame[column].isna().sum()),
                column=column,
                fraction=float(fraction),
            )
        elif fraction > warn_threshold:
            report.add(
                "elevated_missingness",
                ValidationSeverity.WARNING,
                f"Column {column!r} is {fraction:.1%} missing (warning at {warn_threshold:.1%})",
                count=int(frame[column].isna().sum()),
                column=column,
                fraction=float(fraction),
            )


def _check_physical_ranges(frame: pd.DataFrame, cfg: Config, report: ValidationReport) -> None:
    ranges = cfg.get("validation.physical_ranges")
    if ranges is None:
        return
    for column, bounds in dict(ranges).items():
        if column not in frame.columns:
            continue
        low, high = float(bounds[0]), float(bounds[1])
        values = pd.to_numeric(frame[column], errors="coerce")
        violations = int(((values < low) | (values > high)).sum())
        if violations:
            report.add(
                "physically_implausible_values",
                ValidationSeverity.WARNING,
                f"{violations} value(s) of {column!r} fall outside the plausible range [{low}, {high}]",
                count=violations,
                column=column,
                lower=low,
                upper=high,
            )


def _check_categoricals(frame: pd.DataFrame, cfg: Config, report: ValidationReport) -> None:
    if OPERATIONAL_STATUS not in frame.columns:
        return
    allowed = set(cfg.get("validation.allowed_operational_status", []))
    if not allowed:
        return
    observed = set(frame[OPERATIONAL_STATUS].dropna().astype(str).unique())
    unexpected = sorted(observed - allowed)
    if unexpected:
        report.add(
            "unexpected_categorical",
            ValidationSeverity.ERROR,
            f"Unexpected {OPERATIONAL_STATUS} value(s): {unexpected}",
            count=len(unexpected),
            values=unexpected,
        )


def _check_turbine_history(frame: pd.DataFrame, cfg: Config, report: ValidationReport) -> None:
    if TURBINE_ID not in frame.columns:
        return
    minimum = int(cfg.get("validation.min_rows_per_turbine", 0))
    counts = frame[TURBINE_ID].value_counts(dropna=True)
    short = counts[counts < minimum]
    if not short.empty:
        report.add(
            "insufficient_history",
            ValidationSeverity.WARNING,
            f"{len(short)} turbine(s) have fewer than {minimum} observations: {list(short.index[:5])}",
            count=len(short),
            turbines=[str(t) for t in short.index],
        )

    if FAILURE_EVENT in frame.columns:
        per_turbine = frame.groupby(TURBINE_ID, dropna=True)[FAILURE_EVENT].sum()
        no_events = [str(t) for t, total in per_turbine.items() if total == 0]
        if no_events:
            report.add(
                "no_positive_events",
                ValidationSeverity.WARNING,
                f"{len(no_events)} turbine(s) recorded no failure events: {no_events[:5]}",
                count=len(no_events),
                turbines=no_events,
            )


def _check_event_column(frame: pd.DataFrame, report: ValidationReport) -> None:
    if FAILURE_EVENT not in frame.columns:
        return
    values = pd.to_numeric(frame[FAILURE_EVENT], errors="coerce")
    invalid = int((~values.isin([0, 1]) & values.notna()).sum())
    if invalid:
        report.add(
            "invalid_failure_event",
            ValidationSeverity.ERROR,
            f"{invalid} failure_event value(s) are not 0/1",
            count=invalid,
        )
    if float(values.fillna(0).sum()) == 0.0:
        report.add(
            "no_failures",
            ValidationSeverity.ERROR,
            "Dataset contains no failure events; a supervised failure model cannot be trained",
        )


def _summarise(frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
    }
    if TURBINE_ID in frame.columns:
        summary["n_turbines"] = int(frame[TURBINE_ID].nunique(dropna=True))
    if TIMESTAMP in frame.columns:
        times = pd.to_datetime(frame[TIMESTAMP], errors="coerce")
        summary["time_start"] = None if times.isna().all() else str(times.min())
        summary["time_end"] = None if times.isna().all() else str(times.max())
    if FAILURE_EVENT in frame.columns:
        events = pd.to_numeric(frame[FAILURE_EVENT], errors="coerce").fillna(0)
        summary["n_failure_events"] = int(events.sum())
    numeric = frame.select_dtypes(include=[np.number])
    if not numeric.empty:
        summary["missing_fraction_by_column"] = {
            column: round(float(value), 6) for column, value in numeric.isna().mean().items()
        }
    return summary


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def validate_scada_frame(
    frame: pd.DataFrame,
    cfg: Config,
    required_columns: Sequence[str] | None = None,
) -> ValidationReport:
    """Validate a raw SCADA frame against the shared data contract.

    Args:
        frame: The raw dataframe.
        cfg: Merged configuration.
        required_columns: Override for the required-column list. Defaults to
            :data:`wind_turbine_pm.constants.REQUIRED_RAW_COLUMNS`.

    Returns:
        A populated :class:`ValidationReport` (this function never raises for
        data problems; call :meth:`ValidationReport.raise_for_errors`).
    """
    report = ValidationReport()
    required = REQUIRED_RAW_COLUMNS if required_columns is None else required_columns

    _check_required_columns(frame, required, report)
    if report.errors:
        report.summary = _summarise(frame)
        return report

    _check_keys(frame, report)
    _check_duplicates(frame, report)
    _check_chronology(frame, report)
    _check_missingness(frame, cfg, report)
    _check_physical_ranges(frame, cfg, report)
    _check_categoricals(frame, cfg, report)
    _check_turbine_history(frame, cfg, report)
    _check_event_column(frame, report)

    report.summary = _summarise(frame)
    logger.info(
        "Validation complete",
        extra={"errors": len(report.errors), "warnings": len(report.warnings)},
    )
    return report


def validate_target(frame: pd.DataFrame, target_column: str, report: ValidationReport | None = None) -> ValidationReport:
    """Validate a binary target column.

    Args:
        frame: Frame containing the target.
        target_column: Name of the target column.
        report: Existing report to append to; a new one is created when absent.

    Returns:
        The report with target findings appended.
    """
    report = report or ValidationReport()

    if target_column not in frame.columns:
        report.add(
            "target_missing",
            ValidationSeverity.ERROR,
            f"Target column {target_column!r} is absent",
        )
        return report

    values = pd.to_numeric(frame[target_column], errors="coerce")
    n_null = int(values.isna().sum())
    if n_null:
        report.add(
            "target_nulls",
            ValidationSeverity.ERROR,
            f"{n_null} row(s) have a null {target_column}",
            count=n_null,
        )

    invalid = int((~values.isin([0, 1]) & values.notna()).sum())
    if invalid:
        report.add(
            "target_invalid_values",
            ValidationSeverity.ERROR,
            f"{invalid} {target_column} value(s) are not 0/1",
            count=invalid,
        )

    n_positive = int(values.fillna(0).sum())
    if n_positive == 0:
        report.add(
            "target_no_positives",
            ValidationSeverity.ERROR,
            f"{target_column} contains no positive labels",
        )
    else:
        rate = n_positive / max(len(values), 1)
        report.summary["target_positive_rate"] = round(float(rate), 6)
        report.summary["target_positive_count"] = n_positive
        if rate > 0.5:
            report.add(
                "target_majority_positive",
                ValidationSeverity.WARNING,
                f"{target_column} positive rate is {rate:.1%}, which is unexpected for a rare event",
                count=n_positive,
            )
    return report
