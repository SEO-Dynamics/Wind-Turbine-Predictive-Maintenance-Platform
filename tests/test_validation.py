"""Tests for the data-validation layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wind_turbine_pm.constants import (
    FAILURE_EVENT,
    OPERATIONAL_STATUS,
    TARGET_COLUMN,
    TIMESTAMP,
    TURBINE_ID,
    ValidationSeverity,
)
from wind_turbine_pm.data.validation import (
    DataValidationError,
    ValidationReport,
    validate_scada_frame,
    validate_target,
)


def _check_names(report: ValidationReport) -> set[str]:
    return {finding.check for finding in report.findings}


def test_clean_data_passes(clean_frame, clean_config):
    """Defect-free data must produce no error-severity findings."""
    report = validate_scada_frame(clean_frame, clean_config)
    assert report.is_valid
    assert report.errors == []


def test_missing_required_column_is_an_error(clean_frame, clean_config):
    """A missing contract column must be an error, not a warning."""
    report = validate_scada_frame(clean_frame.drop(columns=["vibration"]), clean_config)
    assert not report.is_valid
    assert "required_columns" in _check_names(report)


def test_injected_defects_are_all_detected(raw_frame, small_config):
    """Every deliberately injected defect must be surfaced."""
    report = validate_scada_frame(raw_frame, small_config)
    checks = _check_names(report)
    assert "missing_turbine_id" in checks
    assert "unparseable_timestamp" in checks
    assert "duplicate_records" in checks
    assert "physically_implausible_values" in checks


def test_out_of_range_values_are_flagged(clean_frame, clean_config):
    """Physically impossible readings must be reported with a count."""
    frame = clean_frame.copy()
    frame.loc[frame.index[:5], "gearbox_temperature"] = 999.0
    report = validate_scada_frame(frame, clean_config)
    findings = [f for f in report.findings if f.check == "physically_implausible_values"]
    assert findings
    assert any(f.details.get("column") == "gearbox_temperature" for f in findings)


def test_excessive_missingness_is_an_error(clean_frame, clean_config):
    """Missingness beyond the configured limit must be an error."""
    frame = clean_frame.copy()
    frame.loc[frame.index[: int(len(frame) * 0.4)], "vibration"] = np.nan
    report = validate_scada_frame(frame, clean_config)
    assert not report.is_valid
    assert "excessive_missingness" in _check_names(report)


def test_unexpected_categorical_is_an_error(clean_frame, clean_config):
    """An unknown operational status must be an error."""
    frame = clean_frame.copy()
    frame.loc[frame.index[0], OPERATIONAL_STATUS] = "exploded"
    report = validate_scada_frame(frame, clean_config)
    assert not report.is_valid
    assert "unexpected_categorical" in _check_names(report)


def test_no_failures_is_an_error(clean_frame, clean_config):
    """A dataset with no positive events cannot train a supervised model."""
    frame = clean_frame.copy()
    frame[FAILURE_EVENT] = 0
    report = validate_scada_frame(frame, clean_config)
    assert not report.is_valid
    assert "no_failures" in _check_names(report)


def test_turbine_with_no_events_is_a_warning(clean_frame, clean_config):
    """A single turbine without events is worth flagging but not fatal."""
    frame = clean_frame.copy()
    frame.loc[frame[TURBINE_ID] == "T01", FAILURE_EVENT] = 0
    report = validate_scada_frame(frame, clean_config)
    assert "no_positive_events" in _check_names(report)


def test_insufficient_history_is_a_warning(clean_frame, clean_config):
    """A turbine with too little history must be reported."""
    frame = pd.concat(
        [clean_frame, clean_frame.head(3).assign(**{TURBINE_ID: "T99"})], ignore_index=True
    )
    report = validate_scada_frame(frame, clean_config)
    assert "insufficient_history" in _check_names(report)


def test_out_of_order_timestamps_are_detected(clean_frame, clean_config):
    """Non-chronological records within a turbine must be flagged."""
    frame = clean_frame.copy()
    turbine_rows = frame.index[frame[TURBINE_ID] == "T01"][:10]
    frame.loc[turbine_rows, TIMESTAMP] = frame.loc[turbine_rows, TIMESTAMP].to_numpy()[::-1]
    report = validate_scada_frame(frame, clean_config)
    assert "chronological_order" in _check_names(report)


def test_report_is_json_serialisable(raw_frame, small_config):
    """The report must round-trip through JSON for machine consumption."""
    import json

    report = validate_scada_frame(raw_frame, small_config)
    payload = json.loads(json.dumps(report.to_dict(), default=str))
    assert "findings" in payload
    assert "summary" in payload
    assert payload["summary"]["n_rows"] == len(raw_frame)
    assert isinstance(payload["n_errors"], int)


def test_raise_for_errors(clean_frame, clean_config):
    """``raise_for_errors`` must raise only when errors exist."""
    good = validate_scada_frame(clean_frame, clean_config)
    good.raise_for_errors()  # must not raise

    bad = validate_scada_frame(clean_frame.drop(columns=["vibration"]), clean_config)
    with pytest.raises(DataValidationError):
        bad.raise_for_errors()


def test_validate_target_accepts_valid_labels():
    """A binary target with positives must validate cleanly."""
    frame = pd.DataFrame({TARGET_COLUMN: [0, 1, 0, 0, 1]})
    report = validate_target(frame, TARGET_COLUMN)
    assert report.is_valid
    assert report.summary["target_positive_count"] == 2


def test_validate_target_rejects_non_binary():
    """Non-binary target values must be an error."""
    frame = pd.DataFrame({TARGET_COLUMN: [0, 1, 5]})
    report = validate_target(frame, TARGET_COLUMN)
    assert not report.is_valid


def test_validate_target_rejects_all_zero():
    """A target with no positives must be an error."""
    frame = pd.DataFrame({TARGET_COLUMN: [0, 0, 0]})
    report = validate_target(frame, TARGET_COLUMN)
    assert not report.is_valid
    assert any(f.check == "target_no_positives" for f in report.findings)


def test_validate_target_missing_column():
    """An absent target column must be reported as an error."""
    report = validate_target(pd.DataFrame({"x": [1]}), TARGET_COLUMN)
    assert not report.is_valid


def test_severity_levels_are_used(raw_frame, small_config):
    """Findings must carry proper severity enum values."""
    report = validate_scada_frame(raw_frame, small_config)
    severities = {finding.severity for finding in report.findings}
    assert severities <= {
        ValidationSeverity.INFO,
        ValidationSeverity.WARNING,
        ValidationSeverity.ERROR,
    }
