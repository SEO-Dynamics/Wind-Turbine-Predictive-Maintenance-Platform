"""Static figure tests for Stage 3 evaluation and policy documentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wind_turbine_pm.anomaly.evaluation import (
    plot_anomaly_calibration,
    plot_anomaly_model_comparison,
    plot_maintenance_policy,
)
from wind_turbine_pm.anomaly.modeling import fit_calibration
from wind_turbine_pm.config import Config


def _assert_png(path: Path) -> None:
    assert path.is_file()
    assert path.read_bytes().startswith(b"\x89PNG")
    assert path.stat().st_size > 1_000


def test_anomaly_model_comparison_figure(tmp_path: Path) -> None:
    comparison = pd.DataFrame(
        [
            {
                "model": "isolation_forest",
                "selected": False,
                "pr_auc": 0.31,
                "recall": 0.42,
                "precision": 0.25,
                "f2": 0.37,
            },
            {
                "model": "local_outlier_factor",
                "selected": True,
                "pr_auc": 0.73,
                "recall": 0.69,
                "precision": 0.66,
                "f2": 0.68,
            },
        ]
    )
    target = plot_anomaly_model_comparison(comparison, tmp_path / "comparison.png")
    _assert_png(target)


def test_anomaly_calibration_figure(tmp_path: Path) -> None:
    calibration = fit_calibration(np.linspace(-0.6, 1.2, 1_000), 0.05, 0.01)
    target = plot_anomaly_calibration(calibration, tmp_path / "calibration.png")
    _assert_png(target)


def test_maintenance_policy_figure(tmp_path: Path) -> None:
    cfg = Config(
        {
            "maintenance": {
                "weights": {"failure": 0.5, "anomaly": 0.3, "health": 0.2},
                "actions": {
                    "planned_inspection_hours": 168,
                    "urgent_review_hours": 48,
                    "immediate_review_hours": 8,
                },
            }
        }
    )
    target = plot_maintenance_policy(cfg, tmp_path / "policy.png")
    _assert_png(target)
