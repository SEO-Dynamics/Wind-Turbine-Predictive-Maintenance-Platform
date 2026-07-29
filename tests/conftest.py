"""Shared pytest fixtures.

Fixtures build a **small** synthetic dataset (a few turbines, a few weeks) so
the whole suite runs in seconds without production-scale training.  Everything
is seeded, so every test is deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Support running the suite from a checkout without `pip install -e .`.
_ROOT = Path(__file__).resolve().parent.parent
for candidate in (_ROOT, _ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from wind_turbine_pm.config import Config, load_config  # noqa: E402
from wind_turbine_pm.constants import TARGET_COLUMN, TIMESTAMP, TURBINE_ID  # noqa: E402
from wind_turbine_pm.data.preprocessing import (  # noqa: E402
    apply_modelling_filter,
    create_failure_target,
    preprocess,
)
from wind_turbine_pm.data.synthetic import generate_scada_dataset  # noqa: E402
from wind_turbine_pm.features.failure_features import build_failure_features  # noqa: E402


@pytest.fixture(scope="session")
def base_config() -> Config:
    """The project configuration exactly as shipped."""
    return load_config()


@pytest.fixture(scope="session")
def small_config(base_config: Config) -> Config:
    """A configuration scaled down so tests run fast but stay realistic.

    Four turbines over roughly six weeks, with shorter degradation ramps so
    several failures still occur inside the window.
    """
    data = base_config.to_dict()
    data["synthetic"].update(
        {
            "n_turbines": 4,
            "months": 2,
            "freq": "1h",
            "seed": 7,
            "failures_per_turbine_per_year": 26.0,
            "degradation_hours_min": 48,
            "degradation_hours_max": 96,
            "repair_hours_min": 4,
            "repair_hours_max": 10,
            "recovery_hours": 12,
            "benign_degradation_ratio": 0.4,
            "scheduled_maintenance_per_year": 6.0,
        }
    )
    data["random_seed"] = 7
    # Narrow the feature expansion so fixtures stay small and fast.
    data["features"]["dynamic_sensors"] = [
        "vibration",
        "gearbox_temperature",
        "oil_pressure",
        "power_output",
    ]
    data["features"]["lags_hours"] = [1, 3, 6]
    data["features"]["rolling"]["windows_hours"] = [3, 6, 12]
    data["features"]["trend"]["slope_windows_hours"] = [6]
    data["features"]["trend"]["diff_hours"] = [1, 6]
    data["features"]["trend"]["pct_change_hours"] = [6]
    data["features"]["turbine_relative"]["min_periods"] = 12
    data["validation"]["min_rows_per_turbine"] = 100
    data["split"] = {"train_end_fraction": 0.6, "valid_end_fraction": 0.8, "embargo_hours": 48}
    data["serving"]["min_history_hours"] = 24
    return Config(data)


@pytest.fixture(scope="session")
def clean_config(small_config: Config) -> Config:
    """Like :func:`small_config` but with no injected data-quality defects."""
    data = small_config.to_dict()
    data["synthetic"].update({"missing_rate": 0.0, "invalid_rate": 0.0, "duplicate_rate": 0.0})
    return Config(data)


@pytest.fixture(scope="session")
def raw_frame(small_config: Config) -> pd.DataFrame:
    """A small raw dataset including deliberately injected data defects."""
    return generate_scada_dataset(small_config, inject_issues=True)


@pytest.fixture(scope="session")
def clean_frame(clean_config: Config) -> pd.DataFrame:
    """A small raw dataset with no injected defects."""
    return generate_scada_dataset(clean_config, inject_issues=False)


@pytest.fixture(scope="session")
def prepared(small_config: Config, raw_frame: pd.DataFrame):
    """Preprocessed, labelled data with its feature matrix.

    Returns:
        ``(labelled_frame, feature_matrix, eligible_frame)``.
    """
    processed, _ = preprocess(raw_frame, small_config)
    labelled = create_failure_target(processed, small_config)
    features, _ = build_failure_features(labelled, small_config)
    eligible = apply_modelling_filter(labelled)
    return labelled, features, eligible


@pytest.fixture(scope="session")
def feature_matrix(prepared) -> pd.DataFrame:
    """The small feature matrix."""
    return prepared[1]


@pytest.fixture(scope="session")
def labelled_frame(prepared) -> pd.DataFrame:
    """The small labelled frame (before eligibility filtering)."""
    return prepared[0]


@pytest.fixture
def toy_frame() -> pd.DataFrame:
    """A tiny hand-built two-turbine frame with known failure positions.

    Turbine ``A`` fails at hour 100; turbine ``B`` never fails.  Small enough
    that target and leakage assertions can be reasoned about by hand.
    """
    rows = []
    start = pd.Timestamp("2024-03-01")
    rng = np.random.default_rng(0)
    for turbine, failure_hour in (("A", 100), ("B", None)):
        for hour in range(160):
            rows.append(
                {
                    TIMESTAMP: start + pd.Timedelta(hours=hour),
                    TURBINE_ID: turbine,
                    "wind_speed": 8.0 + rng.normal(0, 0.4),
                    "rotor_speed": 12.0 + rng.normal(0, 0.2),
                    "generator_speed": 1160.0 + rng.normal(0, 8),
                    "power_output": 700.0 + rng.normal(0, 20),
                    "generator_temperature": 60.0 + rng.normal(0, 0.5),
                    "gearbox_temperature": 54.0 + rng.normal(0, 0.5),
                    "bearing_temperature": 47.0 + rng.normal(0, 0.5),
                    "oil_temperature": 45.0 + rng.normal(0, 0.5),
                    "oil_pressure": 5.0 + rng.normal(0, 0.05),
                    "vibration": 3.0 + rng.normal(0, 0.1),
                    "ambient_temperature": 10.0 + rng.normal(0, 0.3),
                    "nacelle_temperature": 18.0 + rng.normal(0, 0.3),
                    "hydraulic_pressure": 190.0 + rng.normal(0, 2),
                    "brake_temperature": 16.0 + rng.normal(0, 0.3),
                    "operational_status": "normal",
                    "failure_event": int(failure_hour is not None and hour == failure_hour),
                    "maintenance_event": 0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def trained_bundle(small_config: Config, prepared):
    """A model trained on the small fixture, plus its splits.

    Returns:
        ``(estimator, feature_names, threshold_result, splits)``.
    """
    from wind_turbine_pm.constants import SPLIT_COLUMN
    from wind_turbine_pm.data.splitting import assign_splits, compute_boundaries
    from wind_turbine_pm.models.training import to_split_data, train_candidates

    _, features, eligible = prepared
    boundaries = compute_boundaries(eligible, small_config)
    eligible = eligible.copy()
    eligible[SPLIT_COLUMN] = assign_splits(eligible, boundaries)

    data = small_config.to_dict()
    # Only two fast candidates are needed to exercise the training path.
    data["training"]["candidates"] = {
        "dummy": {"enabled": True, "params": {"strategy": "stratified"}},
        "logistic_regression": {
            "enabled": True,
            "params": {"C": 0.5, "max_iter": 300, "class_weight": "balanced", "solver": "lbfgs"},
        },
    }
    data["training"]["calibration"] = "none"
    fast_config = Config(data)

    splits = {
        name: to_split_data(eligible.loc[eligible[SPLIT_COLUMN] == name], features, TARGET_COLUMN)
        for name in ("train", "valid", "test")
    }
    candidates = train_candidates(splits["train"], splits["valid"], fast_config)
    winner = next(c for c in candidates if c.name == "logistic_regression")
    return winner.estimator, list(features.columns), winner.threshold, splits
