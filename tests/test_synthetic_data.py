"""Tests for the synthetic SCADA generator."""

from __future__ import annotations

import pandas as pd

from wind_turbine_pm.constants import (
    FAILURE_EVENT,
    REQUIRED_RAW_COLUMNS,
    TIMESTAMP,
    TURBINE_ID,
    OperationalStatus,
)
from wind_turbine_pm.data.synthetic import build_profiles, generate_scada_dataset, power_curve


def test_generation_is_deterministic(small_config):
    """The same configuration must produce byte-identical data."""
    first = generate_scada_dataset(small_config)
    second = generate_scada_dataset(small_config)
    pd.testing.assert_frame_equal(first, second)


def test_different_seed_changes_data(small_config):
    """A different seed must produce different data."""
    from wind_turbine_pm.config import Config

    other = small_config.to_dict()
    other["synthetic"]["seed"] = 999
    changed = generate_scada_dataset(Config(other))
    baseline = generate_scada_dataset(small_config)
    assert not changed["vibration"].equals(baseline["vibration"])


def test_required_columns_present(raw_frame):
    """Every column in the shared raw contract must exist."""
    missing = [column for column in REQUIRED_RAW_COLUMNS if column not in raw_frame.columns]
    assert missing == []


def test_turbines_are_distinct_and_complete(raw_frame, small_config):
    """The expected number of distinct turbine ids must be present."""
    ids = set(raw_frame[TURBINE_ID].dropna().unique())
    assert len(ids) == int(small_config.require("synthetic.n_turbines"))
    assert all(turbine.startswith("T") for turbine in ids)


def test_timestamps_are_chronological_per_turbine(clean_frame):
    """Within a turbine, timestamps must be strictly increasing."""
    for _, group in clean_frame.groupby(TURBINE_ID):
        times = pd.to_datetime(group[TIMESTAMP])
        assert times.is_monotonic_increasing
        assert times.is_unique


def test_failure_events_exist_and_are_rare(raw_frame):
    """Failures must occur, and must remain a rare event."""
    events = raw_frame[FAILURE_EVENT].sum()
    assert events > 0
    assert events / len(raw_frame) < 0.05


def test_operating_states_are_diverse(clean_frame):
    """The generator must exercise several distinct operating states."""
    states = set(clean_frame["operational_status"].unique())
    assert {str(OperationalStatus.NORMAL), str(OperationalStatus.IDLE)}.issubset(states)
    assert str(OperationalStatus.FAULT) in states or str(OperationalStatus.MAINTENANCE) in states


def test_data_quality_defects_are_injected(raw_frame, clean_frame):
    """Injection must add missing values that the clean variant does not have."""
    assert raw_frame["vibration"].isna().any()
    assert not clean_frame["vibration"].isna().any()
    assert raw_frame[TURBINE_ID].isna().any()


def test_power_increases_with_wind_then_saturates():
    """The reference power curve must ramp, then hold rated power, then cut out."""
    import numpy as np

    wind = np.array([0.0, 2.0, 5.0, 9.0, 12.5, 20.0, 26.0])
    power = power_curve(wind, rated_power=2000.0, cut_in=3.0, rated_wind=12.5, cut_out=25.0)
    assert power[0] == 0.0 and power[1] == 0.0  # below cut-in
    assert power[2] < power[3] < power[4]  # monotone ramp
    assert power[4] == power[5] == 2000.0  # rated plateau
    assert power[6] == 0.0  # above cut-out


def test_physical_relationships_hold(clean_frame):
    """Generated channels must respect the documented physical relationships."""
    running = clean_frame.loc[clean_frame["operational_status"] == str(OperationalStatus.NORMAL)]
    # Wind drives power.
    assert running["wind_speed"].corr(running["power_output"]) > 0.5
    # Rotor and generator speed are geared together.
    assert running["rotor_speed"].corr(running["generator_speed"]) > 0.95
    # Ambient temperature drives component temperatures.
    assert running["ambient_temperature"].corr(running["nacelle_temperature"]) > 0.5
    # Load drives gearbox temperature.
    assert running["power_output"].corr(running["gearbox_temperature"]) > 0.2


def test_turbines_have_different_baselines(small_config):
    """Per-turbine profiles must differ so absolute thresholds cannot work."""
    profiles = build_profiles(6, seed=42)
    means = {profile.mean_wind_speed for profile in profiles}
    assert len(means) == 6


def test_thermal_inertia_smooths_temperature(clean_frame):
    """Temperatures must change more smoothly than the load that drives them."""
    turbine = clean_frame.loc[clean_frame[TURBINE_ID] == "T01"].sort_values(TIMESTAMP)
    load_volatility = turbine["power_output"].diff().std() / max(
        turbine["power_output"].std(), 1e-9
    )
    temp_volatility = turbine["gearbox_temperature"].diff().std() / max(
        turbine["gearbox_temperature"].std(), 1e-9
    )
    assert temp_volatility < load_volatility


def test_degradation_precedes_failures(clean_frame):
    """Degradation severity must be elevated in the run-up to a failure."""
    failures = clean_frame.loc[clean_frame[FAILURE_EVENT] == 1]
    assert not failures.empty
    assert failures["degradation_level"].mean() > clean_frame["degradation_level"].mean()


def test_some_degradation_episodes_are_benign(clean_frame):
    """Not every degradation episode may end in a failure.

    This is what keeps the target learnable but non-trivial: a rising trend is
    necessary but not sufficient evidence of an imminent failure.
    """
    episodes = clean_frame.loc[clean_frame["episode_id"] >= 0]
    per_episode = episodes.groupby([TURBINE_ID, "episode_id"])[FAILURE_EVENT].max()
    assert per_episode.min() == 0, "expected at least one episode that heals without failing"
    assert per_episode.max() == 1, "expected at least one episode that ends in failure"
