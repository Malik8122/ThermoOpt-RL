"""Step 4C: thermal model, ambient model, temperature zones, calibration."""

import numpy as np
import pytest

from backend.config import load_config
from backend.simulation.enums import TemperatureStatus
from backend.simulation.thermal import AmbientModel, ThermalModel, get_temperature_status

ETA = np.array([1.0, 1.0, 0.75, 1.0])


@pytest.fixture
def model(noiseless_cfg) -> ThermalModel:
    return ThermalModel(noiseless_cfg.thermal, timestep_minutes=1)


def test_single_step_matches_equation(model):
    # T' = T + 0.035*100*L - 0.045*100*C_eff + 0.25*(24 - T)
    T, L, C = 26.0, 0.8, 0.6
    expected = T + 3.5 * L - 4.5 * C + 0.25 * (24.0 - T)
    assert model.step(np.array([T]), np.array([L]), np.array([C]), 24.0)[0] == pytest.approx(expected)


def test_coefficient_units(model, noiseless_cfg):
    """alpha/beta are per percentage point: a fully loaded, uncooled rack settles 14 °C above ambient.

    Regression guard: applying 0.035 to a 0-1 fraction would give only 0.14 °C,
    which makes WARNING/CRITICAL unreachable. That is why the model scales by 100.
    """
    th = noiseless_cfg.thermal
    rise = model.equilibrium(np.array([1.0]), np.array([0.0]), 24.0)[0] - 24.0
    assert rise == pytest.approx(th.alpha * 100 / th.ambient_coupling) == pytest.approx(14.0)
    assert th.alpha / th.ambient_coupling == pytest.approx(0.14)   # the rejected interpretation


def test_higher_cooling_lowers_temperature(model):
    T, L = np.full(5, 27.0), np.full(5, 0.7)
    C = np.array([0.0, 0.2, 0.4, 0.6, 1.0])
    nxt = model.step(T, L, C, 24.0)
    assert np.all(np.diff(nxt) < 0)


def test_higher_utilisation_raises_temperature(model):
    T, C = np.full(5, 25.0), np.full(5, 0.6)
    L = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    nxt = model.step(T, L, C, 24.0)
    assert np.all(np.diff(nxt) > 0)


def test_ambient_term_pulls_towards_ambient(model):
    """No load, no cooling: a hot rack cools, a cold rack warms, a rack at ambient stays."""
    nxt = model.step(np.array([30.0, 20.0, 24.0]), np.zeros(3), np.zeros(3), 24.0)
    assert nxt[0] < 30.0 and nxt[1] > 20.0 and nxt[2] == pytest.approx(24.0)


def test_rack3_cools_less_effectively(model):
    """Same load and cooling setting: Rack 3 settles 4.5*C*(1-0.75)/0.25 °C hotter."""
    T = np.full(4, 25.0)
    load, level = np.full(4, 0.6), 0.8
    effective = ETA * level
    assert effective.tolist() == pytest.approx([0.8, 0.8, 0.6, 0.8])
    for _ in range(200):
        T = model.step(T, load, effective, 24.0)
    assert T[2] - T[0] == pytest.approx(4.5 * level * 0.25 / 0.25, abs=1e-6)


def test_long_run_converges_to_equilibrium(model):
    load, eff = np.array([0.856]), np.array([0.65 * 0.75])
    T = np.array([28.1])
    for _ in range(200):
        T = model.step(T, load, eff, 24.0)
    assert T[0] == pytest.approx(model.equilibrium(load, eff, 24.0)[0], abs=1e-6)


def test_temperatures_stay_within_bounds():
    cfg = load_config(overrides={"thermal": {"noise_std_c": 5.0}})   # absurd noise to stress the clip
    model = ThermalModel(cfg.thermal, 1)
    rng = np.random.default_rng(0)
    T = np.full(4, 25.0)
    for i in range(2000):
        load = np.full(4, 1.0 if i % 400 < 200 else 0.0)
        cool = np.full(4, 0.0 if i % 400 < 200 else 1.0)
        T = model.step(T, load, cool, 24.0, rng)
        assert np.all(T >= 18.0) and np.all(T <= 40.0)


@pytest.mark.parametrize("temp, status", [
    (18.0, TemperatureStatus.NORMAL), (26.99, TemperatureStatus.NORMAL),
    (27.0, TemperatureStatus.WARNING), (29.99, TemperatureStatus.WARNING),
    (30.0, TemperatureStatus.CRITICAL), (40.0, TemperatureStatus.CRITICAL),
])
def test_temperature_zone_boundaries(cfg, temp, status):
    assert get_temperature_status(temp, cfg.thermal.zones) is status


def test_ambient_profile(cfg):
    amb = AmbientModel.from_config(cfg.thermal)
    assert amb.temperature_at(15.0) == pytest.approx(26.0)     # peak
    assert amb.temperature_at(3.0) == pytest.approx(22.0)      # trough, 12 h later
    assert amb.temperature_at(9.0) == pytest.approx(24.0)      # mean crossing
    flat = AmbientModel(24.0, 0.0, 15.0)
    assert {flat.temperature_at(h) for h in range(24)} == {24.0}


def test_calibration_scenarios(cfg):
    """The design-document calibration table: Rack 3 is the structural hotspot."""
    model = ThermalModel(cfg.thermal, 1)
    amb = AmbientModel.from_config(cfg.thermal)
    load = model.thermal_load(np.array([0.72, 0.51, 0.88, 0.43]), np.array([0.55, 0.40, 0.82, 0.30]))
    morning = model.equilibrium(load, ETA * 0.65, amb.temperature_at(8.0))
    afternoon = model.equilibrium(load, ETA * 0.65, amb.temperature_at(15.0))
    afternoon_low = model.equilibrium(load, ETA * 0.40, amb.temperature_at(15.0))
    assert np.argmax(morning) == 2 and morning.max() < 27.0                       # NORMAL in the morning
    assert 27.0 <= afternoon[2] < 30.0 and np.all(np.delete(afternoon, 2) < 27.0)  # only Rack 3 in WARNING
    assert afternoon_low[2] >= 30.0                                                # CRITICAL if cooling is cut
