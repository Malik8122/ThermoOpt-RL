"""Step 4A/4B: IT power model and cooling model."""

import pytest

from backend.config import load_config
from backend.simulation.cooling import CoolingSystem
from backend.simulation.enums import SimulationInvariantError
from backend.simulation.power import PowerModel


@pytest.fixture
def power(cfg) -> PowerModel:
    return PowerModel.from_config(cfg.power)


@pytest.fixture
def cooling(cfg) -> CoolingSystem:
    return CoolingSystem.from_config(cfg.cooling)


# ------------------------------------------------------------------ power
def test_device_power_at_idle_and_full_load(power):
    assert power.server_power_w(0.0) == 150.0
    assert power.server_power_w(1.0) == 500.0
    assert power.gpu_power_w(0.0) == 100.0
    assert power.gpu_power_w(1.0) == 700.0


def test_linear_power_equation(power):
    # P = P_idle + (P_max - P_idle) * u
    assert power.server_power_w(0.72) == pytest.approx(150 + 350 * 0.72)
    assert power.gpu_power_w(0.55) == pytest.approx(100 + 600 * 0.55)


def test_higher_utilisation_increases_power(power):
    grid = [i / 20 for i in range(21)]
    server = [power.server_power_w(u) for u in grid]
    gpu = [power.gpu_power_w(u) for u in grid]
    assert all(b > a for a, b in zip(server, server[1:]))
    assert all(b > a for a, b in zip(gpu, gpu[1:]))


def test_power_exponent_is_configurable(cfg):
    convex = PowerModel.from_config(load_config(overrides={"power": {"utilisation_exponent": 2.0}}).power)
    assert convex.server_power_w(0.5) == pytest.approx(150 + 350 * 0.25)


@pytest.mark.parametrize("u", [-0.01, 1.01])
def test_power_rejects_impossible_utilisation(power, u):
    with pytest.raises(SimulationInvariantError):
        power.server_power_w(u)


# ---------------------------------------------------------------- cooling
@pytest.mark.parametrize("level, expected_kw", [(0.0, 0.0), (0.6, 9.6), (0.65, 10.4), (1.0, 16.0)])
def test_cooling_power_equation(cooling, level, expected_kw):
    # P_cooling = level * 16 kW
    assert cooling.power_kw(level) == pytest.approx(expected_kw)


def test_cooling_response_is_not_instant(cooling):
    """65 % -> 100 % ramps linearly over the 3-minute response time."""
    assert cooling.current_level == pytest.approx(0.65)
    cooling.set_target(1.0)
    assert cooling.current_level == pytest.approx(0.65)            # nothing happens until time passes
    assert cooling.advance(1) == pytest.approx(0.65 + 0.35 / 3)
    assert cooling.in_transition and cooling.progress == pytest.approx(1 / 3)
    assert cooling.advance(1) == pytest.approx(0.65 + 0.35 * 2 / 3)
    assert cooling.advance(1) == pytest.approx(1.0)
    assert not cooling.in_transition
    assert cooling.advance(5) == pytest.approx(1.0)                # stays at target


def test_cooling_retarget_mid_transition_starts_from_delivered_level(cooling):
    cooling.set_target(1.0)
    cooling.advance(1)                         # delivered ~0.7667
    delivered = cooling.current_level
    cooling.set_target(0.4)
    assert cooling.advance(1) == pytest.approx(delivered + (0.4 - delivered) / 3)


def test_cooling_power_follows_delivered_not_requested_level(cooling):
    cooling.set_target(1.0)
    cooling.advance(1)
    assert cooling.power_kw() == pytest.approx(16.0 * (0.65 + 0.35 / 3))


def test_increase_and_decrease_step_through_discrete_levels(cooling):
    assert cooling.increase() == pytest.approx(0.8)      # 65 % is off-grid: next level up is 80 %
    assert cooling.increase() == pytest.approx(1.0)
    assert cooling.increase() == pytest.approx(1.0)      # saturates at the top
    fresh = CoolingSystem.from_config(load_config().cooling)
    assert fresh.decrease() == pytest.approx(0.6)
    for _ in range(5):
        fresh.decrease()
    assert fresh.target_level == pytest.approx(0.0)


def test_zero_response_time_is_instant(cfg):
    instant = CoolingSystem.from_config(load_config(overrides={"cooling": {"response_time_minutes": 0}}).cooling)
    instant.set_target(0.2)
    assert instant.current_level == pytest.approx(0.2)


@pytest.mark.parametrize("level", [-0.1, 1.1])
def test_cooling_rejects_out_of_range_target(cooling, level):
    with pytest.raises(SimulationInvariantError):
        cooling.set_target(level)
