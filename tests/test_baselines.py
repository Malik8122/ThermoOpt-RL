"""Step 6: baseline controllers and the episode runner."""

import numpy as np
import pytest

from backend.config import load_config
from backend.evaluation.runner import METRICS, run_episode, summarise
from backend.rl.actions import CoolingCommand
from backend.rl.baselines import (
    BASELINES,
    ObservationReader,
    RandomController,
    RoundRobinController,
    RuleBasedController,
    make_baseline,
)
from backend.rl.environment import DataCenterEnv
from tests.conftest import make_job


@pytest.fixture
def quiet_cfg():
    """No random arrivals, no thermal noise: only the jobs a test submits."""
    return load_config(overrides={"workloads": {"arrival": {"mean_rate_per_minute": 0.0}},
                                  "thermal": {"noise_std_c": 0.0}})


def quiet_env(cfg, **kwargs) -> DataCenterEnv:
    env = DataCenterEnv(config=cfg, episode_minutes=kwargs.pop("episode_minutes", 60), **kwargs)
    env.reset(seed=42)
    return env


def empty_racks(env: DataCenterEnv) -> None:
    """Remove the background jobs so every rack has full capacity."""
    for rack in env.dc.racks:
        for job_id in list(rack.active_workload_ids):
            rack.remove_workload(job_id)
            del env.dc.workloads[job_id]


def observe(env: DataCenterEnv) -> np.ndarray:
    return env.observation_builder.build(env.dc)


def decode(env: DataCenterEnv, action: int):
    return env.codec.decode(action)


# ------------------------------------------------------------ observation reader
def test_reader_recovers_physical_values(quiet_cfg):
    env = quiet_env(quiet_cfg)
    env.dc.submit_workload(make_job("J", cpu=10, slack=100, duration=5, arrival=env.dc.minute))
    s = ObservationReader(quiet_cfg).read(observe(env))
    np.testing.assert_allclose(s.rack_temperature_c, env.dc.temperatures(), atol=1e-4)
    assert s.cooling_setpoint == pytest.approx(env.dc.cooling.target_level, abs=1e-6)
    assert s.job_waiting and s.head_flexible
    assert s.head_slack_minutes == pytest.approx(env.dc.queue[0].slack_at(env.dc.minute), abs=1e-3)
    assert s.queue_length == pytest.approx(1.0)
    assert s.carbon_g_per_kwh == pytest.approx(env.dc.grid.carbon_at(env.dc.hour_of_day), rel=1e-5)
    assert list(s.head_fits) == [r.can_accept_workload(env.dc.queue[0]).success for r in env.dc.racks]


# ----------------------------------------------------------------- registry
def test_registry_and_factory(cfg):
    assert set(BASELINES) == {"random", "round_robin", "rule_based"}
    assert isinstance(make_baseline("rule_based", cfg), RuleBasedController)
    with pytest.raises(ValueError):
        make_baseline("oracle", cfg)


# ------------------------------------------------------------------- random
def test_random_respects_mask_and_is_reproducible(cfg):
    env = DataCenterEnv(episode_minutes=30)

    def actions(seed):
        c = RandomController(cfg)
        obs, _ = env.reset(seed=seed)
        c.reset(seed)
        out = []
        for _ in range(30):
            mask = env.action_masks()
            a = c.act(obs, mask)
            assert mask[a]
            out.append(a)
            obs, *_ = env.step(a)
        return out

    assert actions(42) == actions(42)
    assert actions(42) != actions(43)


def test_random_without_mask_covers_whole_space(cfg):
    c = RandomController(cfg)
    c.reset(1)
    seen = {c.act(np.zeros(30, dtype=np.float32)) for _ in range(500)}
    assert seen == set(range(18))


# -------------------------------------------------------------- round-robin
def test_round_robin_cycles_racks_and_skips_full_ones(quiet_cfg):
    env = quiet_env(quiet_cfg)
    empty_racks(env)
    c = RoundRobinController(quiet_cfg)
    c.reset()
    placed = []
    for i in range(5):
        env.dc.submit_workload(make_job(f"J{i}", cpu=10, duration=30, arrival=env.dc.minute))
        a = decode(env, c.act(observe(env)))
        placed.append(a.rack_index)
        env.step(env.codec.encode(a.placement, CoolingCommand.MAINTAIN))
    assert placed == [0, 1, 2, 3, 0]

    # Rack 2 (index 1) is full: the next job skips it.
    env.dc.racks[1].assign_workload(make_job("FILL", cpu=env.dc.racks[1].get_available_cpu(), duration=60,
                                             arrival=env.dc.minute), env.dc.minute)
    env.dc.submit_workload(make_job("J5", cpu=10, duration=30, arrival=env.dc.minute))
    assert decode(env, c.act(observe(env))).rack_index == 2


def test_round_robin_delays_when_nothing_fits_and_noops_on_empty_queue(quiet_cfg):
    env = quiet_env(quiet_cfg)
    c = RoundRobinController(quiet_cfg)
    assert decode(env, c.act(observe(env))).placement == env.codec.delay_index   # empty queue
    env.dc.submit_workload(make_job("BIG", cpu=101, duration=5, arrival=env.dc.minute))
    assert decode(env, c.act(observe(env))).placement == env.codec.delay_index


def test_round_robin_holds_cooling_at_configured_level(quiet_cfg):
    env = quiet_env(quiet_cfg)
    c = RoundRobinController(quiet_cfg)
    c.reset()
    obs = observe(env)
    for _ in range(10):
        obs, *_ = env.step(c.act(obs))
    assert env.dc.cooling.target_level == pytest.approx(0.60)
    assert decode(env, c.act(obs)).cooling is CoolingCommand.MAINTAIN


# --------------------------------------------------------------- rule-based
def rule_env(quiet_cfg, start_hour: float):
    env = DataCenterEnv(config=quiet_cfg, episode_minutes=60, start_hour=start_hour)
    env.reset(seed=42)
    empty_racks(env)
    return env


def set_temperatures(env, temps):
    for rack, t in zip(env.dc.racks, temps):
        rack.temperature_c = t


def test_rule_based_places_on_least_utilised_rack_below_threshold(quiet_cfg):
    env = rule_env(quiet_cfg, start_hour=16.0)          # low carbon (250 g/kWh)
    set_temperatures(env, [25.0, 25.0, 25.0, 25.0])
    for i, cpu in enumerate([50, 20, 10, 30]):
        env.dc.racks[i].assign_workload(make_job(f"B{i}", cpu=cpu, duration=60), env.dc.minute)
    set_temperatures(env, [25.0, 25.0, 28.0, 25.0])     # rack 3 is least utilised but hot
    env.dc.submit_workload(make_job("J", cpu=10, slack=0, duration=5, arrival=env.dc.minute))
    a = decode(env, RuleBasedController(quiet_cfg).act(observe(env)))
    assert a.rack_index == 1


def test_rule_based_delays_flexible_job_in_high_carbon_only_when_it_can_wait(quiet_cfg):
    env = rule_env(quiet_cfg, start_hour=20.0)          # 520 g/kWh
    set_temperatures(env, [24.5] * 4)
    c = RuleBasedController(quiet_cfg)
    env.dc.submit_workload(make_job("FLEX", cpu=10, slack=120, duration=5, arrival=env.dc.minute))
    assert decode(env, c.act(observe(env))).placement == env.codec.delay_index

    env.dc.queue.clear()
    env.dc.submit_workload(make_job("STRICT", cpu=10, slack=5, duration=5, arrival=env.dc.minute))
    assert decode(env, c.act(observe(env))).rack_index is not None

    env.dc.queue.clear()
    env.dc.submit_workload(make_job("LATE", cpu=10, slack=20, duration=5, arrival=env.dc.minute))  # < 30 min slack
    assert decode(env, c.act(observe(env))).rack_index is not None


def test_rule_based_never_delays_when_other_jobs_are_waiting(quiet_cfg):
    env = rule_env(quiet_cfg, start_hour=20.0)
    set_temperatures(env, [24.5] * 4)
    env.dc.submit_workload(make_job("FLEX", cpu=10, slack=120, duration=5, arrival=env.dc.minute))
    env.dc.submit_workload(make_job("STRICT", cpu=10, slack=2, duration=5, arrival=env.dc.minute))
    assert decode(env, RuleBasedController(quiet_cfg).act(observe(env))).rack_index is not None


def test_rule_based_does_not_delay_in_low_carbon(quiet_cfg):
    env = rule_env(quiet_cfg, start_hour=16.0)
    set_temperatures(env, [24.5] * 4)
    env.dc.submit_workload(make_job("FLEX", cpu=10, slack=120, duration=5, arrival=env.dc.minute))
    assert decode(env, RuleBasedController(quiet_cfg).act(observe(env))).rack_index is not None


def test_rule_based_uses_coolest_hot_rack_for_urgent_job(quiet_cfg):
    env = rule_env(quiet_cfg, start_hour=16.0)
    set_temperatures(env, [29.0, 28.0, 29.5, 27.5])
    c = RuleBasedController(quiet_cfg)
    env.dc.submit_workload(make_job("STRICT", cpu=10, slack=0, duration=5, arrival=env.dc.minute))
    assert decode(env, c.act(observe(env))).rack_index == 3
    env.dc.queue.clear()
    env.dc.submit_workload(make_job("FLEX", cpu=10, slack=120, duration=5, arrival=env.dc.minute))
    assert decode(env, c.act(observe(env))).placement == env.codec.delay_index   # waits for a cool rack


@pytest.mark.parametrize("temps, expected", [
    ([24.0, 25.0, 27.5, 25.0], CoolingCommand.INCREASE),
    ([23.0, 23.5, 23.9, 22.0], CoolingCommand.DECREASE),
    ([23.0, 25.0, 26.9, 24.0], CoolingCommand.MAINTAIN),
])
def test_rule_based_cooling_thresholds(quiet_cfg, temps, expected):
    env = rule_env(quiet_cfg, start_hour=16.0)
    set_temperatures(env, temps)
    assert decode(env, RuleBasedController(quiet_cfg).act(observe(env))).cooling is expected


def test_rule_based_cooling_stays_within_levels(quiet_cfg):
    env = rule_env(quiet_cfg, start_hour=16.0)
    c = RuleBasedController(quiet_cfg)
    env.dc.set_cooling_target(1.0)
    set_temperatures(env, [29.0] * 4)
    assert decode(env, c.act(observe(env))).cooling is CoolingCommand.MAINTAIN
    env.dc.set_cooling_target(0.0)
    set_temperatures(env, [20.0] * 4)
    assert decode(env, c.act(observe(env))).cooling is CoolingCommand.MAINTAIN


# ------------------------------------------------------------------ runner
@pytest.mark.parametrize("name", sorted(BASELINES))
def test_full_day_episode_runs_with_no_invalid_actions(cfg, name):
    env = DataCenterEnv(episode_minutes=1440)
    result = run_episode(env, make_baseline(name, cfg), seed=42)
    assert set(METRICS) <= set(result)
    assert result["invalid_action_rate"] == 0.0
    assert result["energy_kwh"] == pytest.approx(result["it_energy_kwh"] + result["cooling_energy_kwh"])
    assert result["completed_jobs"] > 0


def test_runner_is_deterministic(cfg):
    env = DataCenterEnv(episode_minutes=120)
    a = run_episode(env, make_baseline("random", cfg), seed=7)
    b = run_episode(env, make_baseline("random", cfg), seed=7)
    assert a == b


def test_summarise_mean_and_std():
    s = summarise([{"x": 1.0}, {"x": 3.0}], metrics=("x",))
    assert s["x"]["mean"] == 2.0 and s["x"]["std"] == pytest.approx(np.sqrt(2.0))
