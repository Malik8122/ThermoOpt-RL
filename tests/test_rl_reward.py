"""Step 5: reward function, its normalisation, and its link to simulator outputs."""

import hashlib

import pytest

from backend.config import load_config
from backend.rl.actions import CoolingCommand
from backend.rl.environment import DataCenterEnv
from backend.rl.reward import RewardInputs, calculate_reward, max_job_run_cost, thermal_penalty
from backend.simulation.run_baseline import run, write_csv
from tests.conftest import make_job

COMPONENTS = ("energy", "carbon", "thermal", "sla", "action", "completion")


def inputs(**kw) -> RewardInputs:
    base = dict(total_energy_kwh=0.3, carbon_g=120.0, timestep_minutes=1,
                rack_temperatures_c=(24.0, 24.0, 24.0, 24.0), jobs_failed=0)
    return RewardInputs(**{**base, **kw})


@pytest.fixture
def quiet_cfg():
    return load_config(overrides={"workloads": {"arrival": {"mean_rate_per_minute": 0.0}},
                                  "thermal": {"noise_std_c": 0.0}})


def test_breakdown_has_all_components_and_total_is_their_sum(cfg):
    b = calculate_reward(inputs(jobs_failed=1, started_wait_fractions=(0.5,), completed_priorities=(2, 3),
                                cooling_changed=True, migrations=1, invalid_action=False,
                                rack_temperatures_c=(28.0, 24.0, 31.0, 24.0)), cfg)
    assert tuple(c.name for c in b.components) == COMPONENTS
    assert b.total == pytest.approx(sum(c.contribution for c in b.components))
    # explicit formula
    w, r = cfg.reward.weights, cfg.reward
    e = 0.3 / (cfg.max_total_power_kw / 60)
    c = 120.0 / (cfg.max_total_power_kw / 60 * cfg.grid.max_carbon)
    th = (1 / 3 + 0 + (10 + 2 * 1) + 0) / 4
    s = (r.sla_violation_penalty * 1 + r.sla_delay_penalty * 0.5) / r.sla_normalisation
    m = (1 + 1) / 2
    expected = -(w.energy * e + w.carbon * c + w.thermal * th + w.sla * s + w.switching * m) + r.completion_bonus * 5
    assert b.total == pytest.approx(expected)


def test_reward_is_deterministic(cfg):
    x = inputs(jobs_failed=2, rack_temperatures_c=(30.5, 27.5, 25.0, 22.0))
    assert calculate_reward(x, cfg) == calculate_reward(x, cfg)


def test_energy_and_carbon_normalisation_range(cfg):
    p_max = cfg.max_total_power_kw
    worst = calculate_reward(inputs(total_energy_kwh=p_max / 60, carbon_g=p_max / 60 * cfg.grid.max_carbon), cfg)
    assert worst.energy.value == pytest.approx(1.0) and worst.carbon.value == pytest.approx(1.0)
    idle = calculate_reward(inputs(total_energy_kwh=0.0, carbon_g=0.0), cfg)
    assert idle.energy.contribution == 0.0 and idle.carbon.contribution == 0.0


def test_thermal_penalty_shape(cfg):
    z, t = cfg.thermal.zones, cfg.reward.thermal
    assert thermal_penalty(26.99, z, t) == 0.0
    assert thermal_penalty(27.0, z, t) == 0.0
    assert thermal_penalty(28.5, z, t) == pytest.approx(0.5)
    assert thermal_penalty(29.99, z, t) == pytest.approx(2.99 / 3)
    assert thermal_penalty(29.99, z, t) < 1.0 < 10.0 == thermal_penalty(30.0, z, t)
    assert thermal_penalty(32.0, z, t) == pytest.approx(14.0)


def test_critical_rack_outweighs_any_energy_saving(cfg):
    """Zero energy with one critical rack must still score worse than maximum energy/carbon with all racks normal."""
    p_max = cfg.max_total_power_kw
    critical_but_free = calculate_reward(inputs(total_energy_kwh=0.0, carbon_g=0.0,
                                                rack_temperatures_c=(30.0, 20.0, 20.0, 20.0)), cfg)
    safe_but_expensive = calculate_reward(inputs(total_energy_kwh=p_max / 60, carbon_g=p_max / 60 * cfg.grid.max_carbon,
                                                 rack_temperatures_c=(26.9, 26.9, 26.9, 26.9)), cfg)
    assert critical_but_free.total < safe_but_expensive.total
    assert critical_but_free.thermal.contribution == pytest.approx(-0.75)


def test_deadline_violation_costs_more_than_maximal_delay(cfg):
    violation = calculate_reward(inputs(jobs_failed=1), cfg)
    worst_delay = calculate_reward(inputs(started_wait_fractions=(1.0,)), cfg)   # used all its slack, still on time
    on_time = calculate_reward(inputs(started_wait_fractions=(0.0,)), cfg)
    assert violation.sla.contribution < worst_delay.sla.contribution < on_time.sla.contribution == 0.0
    r = cfg.reward
    assert violation.sla.contribution / worst_delay.sla.contribution == pytest.approx(
        r.sla_violation_penalty / r.sla_delay_penalty)


def test_missing_a_deadline_never_pays_off(cfg):
    """A violation must cost more than the energy+carbon penalty of running the largest possible job."""
    worst_run_cost = max_job_run_cost(cfg)
    assert worst_run_cost == pytest.approx(4.2548, abs=1e-3)   # AI_TRAINING: 50 CPU, 100 GPU, 120 min
    violation = -calculate_reward(inputs(jobs_failed=1), cfg).sla.contribution
    assert violation > worst_run_cost


def test_invalid_action_penalty_is_in_action_component(cfg):
    b = calculate_reward(inputs(invalid_action=True), cfg)
    assert b.action.contribution == pytest.approx(-cfg.environment.invalid_action_penalty)


# -------------------------------------------- reward wired to the simulator
def test_energy_and_carbon_rewards_use_actual_simulator_outputs():
    env = DataCenterEnv(episode_minutes=30)
    env.reset(seed=42)
    cfg = env.config
    prev = env.dc.accountant.totals.total_kwh
    for _ in range(30):
        _, reward, _, _, info = env.step(env.codec.encode(env.codec.delay_index, CoolingCommand.MAINTAIN))
        step_energy = env.dc.accountant.totals.total_kwh - prev
        prev = env.dc.accountant.totals.total_kwh
        assert info["energy_kwh"] == pytest.approx(step_energy)
        assert info["energy_kwh"] == pytest.approx(info["it_energy_kwh"] + info["cooling_energy_kwh"])
        assert info["carbon_kg"] * 1000 == pytest.approx(info["energy_kwh"] * info["carbon_intensity"])
        assert info["reward_energy"] == pytest.approx(-0.30 * info["energy_kwh"] / (cfg.max_total_power_kw / 60))
        assert info["reward_carbon"] == pytest.approx(
            -0.20 * info["carbon_kg"] * 1000 / (cfg.max_total_power_kw / 60 * cfg.grid.max_carbon))
        parts = sum(info[f"reward_{c}"] for c in COMPONENTS)
        assert reward == pytest.approx(parts) == pytest.approx(info["reward_total"])


def test_delayed_job_is_penalised_less_than_expired_job(quiet_cfg):
    env = DataCenterEnv(config=quiet_cfg, episode_minutes=60)
    env.reset(seed=42)
    delay = env.codec.encode(env.codec.delay_index, CoolingCommand.MAINTAIN)
    flexible = make_job("FLEX", cpu=10, duration=5, slack=100)
    env.dc.submit_workload(flexible)
    for _ in range(20):
        env.step(delay)                                   # waits 20 of its 100 slack minutes
    _, _, _, _, info = env.step(env.codec.encode(3, CoolingCommand.MAINTAIN))
    assert info["action_valid"] and info["jobs_failed"] == 0
    r = env.config.reward
    assert info["reward_sla"] == pytest.approx(-0.15 * r.sla_delay_penalty * (20 / 100) / r.sla_normalisation)

    strict = make_job("STRICT", cpu=10, duration=5, slack=2, arrival=env.dc.minute)
    env.dc.submit_workload(strict)
    infos = [env.step(delay)[4] for _ in range(4)]
    failed_step = next(i for i in infos if i["jobs_failed"])
    assert failed_step["failed_job_ids"] == ["STRICT"]
    assert failed_step["reward_sla"] == pytest.approx(-0.15 * r.sla_violation_penalty / r.sla_normalisation)
    assert failed_step["reward_sla"] < info["reward_sla"]


def test_completion_reward_counts_only_agent_scheduled_jobs(quiet_cfg):
    env = DataCenterEnv(config=quiet_cfg, episode_minutes=200)
    env.reset(seed=42)
    job = make_job("DONE", cpu=10, duration=3)
    job.priority = 3
    env.dc.submit_workload(job)
    infos = [env.step(env.codec.encode(3, CoolingCommand.MAINTAIN))[4]]
    infos += [env.step(env.codec.encode(4, CoolingCommand.MAINTAIN))[4] for _ in range(150)]
    done = [i for i in infos if i["reward_completion"] > 0]
    assert len(done) == 1 and done[0]["reward_completion"] == pytest.approx(0.05 * 3)
    assert sum(i["jobs_completed"] for i in infos) > 1           # background jobs completed too, unrewarded


def test_baseline_simulation_unchanged_by_step5(cfg, tmp_path):
    """The Steps 2-4 seed-42 baseline must stay byte-identical."""
    digest = write_csv(run(cfg, 60, seed=42), tmp_path / "baseline.csv")
    assert digest == "e7fe90bec76dafbc1411ed821f86a59a2d8f54bdf13d4d445abd5639a89cafab"
    assert hashlib.sha256((tmp_path / "baseline.csv").read_bytes()).hexdigest() == digest
