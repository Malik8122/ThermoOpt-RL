"""Step 5: Gymnasium API, action validation, observations, seeding, migration."""

import copy

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from backend.config import load_config
from backend.rl import ENV_ID
from backend.rl.actions import CoolingCommand, TabularPlacement, apply_action, encode_tabular_action, validate_action
from backend.rl.environment import DataCenterEnv
from backend.simulation.enums import WorkloadStatus
from tests.conftest import make_job

MAINTAIN = int(CoolingCommand.MAINTAIN)


@pytest.fixture
def quiet_cfg():
    """No random arrivals, no thermal noise: only the jobs a test submits."""
    return load_config(overrides={"workloads": {"arrival": {"mean_rate_per_minute": 0.0}},
                                  "thermal": {"noise_std_c": 0.0}})


def quiet_env(quiet_cfg, **kwargs) -> DataCenterEnv:
    env = DataCenterEnv(config=quiet_cfg, episode_minutes=kwargs.pop("episode_minutes", 60), **kwargs)
    env.reset(seed=42)
    return env


def act(env: DataCenterEnv, placement: int, cooling: CoolingCommand = CoolingCommand.MAINTAIN) -> int:
    return env.codec.encode(placement, cooling)


def rack_usage(env):
    return [(r.get_available_cpu(), r.get_available_gpu(), tuple(sorted(r.active_workload_ids))) for r in env.dc.racks]


# ------------------------------------------------------------------ Gym API
def test_registered_and_passes_gymnasium_checker():
    env = gym.make(ENV_ID, episode_minutes=30)
    assert isinstance(env.unwrapped, DataCenterEnv)
    check_env(env.unwrapped, skip_render_check=True)
    tab = gym.make(ENV_ID, representation="tabular", episode_minutes=30)
    assert tab.action_space.n == 9 and tab.observation_space.n == 144
    check_env(tab.unwrapped, skip_render_check=True)


def test_reset_returns_valid_observation_and_info():
    env = DataCenterEnv(episode_minutes=60)
    obs, info = env.reset(seed=42)
    assert obs.shape == (30,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert info["timestamp"] == "08:00" and info["episode_seed"] == 42
    assert info["action_mask"].shape == (18,)


def test_step_returns_gymnasium_tuple_with_required_info():
    env = DataCenterEnv(episode_minutes=60)
    env.reset(seed=42)
    result = env.step(act(env, env.codec.delay_index))
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float) and terminated is False and truncated is False
    required = {"timestamp", "step", "action", "action_valid", "invalid_action_reason",
                "reward_total", "reward_energy", "reward_carbon", "reward_thermal", "reward_sla",
                "reward_action", "reward_completion", "energy_kwh", "it_energy_kwh", "cooling_energy_kwh",
                "carbon_kg", "electricity_cost", "carbon_intensity", "electricity_price",
                "avg_temperature", "peak_temperature", "thermal_violations", "critical_violations",
                "jobs_arrived", "jobs_completed", "jobs_failed", "active_jobs"}
    assert required <= set(info)


@pytest.mark.parametrize("minutes", [60, 120])
def test_episode_truncates_at_configured_length(minutes):
    env = DataCenterEnv(episode_minutes=minutes)
    env.reset(seed=1)
    steps, truncated = 0, False
    while not truncated:
        _, _, terminated, truncated, info = env.step(act(env, env.codec.delay_index))
        assert not terminated
        steps += 1
    assert steps == minutes and info["timestamp"] == env.dc.clock
    with pytest.raises(RuntimeError):
        env.step(0)


def test_default_episode_is_one_day_and_start_hour_is_configurable():
    env = DataCenterEnv()
    assert env.max_steps == 1440
    _, info = env.reset(seed=1, options={"start_hour": 13.5})
    assert info["timestamp"] == "13:30"


def test_step_before_reset_and_out_of_space_actions_raise():
    env = DataCenterEnv(episode_minutes=10)
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(18)


# ------------------------------------------------------------ observations
def test_exactly_30_named_base_features():
    env = DataCenterEnv(episode_minutes=10)
    b = env.observation_builder
    assert b.size == b.base_size == 30 and len(set(b.feature_names)) == 30
    assert not b.include_predictions


def test_observation_matches_simulator_and_stays_in_bounds():
    env = DataCenterEnv(episode_minutes=240)
    obs, _ = env.reset(seed=3)
    rng = np.random.default_rng(3)
    truncated = False
    while not truncated:
        feats = env.observation_builder.as_dict(obs)
        assert env.observation_space.contains(obs)
        assert feats["rack1_temperature"] == pytest.approx((env.dc.racks[0].temperature_c - 18) / 22, abs=1e-6)
        assert feats["cooling_delivered"] == pytest.approx(env.dc.cooling.current_level, abs=1e-6)
        assert feats["job_waiting"] == float(bool(env.dc.queue))
        if not env.dc.queue:
            assert feats["head_cpu"] == feats["head_urgency"] == feats["head_fits_rack1"] == 0.0
        obs, _, _, truncated, _ = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))


def test_predictive_extension_adds_12_forecast_features():
    persistence = lambda dc: np.repeat(dc.temperatures()[:, None], 3, axis=1)  # noqa: E731  (stand-in, not a model)
    env = DataCenterEnv(episode_minutes=10, prediction_provider=persistence)
    obs, _ = env.reset(seed=42)
    assert obs.shape == (42,) and env.observation_space.shape == (42,)
    assert env.observation_builder.base_size == 30 and env.observation_builder.prediction_size == 12
    assert env.observation_space.contains(obs)
    assert np.allclose(obs[30:33], obs[0], atol=1e-6)          # rack 1 forecasts == its current temperature
    bad = DataCenterEnv(episode_minutes=10, prediction_provider=lambda dc: np.zeros((4, 2)))
    with pytest.raises(ValueError):
        bad.reset(seed=42)


# ------------------------------------------------------- action validation
def test_valid_placement_is_applied(quiet_cfg):
    env = quiet_env(quiet_cfg)
    job = make_job("J", cpu=20, gpu=10, duration=5)
    env.dc.submit_workload(job)
    free_before = env.dc.racks[3].get_available_cpu()
    _, _, _, _, info = env.step(act(env, 3))
    assert info["action_valid"] and info["invalid_action_reason"] is None and info["placement_effect"] == "place"
    assert job.status is WorkloadStatus.RUNNING and job.assigned_rack == 3
    assert env.dc.racks[3].get_available_cpu() == free_before - 20


@pytest.mark.parametrize("cpu, gpu, reason", [(30, 0, "INSUFFICIENT_CPU"), (5, 30, "INSUFFICIENT_GPU")])
def test_invalid_placement_is_detected_and_changes_nothing(quiet_cfg, cpu, gpu, reason):
    """Rack 3 has 12 CPU and 18 GPU units free. The joint action is atomic: the valid cooling part is not applied."""
    env = quiet_env(quiet_cfg)
    job = make_job("J", cpu=cpu, gpu=gpu)
    env.dc.submit_workload(job)
    check = validate_action(env.dc, env.decode_action(act(env, 2, CoolingCommand.INCREASE)))
    usage, target = rack_usage(env), env.dc.cooling.target_level
    apply_action(env.dc, check)                                   # direct call: must be a no-op
    assert rack_usage(env) == usage and env.dc.cooling.target_level == target
    _, reward, _, _, info = env.step(act(env, 2, CoolingCommand.INCREASE))
    assert info["action_valid"] is False and info["invalid_action_reason"] == reason
    assert info["primitive_action"] == act(env, 2, CoolingCommand.INCREASE)   # not converted to another action
    assert job.status is WorkloadStatus.QUEUED and job.assigned_rack is None and env.dc.queue == [job]
    assert env.dc.cooling.target_level == target                  # cooling part was not applied either
    assert info["reward_action"] == pytest.approx(-0.10)
    env.dc.check_invariants()


def test_cooling_limits_are_invalid_not_saturated(quiet_cfg):
    env = quiet_env(quiet_cfg)
    env.dc.set_cooling_target(1.0)
    _, _, _, _, info = env.step(act(env, env.codec.delay_index, CoolingCommand.INCREASE))
    assert not info["action_valid"] and info["invalid_action_reason"] == "COOLING_AT_MAXIMUM"
    env.dc.set_cooling_target(0.0)
    _, _, _, _, info = env.step(act(env, env.codec.delay_index, CoolingCommand.DECREASE))
    assert not info["action_valid"] and info["invalid_action_reason"] == "COOLING_AT_MINIMUM"


def test_empty_queue_placement_is_a_valid_no_op(quiet_cfg):
    env = quiet_env(quiet_cfg)
    usage = rack_usage(env)
    check = validate_action(env.dc, env.decode_action(act(env, 0)))
    assert check.valid and check.placement_effect == "no_job"
    apply_action(env.dc, check)
    assert rack_usage(env) == usage


def test_action_mask_agrees_with_step_validity(quiet_cfg):
    env = quiet_env(quiet_cfg)
    env.dc.submit_workload(make_job("J", cpu=30, gpu=50))
    mask = env.action_masks()
    for a in range(18):
        clone = copy.deepcopy(env)
        _, _, _, _, info = clone.step(a)
        assert info["action_valid"] == mask[a], a
    assert not mask.all() and mask.any()


def test_tabular_macro_actions_resolve_to_documented_racks(quiet_cfg):
    env = quiet_env(quiet_cfg, representation="tabular")
    env.dc.racks[3].update_temperature(26.0)   # Rack 4 is now warmer than Rack 2 (23.8 °C) but still least utilised
    env.dc.submit_workload(make_job("A", cpu=10, gpu=50))   # fits Rack 2 (60 GPU free) and Rack 4 (70 free) only
    _, _, _, _, info = env.step(encode_tabular_action(TabularPlacement.PLACE_COOLEST, CoolingCommand.MAINTAIN))
    assert info["action_valid"] and env.dc.workloads["A"].assigned_rack == 1
    env.dc.submit_workload(make_job("B", cpu=10, gpu=5, arrival=env.dc.minute))
    _, _, _, _, info = env.step(encode_tabular_action(TabularPlacement.PLACE_LEAST_UTILISED, CoolingCommand.MAINTAIN))
    assert info["action_valid"] and env.dc.workloads["B"].assigned_rack == 3


def test_tabular_placement_without_feasible_rack_is_invalid(quiet_cfg):
    env = quiet_env(quiet_cfg, representation="tabular")
    env.dc.submit_workload(make_job("BIG", cpu=90, gpu=90))
    _, _, _, _, info = env.step(encode_tabular_action(TabularPlacement.PLACE_COOLEST, CoolingCommand.INCREASE))
    assert not info["action_valid"] and info["invalid_action_reason"] == "NO_FEASIBLE_RACK"
    assert env.dc.cooling.target_level == pytest.approx(0.65)


def test_migration_moves_load_from_hottest_rack(quiet_cfg):
    env = quiet_env(quiet_cfg)
    before = {r.rack_id: (r.get_available_cpu(), r.get_available_gpu()) for r in env.dc.racks}
    check = validate_action(env.dc, env.decode_action(act(env, env.codec.migrate_index)))
    assert check.valid and check.migration.source_rack == 2          # Rack 3 is hottest (28.1 °C)
    job = env.dc.workloads[check.migration.job_id]
    remaining = job.remaining_time
    applied = apply_action(env.dc, check)
    assert applied.migrated and job.assigned_rack == check.migration.target_rack and job.migration_count == 1
    assert job.remaining_time == remaining                           # progress kept
    src, dst = env.dc.racks[2], env.dc.racks[check.migration.target_rack]
    assert src.get_available_cpu() == before[2][0] + job.cpu_requirement
    assert dst.get_available_cpu() == before[dst.rack_id][0] - job.cpu_requirement
    env.dc.check_invariants()


def test_migration_without_cooler_rack_is_invalid(quiet_cfg):
    env = quiet_env(quiet_cfg)
    for rack in env.dc.racks:
        rack.update_temperature(25.0)                                 # no rack is strictly cooler than the hottest
    _, _, _, _, info = env.step(act(env, env.codec.migrate_index))
    assert not info["action_valid"] and info["invalid_action_reason"] == "NO_MIGRATION_CANDIDATE"


def test_failed_simulator_migration_is_atomic(quiet_cfg):
    env = quiet_env(quiet_cfg)
    big = max(env.dc.racks[2].workloads, key=lambda w: w.cpu_requirement)
    env.dc.racks[0].assign_workload(make_job("FILL", cpu=env.dc.racks[0].get_available_cpu()), 0)
    usage = rack_usage(env)
    result = env.dc.migrate_workload(big.job_id, 0)
    assert not result.success and rack_usage(env) == usage and big.assigned_rack == 2


# ---------------------------------------------------------- reproducibility
def run_trajectory(seed: int, actions: list[int]):
    env = DataCenterEnv(episode_minutes=len(actions))
    obs, _ = env.reset(seed=seed)
    trajectory = [obs]
    for a in actions:
        obs, reward, _, _, info = env.step(a)
        trajectory.append((obs, reward, info["action_valid"], info["energy_kwh"], info["rack_temperatures"]))
    return trajectory


def test_same_seed_and_actions_reproduce_trajectory():
    actions = list(np.random.default_rng(0).integers(0, 18, size=90))
    a, b = run_trajectory(42, actions), run_trajectory(42, actions)
    assert np.array_equal(a[0], b[0])
    for x, y in zip(a[1:], b[1:]):
        assert np.array_equal(x[0], y[0]) and x[1:] == y[1:]


def test_different_seeds_give_different_trajectories():
    actions = [act(DataCenterEnv(episode_minutes=1), 4)] * 60
    a, b = run_trajectory(42, actions), run_trajectory(43, actions)
    assert any(x[4] != y[4] for x, y in zip(a[1:], b[1:]))


def test_unseeded_resets_are_reproducible_from_first_seed():
    env = DataCenterEnv(episode_minutes=10)
    env.reset(seed=7)
    seeds_a = [env.reset()[1]["episode_seed"] for _ in range(3)]
    env.reset(seed=7)
    seeds_b = [env.reset()[1]["episode_seed"] for _ in range(3)]
    assert seeds_a == seeds_b and len(set(seeds_a)) == 3


def test_render_and_explanation_are_informative():
    env = DataCenterEnv(episode_minutes=10, render_mode="ansi")
    env.reset(seed=42)
    env.step(act(env, 3))
    text = env.render()
    assert "Time: 08:01" in text and "R3:" in text and "Cooling: target 65%" in text and "Total" in text
    explanation = env.explain_reward()
    for name in ("Energy", "Carbon", "Thermal", "Sla", "Action", "Completion", "Total reward"):
        assert name in explanation
    assert "weight             = 0.3000" in explanation
